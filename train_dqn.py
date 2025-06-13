# train_dqn.py
# DQN on MountainCar-v0 with optional live rendering and TensorBoard logging

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from mountain_car_env import MountainCarEnv
import gymnasium as gym

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Determine flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_features = self.feature_extractor(dummy).shape[1]
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.fc(x)


def train(train_env, args, render_env=None):
    writer = SummaryWriter(log_dir=args.log_dir)

    # Initialize networks
    input_shape = (1, 84, 84)
    num_actions = train_env.action_space.n
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay_buffer = deque(maxlen=args.buffer_size)
    steps_done = 0

    for episode in range(args.episodes):
        # Reset both training and rendering envs
        obs, _ = train_env.reset()
        if render_env:
            render_env.reset()
            render_env.render()
        state = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        total_reward = 0.0
        losses = []

        for t in range(args.max_steps):
            # Epsilon-greedy action
            eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-steps_done / args.eps_decay)
            steps_done += 1
            if np.random.rand() < eps:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(state)
                    action = q_vals.argmax(dim=1).item()

            # Step training env
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            next_state = torch.from_numpy(next_obs.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            total_reward += reward

            # Step render env if requested
            if render_env:
                render_env.step(action)
                render_env.render()

            # Store transition
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # Optimize when enough samples
            if len(replay_buffer) >= args.batch_size:
                batch_idxs = np.random.choice(len(replay_buffer), args.batch_size, replace=False)
                batch = [replay_buffer[idx] for idx in batch_idxs]
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states, dim=0)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, device=device).unsqueeze(1)
                next_states = torch.cat(next_states, dim=0)
                dones = torch.tensor(dones, device=device).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0].unsqueeze(1)
                    q_target = rewards + args.gamma * q_next * (~dones)

                loss = nn.functional.mse_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                writer.add_scalar('Loss/Q', loss.item(), steps_done)

            if done:
                break

        # Log episode stats
        writer.add_scalar('Reward/Episode', total_reward, episode)
        if losses:
            writer.add_scalar('Loss/Avg', np.mean(losses), episode)

        # Update target net
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Reward: {total_reward:.2f}, AvgLoss: {np.mean(losses) if losses else 0:.4f}")

    # Cleanup
    train_env.close()
    if render_env:
        render_env.close()
    torch.save(policy_net.state_dict(), "dqn_mountaincar.pth")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DQN on MountainCar-v0 with optional rendering and TensorBoard")
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=10000.0)
    parser.add_argument('--target_update', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='runs/mountaincar')
    parser.add_argument('--render', action='store_true', help='Display environment during training')
    args = parser.parse_args()

    # Create training env (pixel-based)
    train_env = MountainCarEnv(return_rgb=False, render_mode='rgb_array')
    # Optionally, a separate human-rendering env
    render_env = None
    if args.render:
        render_env = gym.make("MountainCar-v0", render_mode='human')
    train(train_env, args, render_env)
