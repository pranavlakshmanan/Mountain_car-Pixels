#!/usr/bin/env python3
"""
train_and_record.py

Extended DQN training & per-episode frame saving with reduced memory via uint8 replay.
Runs MountainCar-v0 from pixels, saves full episodes of frames, logs to TensorBoard.

Usage example:
  python train_and_record.py \
    --episodes 100 --max_steps 10000 --image_size 84 --render --device cuda
  tensorboard --logdir runs/train_and_record
"""
import os, random, argparse, numpy as np
from collections import deque
from PIL import Image
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mountain_car_env import MountainCarEnv

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buf)
    def push(self, *args):
        self.buf.append(tuple(args))
    def sample(self, n):
        batch = random.sample(self.buf, n)
        return map(np.stack, zip(*batch))

class DQN(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        c, h, w = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            conv_out = self.conv(x).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def softmax_action(q_vals, temp):
    probs = torch.softmax(q_vals / temp, -1).cpu().numpy().ravel()
    return np.random.choice(len(probs), p=probs)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes',    type=int,   default=100)
    p.add_argument('--max_steps',   type=int,   default=10000)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--buffer_size', type=int,   default=20000)
    p.add_argument('--gamma',       type=float, default=0.99)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--warmup_steps',type=int,   default=1000)
    p.add_argument('--target_update',type=int,  default=1000)
    p.add_argument('--alpha',       type=float, default=0.1)
    p.add_argument('--eps_start',   type=float, default=1.0)
    p.add_argument('--eps_end',     type=float, default=0.05)
    p.add_argument('--eps_decay',   type=float, default=100000)
    p.add_argument('--temp_start',  type=float, default=1.0)
    p.add_argument('--temp_end',    type=float, default=0.1)
    p.add_argument('--temp_decay',  type=float, default=100000)
    p.add_argument('--image_size',  type=int,   default=84)
    p.add_argument('--render',      action='store_true')
    p.add_argument('--device',      type=str,   default='cuda')
    args = p.parse_args()

    print(f"[Debug] max_steps={args.max_steps}")
    os.makedirs('recorded_frames', exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/train_and_record')

    # Environments
    env = MountainCarEnv(return_rgb=False, render_mode='rgb_array')
    rec_env = MountainCarEnv(return_rgb=True, render_mode='rgb_array')
    from gymnasium.wrappers import TimeLimit
    env.env     = TimeLimit(env.env, args.max_steps)
    rec_env.env = TimeLimit(rec_env.env, args.max_steps)
    render_env = None
    if args.render:
        render_env = gym.make('MountainCar-v0', render_mode='human')
        render_env = TimeLimit(render_env, args.max_steps)

    # Networks and replay
    in_shape   = (4, args.image_size, args.image_size)
    n_actions  = env.action_space.n
    policy_net = DQN(in_shape, n_actions).to(device)
    target_net = DQN(in_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
    optimizer  = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay_buf = ReplayBuffer(args.buffer_size)

    global_step = 0
    for ep in range(1, args.episodes+1):
        obs, _ = env.reset()
        rec_env.reset()
        if render_env: render_env.reset()

        # Initialize frame stack
        frame0 = Image.fromarray(obs).resize((args.image_size, args.image_size))
        gray0  = np.array(frame0, dtype=np.uint8)
        dq     = deque([gray0]*4, maxlen=4)
        state  = torch.tensor(np.stack(dq)/255.0, dtype=torch.float32, device=device).unsqueeze(0)

        total_r = 0.0
        pos_prev = env.env.env.state[0]
        recorded = []

        for t in range(args.max_steps):
            temp = args.temp_end + (args.temp_start - args.temp_end) * np.exp(-global_step/args.temp_decay)
            eps  = args.eps_end  + (args.eps_start  - args.eps_end)  * np.exp(-global_step/args.eps_decay)
            with torch.no_grad(): qv = policy_net(state)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = softmax_action(qv, temp)

            # Step envs
            ob2, r, done, _, _ = env.step(action)
            shaped = r + args.alpha * max(0, env.env.env.state[0] - pos_prev)
            pos_prev = env.env.env.state[0]
            rec_frame, *_ = rec_env.step(action)
            recorded.append(rec_frame)
            if render_env:
                render_env.step(action); render_env.render()

            # Build next state
            gray2 = np.array(
                Image.fromarray(ob2).resize((args.image_size, args.image_size)),
                dtype=np.uint8
            )
            dq.append(gray2)
            next_state = torch.tensor(np.stack(dq)/255.0, dtype=torch.float32, device=device).unsqueeze(0)

            # Push to replay as uint8
            s_uint8  = (state.squeeze(0).cpu().numpy()  * 255).astype(np.uint8)
            ns_uint8 = (next_state.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            replay_buf.push(s_uint8, action, shaped, ns_uint8, done)

            state = next_state
            total_r += shaped

            # Learning
            if len(replay_buf) >= args.batch_size and global_step > args.warmup_steps:
                s_b, a_b, r_b, ns_b, d_b = replay_buf.sample(args.batch_size)
                # decompress to float32
                s_b  = torch.tensor(s_b.astype(np.float32)/255.0, device=device)
                ns_b = torch.tensor(ns_b.astype(np.float32)/255.0, device=device)
                a_b  = torch.tensor(a_b, dtype=torch.int64, device=device).unsqueeze(1)
                r_b  = torch.tensor(r_b, dtype=torch.float32, device=device).unsqueeze(1)
                d_b  = torch.tensor(d_b, dtype=torch.float32, device=device).unsqueeze(1)

                q_curr = policy_net(s_b).gather(1, a_b)
                with torch.no_grad():
                    q_next = target_net(ns_b).max(1)[0].unsqueeze(1)
                q_target = r_b + args.gamma * q_next * (1.0 - d_b)
                loss = F.mse_loss(q_curr, q_target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                writer.add_scalar('Loss/Q', loss.item(), global_step)
                if global_step % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            global_step += 1
            if done:
                break

        # Save full episode frames
        np.savez_compressed(f"recorded_frames/ep_{ep:03d}.npz",
                             frames=np.stack(recorded))
        recorded.clear()

        writer.add_scalar('Reward/Episode', total_r, ep)
        writer.add_scalar('Frames/Episode', len(recorded), ep)
        print(f"Ep{ep:03d} | R {total_r:.1f} | Steps {t+1} | Eps {eps:.3f}")

    writer.close()
    torch.save(policy_net.state_dict(), 'dqn_quantized.pth')
    print("Done.")
