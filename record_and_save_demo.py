#!/usr/bin/env python3
"""
record_and_save_demo.py

Collects frames from a MountainCar-v0 episode using rgb_array rendering (software fallback) and saves a GIF.

Usage:
  python record_and_save_demo.py

Output:
  mountaincar_demo.gif in the current directory
"""
import os
# Force software OpenGL to avoid driver issues
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import gymnasium as gym
import imageio

def main():
    # Create environment in rgb_array mode
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    # Collect first frame
    frame = env.render()  # numpy array
    frames.append(frame)

    done = False
    while not done:
        # Random policy for demo
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frame = env.render()
        frames.append(frame)

    env.close()

    # Save frames as a GIF
    gif_path = "mountaincar_demo.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved demo GIF to {gif_path}")

if __name__ == "__main__":
    main()
