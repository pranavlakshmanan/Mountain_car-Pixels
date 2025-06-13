"""
mountain_car_env.py

Wrapper for MountainCar-v0 with preprocessing.

Usage:
  # Run built-in tests (no GUI):
  python mountain_car_env.py

  # View live MountainCar-v0 environment window:
  python mountain_car_env.py --demo
"""
import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any

class MountainCarEnv:
    """
    Wrapper for MountainCar-v0 environment with preprocessing.
    If return_rgb=True, returns raw RGB frames; else returns preprocessed 84×84 grayscale.
    """
    def __init__(
        self,
        return_rgb: bool = False,
        render_mode: str = "rgb_array"
    ):
        self.return_rgb = return_rgb
        self._render_mode = render_mode
        # Initialize Gym environment
        self.env = gym.make(
            "MountainCar-v0",
            render_mode=render_mode
        )
        self.action_space = self.env.action_space
        print(f"Action space: {self.action_space}")
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84),
            dtype=np.uint8
        )

    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        if self._render_mode == "human":
            # In human mode, env.render() handles display
            return state, info
        frame = self.env.render()
        obs = frame if self.return_rgb else self.preprocess_observation(frame)
        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if self._render_mode == "human":
            self.env.render()
            return state, reward, terminated, truncated, info
        frame = self.env.render()
        obs = frame if self.return_rgb else self.preprocess_observation(frame)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

if __name__ == '__main__':
    import argparse, sys, unittest

    parser = argparse.ArgumentParser(
        description="MountainCarEnv test runner or demo"
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run a live human-rendered MountainCar demo'
    )
    args, remaining = parser.parse_known_args()

    if args.demo:
        # Live human render demo
        env = MountainCarEnv(return_rgb=False, render_mode='human')
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()
    else:
        # Run unit tests (no GUI)
        unittest.main(argv=[sys.argv[0]] + remaining)
