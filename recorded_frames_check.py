import numpy as np
with np.load('recorded_frames/episode_010.npz') as data:
    print(data.files)          # ['frames']
    frames = data['frames']    # shape (T, H, W, C)
    print(frames.shape)
