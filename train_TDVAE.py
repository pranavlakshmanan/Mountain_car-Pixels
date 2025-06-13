#!/usr/bin/env python3
"""
train_TDVAE.py

TD-VAE on MountainCar frames with time-aware regularization and keypoint tracking.
Optimized TensorBoard logging: only key scalars every `log_interval`, flow stats, and sparse dream rollouts every `dream_log_interval`.
"""
import os, argparse, random
import cv2, numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.distributions import Normal, kl_divergence

# -----------------------------
# Data loader
# -----------------------------
class FrameTrajectories:
    def __init__(self, data_dir, img_size):
        files = glob(os.path.join(data_dir, '*.npz'))
        self.trajectories, self.trajectories_raw, self.times = [], [], []
        for f in files:
            data = np.load(f)
            arr = data['frames']
            frames, raws = [], []
            T_max = len(arr)
            for frame in arr:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (img_size, img_size))
                raws.append(gray)
                tensor = torch.from_numpy(gray).unsqueeze(0).float() / 255.0
                frames.append(tensor)
            self.trajectories.append(torch.stack(frames))
            self.trajectories_raw.append(np.stack(raws))
            self.times.append(torch.linspace(0, 1, T_max))
        print(f"Loaded {len(self.trajectories)} trajectories.")

    def sample_pair(self, batch_size, deltas=(1,4,8)):
        samples = []
        for _ in range(batch_size):
            idx = random.randrange(len(self.trajectories))
            traj, raw, time_vec = self.trajectories[idx], self.trajectories_raw[idx], self.times[idx]
            T_max, delta = traj.size(0), random.choice(deltas)
            t1 = random.randint(0, T_max-1)
            t2 = min(t1 + delta, T_max-1)
            samples.append((traj[t1], traj[t2], time_vec[t1], raw[t1], raw[t2]))
        x1 = torch.stack([s[0] for s in samples])
        x2 = torch.stack([s[1] for s in samples])
        tn = torch.stack([s[2] for s in samples]).unsqueeze(1)
        raw1 = np.stack([s[3] for s in samples])
        raw2 = np.stack([s[4] for s in samples])
        return x1, x2, tn, raw1, raw2

# -----------------------------
# Optical flow features
# -----------------------------
def compute_flow_features(raw1, raw2, max_kp=10):
    feats = []
    for i in range(raw1.shape[0]):
        p0 = cv2.goodFeaturesToTrack(raw1[i], maxCorners=max_kp, qualityLevel=0.01, minDistance=5)
        if p0 is None:
            feats.append([0.0, 0.0]); continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(raw1[i], raw2[i], p0, None)
        valid = st.flatten() == 1
        if valid.sum() == 0:
            feats.append([0.0, 0.0]); continue
        flows = (p1[valid] - p0[valid]).reshape(-1, 2)
        mv = flows.mean(axis=0)
        feats.append([float(mv[0]), float(mv[1])])
    return torch.tensor(feats, dtype=torch.float32)

# -----------------------------
# Network definitions
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim, img_size, flow_dim=2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        trunk_out = 128 * (img_size // 8) * (img_size // 8)
        self.head = nn.Linear(trunk_out + flow_dim, latent_dim)
    def forward(self, x, flow_feats=None):
        f = self.trunk(x)
        if flow_feats is not None:
            f = torch.cat([f, flow_feats], dim=1)
        return self.head(f)

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        f = img_size // 8
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128*f*f), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, z):
        f = self.fc(z)
        side = int((f.size(1) / 128)**0.5)
        x = f.view(-1, 128, side, side)
        return self.deconv(x)

class BeliefNet(nn.Module):
    def __init__(self, latent_dim, belief_dim):
        super().__init__(); self.net = nn.Sequential(nn.Linear(latent_dim, belief_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class PriorNet(nn.Module):
    def __init__(self, belief_dim, latent_dim):
        super().__init__(); self.lin = nn.Linear(belief_dim, latent_dim*2)
    def forward(self, b): mu, lv = self.lin(b).chunk(2, -1); return mu, lv

class TransNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__(); self.lin = nn.Linear(latent_dim, latent_dim*2)
    def forward(self, z): mu, lv = self.lin(z).chunk(2, -1); return mu, lv

class PostNet(nn.Module):
    def __init__(self, latent_dim, belief_dim):
        super().__init__(); sz = latent_dim + belief_dim*2; self.lin = nn.Linear(sz, latent_dim*2)
    def forward(self, z2, b1, b2):
        h = torch.cat([z2, b1, b2], -1)
        mu, lv = self.lin(h).chunk(2, -1)
        return mu, lv

# -----------------------------
# Training loop
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--img_size', type=int, default=84)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--belief_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--dream_steps', type=int, default=10)
    parser.add_argument('--kl_anneal_steps', type=int, default=200)
    parser.add_argument('--dyn_weight', type=float, default=1.0)
    parser.add_argument('--var_weight', type=float, default=0.1)
    parser.add_argument('--clock_weight', type=float, default=1.0)
    parser.add_argument('--flow_keypoints', type=int, default=10)
    parser.add_argument('--dream_log_interval', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/tdvae')
    data = FrameTrajectories(args.data_dir, args.img_size)

    enc = Encoder(args.latent_dim, args.img_size).to(dev)
    dec = Decoder(args.latent_dim, args.img_size).to(dev)
    bel = BeliefNet(args.latent_dim, args.belief_dim).to(dev)
    pr  = PriorNet(args.belief_dim, args.latent_dim).to(dev)
    tr  = TransNet(args.latent_dim).to(dev)
    po  = PostNet(args.latent_dim, args.belief_dim).to(dev)

    params = list(enc.parameters()) + list(dec.parameters()) + \
             list(bel.parameters()) + list(pr.parameters()) + \
             list(tr.parameters()) + list(po.parameters())
    opt = optim.Adam(params, lr=args.lr)

    for step in range(1, args.steps+1):
        x1, x2, tn, raw1, raw2 = data.sample_pair(args.batch_size)
        x1, x2, tn = x1.to(dev), x2.to(dev), tn.to(dev)
        flow = compute_flow_features(raw1, raw2, args.flow_keypoints).to(dev)

        # Encode
        z1 = enc(x1, flow); z2 = enc(x2, flow)
        b1 = bel(z1); b2 = bel(z2)
        # Log latent distribution histogram
        if step % args.log_interval == 0:
            writer.add_histogram('latent_new', z1, step)

        # Posterior q(z1|z2,b1,b2)
        mu_q1_raw, lv_q1_raw = po(z2, b1, b2)
        mu_q1 = mu_q1_raw.clamp(-10,10)
        lv_q1 = F.softplus(lv_q1_raw.clamp(-10,10))
        q1 = Normal(mu_q1, (0.5*lv_q1).exp().clamp(1e-3,1e3))

        # Prior p(z1|b1)
        mu_p1_raw, lv_p1_raw = pr(b1)
        mu_p1 = mu_p1_raw.clamp(-10,10)
        lv_p1 = F.softplus(lv_p1_raw.clamp(-10,10))
        p1 = Normal(mu_p1, (0.5*lv_p1).exp().clamp(1e-3,1e3))
        kl1 = kl_divergence(q1,p1).mean()

        # Transition p(z2|z1)
        z1s = q1.rsample()
        mu_tr_raw, lv_tr_raw = tr(z1s)
        mu_tr = mu_tr_raw.clamp(-10,10)
        lv_tr = F.softplus(lv_tr_raw.clamp(-10,10))
        p2 = Normal(mu_tr, (0.5*lv_tr).exp().clamp(1e-3,1e3))
        kl2 = kl_divergence(p2, Normal(z2.detach(), torch.ones_like(z2)*1e-3)).mean()

        # Reconstruction
        x2r = dec(z2)
        if x2r.shape[-1]!=args.img_size or x2r.shape[-2]!=args.img_size:
            x2r = F.interpolate(x2r, size=(args.img_size,args.img_size), mode='bilinear', align_corners=False)
        recon = F.mse_loss(x2r, x2)

        # Additional losses
        dyn_loss   = kl2
        var_loss   = -torch.log(z1.var(0).mean() + 1e-6)
        clock_loss = F.mse_loss(z1[:,0].unsqueeze(1), tn)

        # Total loss
        beta = min(1.0, step/args.kl_anneal_steps)
        loss = recon + beta*(kl1+kl2) + args.dyn_weight*dyn_loss + args.var_weight*var_loss + args.clock_weight*clock_loss

        # Backprop
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()

        # Log scalars + recon every log_interval
        if step % args.log_interval == 0:
            writer.add_scalar('reconsut-new/recon', recon.item(), step)
            writer.add_scalar('latent_new/kl1', kl1.item(), step)
            writer.add_scalar('latent_new/kl2', kl2.item(), step)
            writer.add_scalar('latent_new/dynamics', dyn_loss.item(), step)
            writer.add_scalar('latent_new/variance', var_loss.item(), step)
            writer.add_scalar('latent_new/clock', clock_loss.item(), step)
            writer.add_scalar('latent_new/beta', beta, step)
            writer.add_scalar('flow_new/mean_magnitude', torch.norm(flow,1,1).mean().item(), step)
            # Reconstruction image
            with torch.no_grad():
                gt = x2[:4]; pd = x2r[:4]
                inter = torch.stack([gt,pd],1).view(-1,1,args.img_size,args.img_size)
                writer.add_image('reconst-new/grid', make_grid(inter,nrow=2), step)

        # Sparse dream rollouts every dream_log_interval
        if step % args.dream_log_interval == 0 or step==args.steps:
            with torch.no_grad():
                bcur = bel(z1[:1])
                dreams = []
                for _ in range(args.dream_steps):
                    mu_d_raw, lv_d_raw = pr(bcur)
                    mu_d = mu_d_raw.clamp(-10,10)
                    lv_d = F.softplus(lv_d_raw.clamp(-10,10))
                    sd_d = (0.5*lv_d).exp().clamp(1e-3,1e3)
                    dist = Normal(mu_d, sd_d)
                    zd = dist.rsample()
                    img = dec(zd)
                    img = F.interpolate(img, size=(args.img_size,args.img_size), mode='bilinear', align_corners=False)
                    dreams.append(img)
                    bcur = bel(zd)
                grid = make_grid(torch.cat(dreams,0), nrow=args.dream_steps)
                writer.add_image('dream_new/rollout', grid, step)

    print('Done.')
