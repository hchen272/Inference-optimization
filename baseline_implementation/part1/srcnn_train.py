import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from srcnn_model import SRCNN
from utils.video_io import read_video_frames

class PatchDataset(Dataset):
    def __init__(self, lr_frames, hr_frames, scale, patch_size=33, num_patches=50000):
        self.lr_frames = lr_frames
        self.hr_frames = hr_frames
        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale
        self.num_patches = num_patches
        self.hr_h, self.hr_w = hr_frames[0].shape[:2]
        self.lr_h, self.lr_w = lr_frames[0].shape[:2]
    
    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        # Random frame
        frame_idx = np.random.randint(0, len(self.hr_frames))
        hr = self.hr_frames[frame_idx]
        lr = self.lr_frames[frame_idx]
        # Random HR patch location
        hr_x = np.random.randint(0, self.hr_w - self.patch_size)
        hr_y = np.random.randint(0, self.hr_h - self.patch_size)
        hr_patch = hr[hr_y:hr_y+self.patch_size, hr_x:hr_x+self.patch_size]
        # Corresponding LR patch
        lr_x = hr_x // self.scale
        lr_y = hr_y // self.scale
        lr_patch = lr[lr_y:lr_y+self.lr_patch_size, lr_x:lr_x+self.lr_patch_size]
        # Convert BGR to RGB, then to tensor (C,H,W) and normalize to [0,1]
        hr_patch = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB)
        lr_patch = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2RGB)
        hr_patch = torch.from_numpy(hr_patch).permute(2,0,1).float() / 255.0
        lr_patch = torch.from_numpy(lr_patch).permute(2,0,1).float() / 255.0
        # For SRCNN, we need to bicubic upsample LR patch to HR size first
        # (Better to do in training loop to avoid storing upsampled patches)
        # We'll do upsampling inside the training loop.
        return lr_patch, hr_patch

def train():
    # Configuration
    scale = 2
    batch_size = 32
    epochs = 100
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load video frames
    print("Loading video frames for training...")
    gt_frames = read_video_frames("data/gt_videos/01.mp4")
    lr_frames = read_video_frames("data/input_videos/01.mp4")
    # Ensure same length
    min_len = min(len(gt_frames), len(lr_frames))
    gt_frames = gt_frames[:min_len]
    lr_frames = lr_frames[:min_len]
    
    # Dataset and DataLoader
    dataset = PatchDataset(lr_frames, gt_frames, scale, patch_size=33, num_patches=50000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for lr_patch, hr_patch in loop:
            # lr_patch shape: (B,3,H_lr,W_lr); hr_patch shape: (B,3,H_hr,W_hr)
            # Upsample LR patch to HR size using bicubic (simulate SRCNN input)
            lr_upsampled = torch.nn.functional.interpolate(lr_patch, size=(hr_patch.shape[2], hr_patch.shape[3]), mode='bicubic', align_corners=False)
            lr_upsampled = lr_upsampled.to(device)
            hr_patch = hr_patch.to(device)
            
            optimizer.zero_grad()
            output = model(lr_upsampled)
            loss = criterion(output, hr_patch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        # Save checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/srcnn_x{scale}_epoch{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"models/srcnn_x{scale}.pth")
    print(f"Training completed. Model saved to models/srcnn_x{scale}.pth")

if __name__ == "__main__":
    train()