import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    SRCNN for image super-resolution.
    Architecture: Conv(9x9) + ReLU -> Conv(1x1) + ReLU -> Conv(5x5)
    Input: 3-channel RGB image (normalized to [0,1])
    Output: 3-channel RGB image (same range)
    """
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x