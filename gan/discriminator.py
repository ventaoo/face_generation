import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),           # 64x64 -> 32x32
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),          # 32x32 -> 16x16
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),          # 16x16 -> 8x8
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0)             # 8x8 -> 1x1 (输出score)
            # 注意：无Sigmoid层！
        )

    def forward(self, img):
        return self.model(img).view(-1)  # 展平为1D分数

