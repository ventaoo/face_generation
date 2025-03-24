import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0)),
        )

    def forward(self, img):
        return self.model(img).view(-1)  # 展平为1D分数


if __name__ == "__main__":
    import torch
    # 创建一个批量大小为 1 的随机输入图像 (batch_size, channels, height, width)
    img = torch.randn(1, 3, 64, 64)

    # 初始化 Discriminator
    discriminator = Discriminator()

    # 运行前向传播
    output = discriminator(img)

    # 打印输出形状
    print("Output shape:", output.shape)
