import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.init_size = 4  # 初始特征图尺寸
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size ** 2),
            nn.BatchNorm1d(512 * self.init_size ** 2),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 32x32 -> 64x64
            nn.Tanh()  # 输出归一化到[-1,1]
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
class Generator_conditional(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, num_classes=2, embedding_dim=128):
        super().__init__()
        self.init_size = 4  # 初始特征图尺寸
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size ** 2),
            nn.BatchNorm1d(512 * self.init_size ** 2),
            nn.LeakyReLU(0.2)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 32x32 -> 64x64
            nn.Tanh()  # 输出归一化到[-1,1]
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        z = label_embedding + z
        out = self.fc(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
if __name__ == '__main__':
    z = torch.randn(2, 128)
    labels = torch.randint(0, 2, (2,))  # 随机类别
    print(labels)
    G = Generator()
    G_conditional = Generator_conditional()