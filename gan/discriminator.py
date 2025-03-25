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
            spectral_norm(nn.Conv2d(512, 1, 4, 3, 0)),
        )

    def forward(self, img):
        return self.model(img).view(-1)  # 展平为1D分数

class Discriminator_conditional(nn.Module):
    def __init__(self, img_channels=3, num_classes=2, embedding_dim=512):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2)
        )

        self.label_processing = nn.Sequential(
            nn.Conv2d(512, 1, 4, 3, 0)
        )

    def forward(self, img, labels):
        label_embedding = self.label_embedding(labels) # [1, 512]
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)
        img = self.model(img) # [1, 512, 4, 4]
        out = self.label_processing(img * label_embedding)
        return out.view(-1)  # 展平为1D分数

if __name__ == "__main__":
    import torch
    img = torch.randn(10, 3, 64, 64)
    labels = torch.randint(0, 2, (10,))  # 随机类别

    # 初始化 Discriminator
    discriminator = Discriminator()
    discriminator_conditional = Discriminator_conditional()

    # 运行前向传播
    output = discriminator(img)

    # 打印输出形状
    print(discriminator_conditional(img, labels))


    from generator import Generator_conditional
    z = torch.randn(10, 128)
    labels = torch.randint(0, 2, (10,))  # 随机类别
    G_conditional = Generator_conditional()
    fake_img = G_conditional(z, labels)
    
    print(f"Min value: {fake_img.min().item()}, Max value: {fake_img.max().item()}")

    print(discriminator_conditional(fake_img, labels))
