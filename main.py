import os
import shutil
import argparse

import torch
import kagglehub
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.dataset import CelebADataset
from gan.discriminator import Discriminator
from gan.generator import Generator
from train import train_wgan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditional GAN Training on CelebA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值帮助
    )
    
    # 必需参数
    parser.add_argument("--crop_path", type=str, required=True,
                       help="Path to preprocessed cropped face images")
    parser.add_argument("--attr_path", type=str, required=True,
                       help="Path to CelebA attributes CSV file")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to raw CelebA dataset root directory")

    # 训练参数
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],  # 限制可选值
                       help="Training device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="Number of training epochs")
    
    # 优化器参数
    parser.add_argument("--lr_g", type=float, default=1e-5,
                       help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=1e-4,
                       help="Discriminator learning rate")
    
    # 训练监控
    parser.add_argument("--eval_interval", type=int, default=5,
                       help="Evaluation interval (epochs)")
    parser.add_argument("--sample_interval", type=int, default=100,
                       help="Sample image save interval (batches)")
    
    # 模型配置
    parser.add_argument("--latent_dim", type=int, default=100,
                       help="Dimension of latent space")
    parser.add_argument("--use_gp", action="store_true",
                       help="Enable gradient penalty (WGAN-GP)")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                       help="Gradient penalty coefficient")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data_path):
        # Download latest version
        path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        print("Path to dataset files:", path)
        shutil.move(path, './')
    
    G = Generator().to(args.device)
    D = Discriminator().to(args.device)
    opt_g = torch.optim.RMSprop(G.parameters(), lr=args.lr_g)
    opt_d = torch.optim.RMSprop(D.parameters(), lr=args.lr_d)

    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
    ])

    train_dataset = CelebADataset(args.data_path, args.crop_path, args.attr_path, transform, ratio=0.1)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    # 启动训练
    history = train_wgan(
        G, D, train_loader,
        opt_g, opt_d, args.device,
        n_epochs=args.n_epochs,
        use_gp=True,
        eval_interval=args.eval_interval,
        sample_interval=args.sample_interval,
        n_critic=7
    )

    # python main.py --crop_path ./crop_img --attr_path ./2/list_attr_celeba.csv --data_path ./2/img_align_celeba/img_align_celeba --device cuda --batch_size 256 --lr_g 1e-4 --lr_d 2e-4 --n_epochs 150 --use_gp --eval_interval 5
    # python main.py --crop_path ./crop_img --attr_path ./2/list_attr_celeba.csv --data_path ./2/img_align_celeba/img_align_celeba --device mps --batch_size 128 --lr_g 2e-4 --lr_d 2e-4 --n_epochs 150 --use_gp --eval_interval 5



# TODO
# 测试是否可以正常加载模型进行推理
# ✅固定测试的代码的
# 查看指标是否计算正确
# ✅查看是否可以优化