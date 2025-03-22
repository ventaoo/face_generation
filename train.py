import os

import torch
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_fid import fid_score
from torchmetrics.image.inception import InceptionScore

from crop_util import plot_training_curves

# 训练函数封装
def train_wgan(
    G, D, 
    train_loader, 
    opt_g, opt_d,
    device,
    n_epochs=100,
    latent_dim=100,
    n_critic=5,
    lambda_gp=10,      # 梯度惩罚系数
    use_gp=True,        # 是否使用梯度惩罚
    eval_interval=5,    # 评估间隔
    sample_interval=10, # 采样间隔
):
    # 创建日志字典
    history = {
        'g_loss': [],
        'd_loss': [],
        'fid': [],
        'is_score': [],
        'epochs': []
    }
    
    # 初始化评估指标
    inception_score = InceptionScore().to(device)
    
    # 训练循环
    for epoch in range(n_epochs):
        G.train()
        D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, real_data in enumerate(pbar):
            # real_label 用于在条件生成的时候使用
            real_imgs, real_label = real_data
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # ========================
            #  训练判别器 (Critic)
            # ========================
            d_losses = []
            for _ in range(n_critic):
                # 生成假图像
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = G(z).detach()
                
                # 计算判别器损失
                real_scores = D(real_imgs)
                fake_scores = D(fake_imgs)
                d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
                
                # 梯度惩罚 (WGAN-GP)
                if use_gp:
                    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
                    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
                    d_interpolates = D(interpolates)
                    
                    gradients = torch.autograd.grad(
                        outputs=d_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(d_interpolates).to(device),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    gradients = gradients.view(gradients.size(0), -1)
                    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    d_loss += lambda_gp * gp
                
                # 反向传播
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_losses.append(d_loss.item())
            
            # ========================
            #  训练生成器
            # ========================
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = G(z)
            g_loss = -torch.mean(D(fake_imgs))
            
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            
            # 记录损失
            epoch_g_loss += g_loss.item()
            epoch_d_loss += np.mean(d_losses)
            
            # 更新进度条
            pbar.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': np.mean(d_losses)
            })
            
            # 保存样本图像
            os.makedirs('./samples', exist_ok=True)
            if batch_idx % sample_interval == 0:
                save_image(
                    fake_imgs[:16], 
                    f"samples/epoch_{epoch}_batch_{batch_idx}.png",
                    nrow=4, 
                    normalize=True
                )
        
        # 计算epoch平均损失
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        history['g_loss'].append(epoch_g_loss)
        history['d_loss'].append(epoch_d_loss)
        
        # ========================
        #  评估指标
        # ========================
        if (epoch+1) % eval_interval == 0:
            G.eval()
            # 生成评估样本
            all_samples = []
            with torch.no_grad():
                for _ in range(10):  # 生成1000个样本
                    z = torch.randn(100, latent_dim).to(device)
                    samples = G(z)
                    all_samples.append(samples)
                all_samples = torch.cat(all_samples, dim=0)
            
            # 计算IS
            inception_score.update(all_samples)
            is_mean, is_std = inception_score.compute()
            
            # 计算FID（需要真实图像统计量）
            # 需提前计算真实图像的mu, sigma并保存为npz文件
            fid = fid_score.calculate_fid_given_samples(
                'real_stats.npz',
                all_samples.cpu().numpy(),
                device=device,
                batch_size=100
            )
            
            history['fid'].append(fid)
            history['is_score'].append(is_mean.item())
            history['epochs'].append(epoch)
            
            print(f"\nEpoch {epoch+1} | FID: {fid:.2f} | IS: {is_mean:.2f}±{is_std:.2f}")
            
            # 保存模型检查点
            torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch}.pth")
            torch.save(D.state_dict(), f"checkpoints/D_epoch_{epoch}.pth")
            
        # 绘制训练曲线
        plot_training_curves(history)
    
    return history