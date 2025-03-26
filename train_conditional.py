import os

import torch
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_fid import fid_score

from crop_util import plot_training_curves, calculate_inception_score

# 训练函数封装
def train_wgan_conditional(
    G, D, 
    train_loader, 
    opt_g, opt_d,
    device,
    n_epochs=100,
    latent_dim=128,
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
    
    # 训练循环
    for epoch in range(n_epochs):
        G.train()
        D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        fixed_z = torch.randn(16, latent_dim).to(device)  # 64 个固定噪声样本
        fixed_labels = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]).to(device)
        for batch_idx, real_data in enumerate(pbar):
            # real_label 用于在条件生成的时候使用
            real_imgs, real_labels = real_data
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_imgs.size(0)
            
            # ========================
            #  训练判别器 (Critic)
            # ========================
            d_losses = []
            for _ in range(n_critic):
                # 生成假图像
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_labels = torch.randint(0, 2, (batch_size,)).to(device)
                fake_imgs = G(z, fake_labels).detach()
                
                # 计算判别器损失
                real_scores = D(real_imgs, real_labels)
                fake_scores = D(fake_imgs, real_labels)

                # print(f'fake labels: {fake_labels} | real labels: {real_labels}')
                # print(f"real shape: {real_imgs.shape} | fake shape: {fake_imgs.shape}")
                # print(f"Min value: {real_imgs.min().item()}, Max value: {real_imgs.max().item()}")
                # print(f"Min value: {fake_imgs.min().item()}, Max value: {fake_imgs.max().item()}")
                # print(f'real: {torch.mean(real_scores)} | fake: {torch.mean(fake_scores)} | {torch.mean(real_scores) - torch.mean(fake_scores)}')

                d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
                
                # 梯度惩罚 (WGAN-GP)
                if use_gp:
                    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
                    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
                    d_interpolates = D(interpolates, real_labels)
                    
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
            fake_labels = torch.randint(0, 2, (batch_size,)).to(device)  # 假标签
            fake_imgs = G(z, fake_labels)  # 生成器需要标签作为输入
            g_loss = -torch.mean(D(fake_imgs, fake_labels))
            
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
            if batch_idx % sample_interval == 0:
                os.makedirs('./examples', exist_ok=True)
                with torch.no_grad():
                    fixed_fake_imgs = G(fixed_z, fixed_labels)  # 用固定的噪声生成图像
                save_image(
                    fixed_fake_imgs, 
                    f"examples/epoch_{epoch}_batch_{batch_idx}.png",
                    nrow=8, 
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
            is_mean, is_std = calculate_inception_score(G, device, latent_dim, num_samples=1000, batch_size=100, is_conditional=True)
            
            
            # # 计算FID（需要真实图像统计量, 需提前计算真实图像的mu, sigma并保存为npz文件
            fid = fid_score.calculate_fid_given_paths(
                ['./crop_img', './samples'],
                device=device,
                batch_size=100,
                dims=2048
            )
            
            history['fid'].append(fid)
            history['is_score'].append(is_mean)
            history['epochs'].append(epoch)
            
            print(f"\nEpoch {epoch+1} | FID: {fid:.2f} | IS: {is_mean:.2f}±{is_std:.2f}")
            
            # 保存模型检查点
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch}.pth")
            torch.save(D.state_dict(), f"checkpoints/D_epoch_{epoch}.pth")
            
        # 绘制训练曲线
        plot_training_curves(history)
    
    return history