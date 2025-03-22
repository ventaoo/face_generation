import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def crop_faces(input_dir, output_dir, image_male_dict, target_size=(128, 128), ratio=0.5):
    np.random.seed(703)
    
    # 选择 GPU 或 CPU
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    
    # 初始化 MTCNN（比原版 MTCNN 更快）
    detector = MTCNN(keep_all=False, device=device)  # 只检测单张最大人脸

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'male'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'female'), exist_ok=True)

    images = os.listdir(input_dir)

    def process_image(img_name):
        """ 处理单张图片，进行人脸检测、裁剪、调整大小并保存 """
        img_path = os.path.join(input_dir, img_name)
        key = os.path.basename(img_path)

        # 读取图片
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Failed to load {img_name}")
            return None  # 读取失败，跳过
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测人脸（返回 bounding box）
        boxes, _ = detector.detect(img)
        if boxes is None:
            print(f"No face detected in {img_name}")
            return None  # 未检测到人脸
        
        # 选择最大人脸
        best_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))  
        x1, y1, x2, y2 = map(int, best_box)

        # 扩展边界框避免裁剪过紧
        padding = 0.2
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - padding * w))
        y1 = max(0, int(y1 - padding * h))
        x2 = min(img.shape[1], int(x2 + padding * w))
        y2 = min(img.shape[0], int(y2 + padding * h))

        # 裁剪人脸并调整大小
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, target_size)

        # 确定性别并保存
        gender = 'male' if image_male_dict.get(key) == 1 else 'female'
        output_path = os.path.join(output_dir, gender, img_name)
        cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

    # 多线程加速
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_image, np.random.choice(images, int(len(images) * ratio))), total=int(len(images) * ratio)))


def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(131)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # FID曲线
    plt.subplot(132)
    plt.plot(history['epochs'], history['fid'], 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    
    # IS曲线
    plt.subplot(133)
    plt.plot(history['epochs'], history['is_score'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Inception Score')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()