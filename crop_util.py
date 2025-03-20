import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

def crop_faces(input_dir, output_dir, target_size=(128, 128), ratio=0.5):
    np.random.seed(703)
    
    detector = MTCNN()
    # Male Female
    os.makedirs(os.path.join(output_dir, 'male'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'female'), exist_ok=True)
    images = os.listdir(input_dir)
    
    for img_name in tqdm(np.random.choice(images, int(len(images) * ratio))):
        img_path = os.path.join(input_dir, img_name)
        key = os.path.basename(img_path)
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # 检测人脸
        results = detector.detect_faces(img)
        if not results:
            print(f"No face detected in {img_name}")
            continue
        
        # 提取最大的人脸区域
        max_area = 0
        best_box = None
        for res in results:
            x, y, w, h = res['box']
            area = w * h
            if area > max_area:
                max_area = area
                best_box = (x, y, w, h)
        
        # 扩展边界框避免裁剪过紧
        x, y, w, h = best_box
        padding = 0.2  # 扩展20%区域
        x = max(0, int(x - padding * w))
        y = max(0, int(y - padding * h))
        w = int(w * (1 + 2*padding))
        h = int(h * (1 + 2*padding))
        
        # 裁剪并调整尺寸
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        
        # 保存结果
        output_path = os.path.join(os.path.join(output_dir, 'male' if image_male_dict.get(key) == 1 else 'female'), img_name)
        cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))