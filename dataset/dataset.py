import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from crop_util import crop_faces

class CelebADataset(Dataset):
    def __init__(self, image_path, crop_img_path, attr_path, transform, ratio=1):
        """
        :parm image_path 原始图片的路径
        :parm crop_img_path
        :parm attr_path 标签信息的csv文件的路径
        :transform 数据增强
        """
        self.transform = transform
        self.crop_img_path = crop_img_path
        self.attr_dataframe = pd.read_csv(attr_path)
        self.image_male_dict = self.attr_dataframe.set_index('image_id')['Male'].to_dict()

        # 不存在则处理原始图片并保存
        if not os.path.exists(crop_img_path): 
            crop_faces(image_path, crop_img_path, self.image_male_dict, ratio=ratio)
        else: print('Dataset crop done.')

        self.images = []
        for dir_path, _, filenames in os.walk(self.crop_img_path):
            for file in tqdm(filenames):
                self.images.append(os.path.join(dir_path, file))
        print(f"Len of the images: {len(self.images)}")
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        key = os.path.basename(image)
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, self.image_male_dict.get(key)