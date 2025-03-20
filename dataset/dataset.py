import os
from PIL import Image
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, images, attrs, transform):
        """
        images: 图像路径
        attrs: 文件名和属性的配对 key : 文件名 value : 属性
        transform: 数据增强
        """
        
        self.images = images
        self.attrs = attrs
        self.transform = transform
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        key = os.path.basename(image)
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, self.attrs.get(key)