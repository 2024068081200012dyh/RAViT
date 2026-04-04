import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pycocotools.coco import COCO

class LVISDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_id