import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, 'r') as f:
            for line in f:
                items = line.split()
                image_name = os.path.join(data_dir, items[0])
                image_names.append(image_name)
                label = [int(i) for i in items[1:]]
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
