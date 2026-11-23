import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import io

class Vocabulary():
    def __init__(self, img_dir):
        self.label = {}
        self.idx = []
        for dr in sorted(os.listdir(img_dir)):
            self.label[dr] = len(self.idx)
            self.idx.append(dr)

    def __getitem__(self, idx):
        return self.idx[idx]

    def label_to_idx(self, label):
        return self.label[label]

class training_dataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        super().__init__()
        self.dir = img_dir
        self.transforms = transforms
        self.indices = []
        self.label = []
        count=0
        for dr in os.listdir(img_dir):
            for img in os.listdir(f"{img_dir}/{dr}"):
                self.indices.append(f"{img_dir}/{dr}/{img}")
                self.label.append(count)
            count+=1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img =  io.read_image(self.indices[idx], io.ImageReadMode.RGB)/255.0
        if self.transforms:
            img = self.transforms(img)
        return {
                "img": img,
                "label_idx": self.label[idx]
                }

class testing_dataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        super().__init__()
        self.dir = img_dir
        self.transforms=transforms
        self.size = len(os.listdir(img_dir))
        self.indices = []
        self.img_name = []
        for img in sorted(os.listdir(self.dir)):
            self.indices.append(f"{self.dir}/{img}")
            self.img_name.append(img)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img =  io.read_image(self.indices[idx], io.ImageReadMode.RGB)/255.0
        if self.transforms:
            img = self.transforms(img)
        return img

def Collate_fn(batch, train=True):
    if not train:
        return {
                "img": torch.concat([img.unsqueeze(0) for img in batch], dim=0)
                }
    img, label = [], []
    for X in batch:
        img.append(X["img"].unsqueeze(0))
        label.append(X["label_idx"])
    return {
            "img": torch.concat(img, dim=0),
            "label_idx": torch.tensor(label)
            }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomCrop((224, 224), pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5)
        ])
    testing_dataset = training_dataset("../val", transform)
    vocabulary = Vocabulary(".././train")
    torch.save(vocabulary, "../vocabulary.pt")
    print("End Of Script")
