import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from PIL import Image

def tensor_transforms() -> torchvision.transforms.Compose:
    custom_transforms = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
    ]
    return custom_transforms

class MTLDataset(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.paths = os.listdir(root)

    def __getitem__(self, index) -> Tuple:
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        annotation = {}
        labels = path.split("_")

        # Labels
        annotation['age'] = torch.as_tensor(float(labels[0]), dtype=torch.float32)
        annotation['gender'] = torch.as_tensor(int(labels[1]), dtype=torch.int64)
        annotation['race'] = torch.as_tensor(int(labels[2]), dtype=torch.int64)

        if self.transform is not None:
            img = self.transform(img)

        return img, annotation

    def __len__(self) -> int: 
        return len(self.paths)

def dataloader(train_data_dir: str,
            test_data_dir: str = None,
            batch_size: int = 32,
            test_batch_size: int = 16,
            shuffle: bool = True,
            num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MTLDataset(root=train_data_dir, transform=tensor_transforms())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if test_data_dir is not None:
        val_dataset = MTLDataset(root=test_data_dir, transform=tensor_transforms())
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, val_loader
        
    return train_loader