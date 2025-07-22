import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def get_transforms(mode: str = 'train'):
    """
    Returns torchvision transforms for training or validation/testing.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


class ClassificationDataset(Dataset):
    """
    PyTorch Dataset for image classification based on folder structure.
    Root directory should have subfolders for each class (e.g., Normal/, Stone/).
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # discover classes and assign labels
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        # gather samples
        self.samples = []  # list of (relative_path, label)
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    rel_path = os.path.join(cls_name, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((rel_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def create_data_loaders(root_dir: str,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        val_split: float = 0.2,
                        random_seed: int = 42):
    """
    Create training and validation DataLoader objects by splitting folder-based dataset.
    """
    full_dataset = ClassificationDataset(root_dir, transform=None)
    labels = [label for _, label in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=val_split,
        stratify=labels,
        random_state=random_seed
    )
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    train_ds.dataset.transform = get_transforms('train')
    val_ds.dataset.transform = get_transforms('val')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

