from utils import *


class DogBreedDataset(Dataset):
    def __init__(self, path, transform):
        self.img_files, self.targets, _ = load_dog_dataset(path)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        img = self.transform(img)
        target = self.targets[idx]
        return img, target

class DogBreedClassifierPipeline:
    def __init__(self, path):
        self.path = Path(path)
        _, _, self.dog_targets_map = load_dog_dataset(self.path / "/train/")

        self.transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }

        train_dataset = DogBreedDataset(self.path / "/train/", transforms["train"])
        valid_dataset = DogBreedDataset(self.path / "/valid/", transforms["valid"])
        test_dataset = DogBreedDataset(self.path / "/test/", transforms["test"])

        batch_size = 20
        num_workers = 0
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, 
                                                num_workers=num_workers,
                                                shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size, 
                                                num_workers=num_workers,
                                                shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size, 
                                                num_workers=num_workers,
                                                shuffle=False)