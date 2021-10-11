from torch.utils.data import Dataset

from utils import *


class DogBreedDataset(Dataset):
    """
    Dog breed dataset
    """
    def __init__(self, path, transform):
        """"
        Initialize the dataset
        Input:
            path (string): Path to dataset
            transform: Input data transformation
        """
        self.img_files, self.targets, _ = load_dog_dataset(path)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        img = self.transform(img)
        target = self.targets[idx]
        sample = {"img": img, "target": target}
        return sample
