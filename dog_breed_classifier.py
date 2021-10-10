from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

from utils import *
from datasets import *


class DogBreedClassifierPipeline:
    def __init__(
        self,
        data_path,
        model,
        batch_size=20,
        num_workers=0,
        log_rate=50,
        save_rate=300,
        valid_rate=300,
        save_path="pretrained/models",
        save_prefix="model"
    ):
        # Paths
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = model

        # Logging frequency
        self.log_rate = log_rate
        self.save_rate = save_rate
        self.valid_rate = valid_rate

        # Training parameters
        self.start_epoch = 0
        self.save_prefix = save_prefix

        _, _, self.dog_targets_map = load_dog_dataset(self.data_path / "train")

        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
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

        train_dataset = DogBreedDataset(
            self.data_path / "train", self.data_transforms["train"]
        )
        valid_dataset = DogBreedDataset(
            self.data_path / "valid", self.data_transforms["valid"]
        )
        test_dataset = DogBreedDataset(
            self.data_path / "test", self.data_transforms["test"]
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

    def train(self, num_epochs, loss_fn, optimizer, device):
        self.model.to(device)
        train_loss = 0.0
        min_valid_loss = 999
        for epoch in range(self.start_epoch, num_epochs):
            for i, data in enumerate(self.train_loader):
                # Ensure model is in training mode
                self.model = self.model.train()

                # Load data
                img = data["img"].to(device)
                target = data["target"].type(torch.LongTensor).to(device)

                # Forward pass
                out = self.model(img)

                # Zero optimizer
                optimizer.zero_grad()

                # Calculate loss
                loss = loss_fn(out, target)
                train_loss += loss.item()

                # Backward pass
                loss.backward()

                # Optimize
                optimizer.step()

                n_iter = epoch * len(self.train_loader) + i

                # Logging
                if n_iter % self.log_rate == 0:
                    print(
                        "Epoch: {:5d} | Batch: {:5d} | Training Loss: {:03f}".format(
                            epoch + 1, i + 1, train_loss / self.log_rate
                        )
                    )
                    train_loss = 0.0

                # Save model
                if n_iter % self.save_rate == 0 and n_iter != 0:
                    model_save_info = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S%f")
                    print("Saving model")
                    torch.save(
                        model_save_info,
                        str(
                            self.save_path
                            / "{}_{}_{:05d}.pt".format(self.save_prefix, timestamp, n_iter)
                        ),
                    )

                # Validate model
                if n_iter % self.valid_rate == 0 and n_iter != 0:
                    valid_loss = 0.0
                    # Ensure model is in evaluation mode
                    self.model.eval()
                    for j, data in enumerate(self.valid_loader):
                        # Load data
                        img = data["img"].to(device)
                        target = data["target"].type(torch.LongTensor).to(device)

                        # Forward pass
                        out = self.model(img)

                        # Calculate loss
                        loss = loss_fn(out, target)
                        valid_loss += loss.item()

                    valid_loss = valid_loss / len(self.valid_loader)

                    # Logging
                    print(
                        "Epoch: {:5d} | Batch: {:5d} | Validation Loss: {:03f}".format(
                            epoch + 1, i + 1, valid_loss
                        )
                    )

                    if valid_loss < min_valid_loss:
                        print(
                            "Validation score improved ({} -> {})".format(
                                min_valid_loss, valid_loss
                            )
                        )
                        model_save_info = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S%f")
                        print("Saving model")
                        torch.save(
                            model_save_info,
                            str(
                                self.save_path
                                / "{}_{}_{:05d}_best.pt".format(self.save_prefix, timestamp, n_iter)
                            ),
                        )
                        min_valid_loss = valid_loss

        # Final validation
        valid_loss = 0.0
        # Ensure model is in evaluation mode
        self.model = self.model.eval()
        for j, data in enumerate(self.valid_loader):
            # Load data
            img = data["img"].to(device)
            target = data["target"].type(torch.LongTensor).to(device)

            # Forward pass
            out = self.model(img)

            # Calculate loss
            loss = loss_fn(out, target)
            valid_loss += loss.item()

        valid_loss = valid_loss / len(self.valid_loader)

        # Logging
        print(
            "Epoch: {:5d} | Batch: {:5d} | Validation Loss: {:03f}".format(
                epoch + 1, i + 1, valid_loss
            )
        )

        model_save_postfix = ""
        if valid_loss < min_valid_loss:
            print(
                "Validation score improved ({} -> {})".format(
                    min_valid_loss, valid_loss
                )
            )
            min_valid_loss = valid_loss
            model_save_postfix = "_best"

        model_save_info = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S%f")
        print("Saving model")
        torch.save(
            model_save_info,
            str(
                self.save_path
                / "{}_{}_{:05d}{}.pt".format(self.save_prefix, timestamp, n_iter, model_save_postfix)
            ),
        )

    def load(self, path):
        model_save_info = torch.load(path)
        print("Loading model from checkpoint {}".format(path))
        self.start_epoch = model_save_info["epoch"] + 1
        self.model.load_state_dict(model_save_info["model_state_dict"])

    def evaluate(self, device):
        self.model = self.model.eval()
        y_pred = []
        y_true = []
        for j, data in enumerate(self.test_loader):
            # Load data
            img = data["img"].to(device)
            target = data["target"].type(torch.LongTensor).to(device)

            # Forward pass
            out = self.model(img)
            pred_idx = torch.max(out, 1)[1]

            # Collect results
            y_pred += pred_idx.tolist()
            y_true += target.tolist()

        print(
            "Accuracy: {}".format(
                np.sum(np.array(y_pred) == np.array(y_true)) / len(y_pred)
            )
        )
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("Fscore: {}".format(fscore))
