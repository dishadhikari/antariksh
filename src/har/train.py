from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.har.model import TCN


class HARTrainer:

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 30,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.model = TCN(
            input_features=input_features,
            num_classes=num_classes,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def encode_labels(labels, label_to_index=None):
        if label_to_index is None:
            unique_labels = sorted(set(labels))
            label_to_index = {
                label: index
                for index, label in enumerate(unique_labels)
            }

        encoded = np.array(
            [label_to_index[label] for label in labels],
            dtype=np.int64,
        )

        return encoded, label_to_index

    def create_loader(self, features, labels, shuffle=True):
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(x, y)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(labels)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += len(labels)

        return total_loss / total, correct / total

    def validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for features, labels in loader:

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)

                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * len(labels)

                predictions = outputs.argmax(dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += len(labels)

        return total_loss / total, correct / total

    def fit(
        self,
        train_features,
        train_labels,
        val_features,
        val_labels,
        output_dir="models/har",
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_loader = self.create_loader(
            train_features,
            train_labels,
            shuffle=True,
        )

        val_loader = self.create_loader(
            val_features,
            val_labels,
            shuffle=False,
        )

        best_val_accuracy = 0.0

        checkpoint_path = (
            output_dir / "best_tcn.pt"
        )

        for epoch in range(1, self.epochs + 1):

            train_loss, train_accuracy = (
                self.train_epoch(train_loader)
            )

            val_loss, val_accuracy = (
                self.validate(val_loader)
            )

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                torch.save(
                    {
                        "model_state_dict":
                            self.model.state_dict(),
                        "input_features":
                            train_features.shape[2],
                        "num_classes":
                            len(set(train_labels)),
                    },
                    checkpoint_path,
                )

        print()
        print(
            f"Best model saved to: {checkpoint_path}"
        )

        return checkpoint_path


def load_har_dataset(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"HAR dataset not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    features = data["features"]
    labels = data["labels"].tolist()

    return features, labels


if __name__ == "__main__":
    print("HAR training module ready.")