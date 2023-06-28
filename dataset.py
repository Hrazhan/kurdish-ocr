import pandas
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class KTRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            # stride=32,
            # truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding


class HandWrittenDataset(Dataset):
    def __init__(
        self, root_dir, processor, train=True, max_target_length=128, test_split=0.2
    ):
        self.root_dir = root_dir
        self.processor = processor
        # self.tokenizer = tokenizer
        self.max_target_length = max_target_length

        self.labels = os.listdir(self.root_dir)
        self.images = []
        self.targets = []
        # print(f"Labels {self.labels[0]}")
        for label in self.labels:  # بەفرین
            label_path = os.path.join(self.root_dir, label)
            for img_filename in os.listdir(label_path):
                img_path = os.path.join(label_path, img_filename)
                self.images.append(img_path)
                self.targets.append(label)

        if train:
            # print("Train split", int(len(self.images)*(1-test_split)))
            self.images = self.images[: int(len(self.images) * (1 - test_split))]
            self.targets = self.targets[: int(len(self.targets) * (1 - test_split))]
        else:
            # print("Test split", int(len(self.images)*(1 - test_split)))
            self.images = self.images[int(len(self.images) * (1 - test_split)) :]
            self.targets = self.targets[int(len(self.targets) * (1 - test_split)) :]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        text = self.targets[idx]
        # print(file_name, text)
        img = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(img, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            # stride=32,
            # truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding
