from typing import Optional
from pathlib import Path
import os
import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from datasets.iad_dataset import IadDataset
from utils.configurations import LabelName

def min_max_scale_image(mask, output_dtype: type = np.float32):
    image_array = np.array(mask).astype(output_dtype) / 255.0

    min_val = np.min(image_array)
    max_val = np.max(image_array)
    scaled_array = (image_array - min_val) / (max_val - min_val)
    # Ensure the scaled array is only 0s and 1s
    binary_array = torch.round(torch.tensor(scaled_array)).numpy()

    scaled_image = Image.fromarray((binary_array*255).astype(output_dtype))
    return scaled_image

class VisaDataset(IadDataset):
    def __init__(self,
                 root_path: str,
                 category: str,
                 gt_mask_size: Optional[tuple] = None,
                 img_size=(224,224),
                 preload_imgs: bool = True,
                 norm: bool = True):

        self.root_path = Path(root_path)
        self.csv_path = os.path.join(self.root_path, category, "image_anno.csv")
        self.category = category
        self.preload_imgs = preload_imgs

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size
        self.gt_mask_size = gt_mask_size
        self.img_size = img_size

        #load csv
        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError(f"No samples found for category '{category}' in {self.csv_path}")

        df["image"] = df["image"].apply(lambda x: os.path.join(self.root_path, x))
        df["mask"] = df.apply(
            lambda row: os.path.join(self.root_path, row["mask"]) if row["label"] != "normal" else "",
            axis=1)
        df["label_index"] = df["label"].apply(
            lambda x: LabelName.NORMAL if x == "normal" else LabelName.ABNORMAL
        ).astype(int)

        df.rename(columns={"image": "image_path", "mask": "mask_path"}, inplace=True)

        self.samples = df.reset_index(drop=True)

        if norm:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        else:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
            ]

        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

        #optional preload of images
        if preload_imgs:
            self.data = [
                self.transform_image(
                    Image.open(self.samples.iloc[i].image_path).convert("RGB")
                )
                for i in range(len(self.samples))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        row = self.samples.iloc[index]

        if self.preload_imgs:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(row.image_path).convert("RGB")
            )
        label = row.label_index
        path = row.image_path

        if label == LabelName.ABNORMAL:
            mask = Image.open(row.mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(1, *self.gt_mask_size)

        return image, label, mask.int(), path