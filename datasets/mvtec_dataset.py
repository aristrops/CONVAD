import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms

from typing import Optional
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path

from datasets.iad_dataset import IadDataset
from utils.configurations import Split, LabelName

IMG_EXTENSIONS = (".png", ".PNG")

"""Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
"""


class MVTecDataset(IadDataset):
    """MVTec dataset class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec``.
        category (str): Sub-category of the dataset, e.g. 'bottle'
            Defaults to ``bottle``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``

    """

    def __init__(
        self,
        root: str,
        category: str,
        split: Split,
        norm: bool = True,
        img_size=(224, 224),
        gt_mask_size: Optional[tuple] = None,
        preload_imgs: bool = True,
    ) -> None:
        super(MVTecDataset)

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size

        self.img_size = img_size
        self.gt_mask_size = gt_mask_size

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples: pd.DataFrame = None
        self.preload_imgs = preload_imgs

        if norm:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

        # logo mask
        mask_gemini_logo_path = os.environ.get("GEMINI_LOGO_MASK_PATH")
        self.mask_gemini_logo_path = (
            Path(mask_gemini_logo_path)
            if isinstance(mask_gemini_logo_path, str)
            else mask_gemini_logo_path
        )
        # load and save the gemini logo mask if provided
        if self.mask_gemini_logo_path is not None:
            if not self.mask_gemini_logo_path.exists():
                raise FileNotFoundError(
                    f"Gemini logo mask file not found: {self.mask_gemini_logo_path}"
                )
            # load the image with PIL
            logo_mask_img = Image.open(self.mask_gemini_logo_path).convert("L")
            # transform the mask
            self.gemini_logo_mask = self.transform_mask(logo_mask_img)
        else:
            self.gemini_logo_mask = None

    def contains(self, item) -> bool:
        return self.samples["image_path"].eq(item["image_path"]).any()

    def load_dataset(self, use_gen_anomalies: bool = False):

        root = Path(self.root_category)

        if use_gen_anomalies and self.split == Split.TEST:
            # redirect test root to the generated anomaly directory
            gen_root = root / "generated_anomalies"
            if not gen_root.exists():
                raise FileNotFoundError(
                    f"Generated anomalies directory not found: {gen_root}"
                )
            print(f"Using generated anomalies from {gen_root}")

            # normal test (good images)
            samples_list = [
                (str(root), "test", "good", f.name)
                for f in (root / "test" / "good").glob("**/*")
                if f.suffix in IMG_EXTENSIONS
            ]

            # generated anomalies (each defect type folder)
            for f in gen_root.glob("**/*"):
                if f.suffix in IMG_EXTENSIONS:
                    defect_type = f.parts[-2]  # e.g. "broken_large"
                    samples_list.append(
                        (str(root), "generated_anomalies", defect_type, f.name)
                    )

        else:
            samples_list = [
                (str(root),) + f.parts[-3:]
                for f in root.glob(r"**/*")
                if f.suffix in IMG_EXTENSIONS and "generated_anomalies" not in f.parts
            ]

        if not samples_list:
            msg = f"Found 0 images in {root}"
            raise RuntimeError(msg)

        samples = pd.DataFrame(
            samples_list, columns=["path", "split", "label", "image_path"]
        )

        # Modify image_path column by converting to absolute path
        samples["image_path"] = (
            samples.path
            + "/"
            + samples.split
            + "/"
            + samples.label
            + "/"
            + samples.image_path
        )

        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:
            if use_gen_anomalies:
                # we don't have ground-truth masks available
                samples["mask_path"] = ""
            else:
                # separate masks from samples
                mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
                    by="image_path", ignore_index=True
                )
                samples = samples[samples.split != "ground_truth"].sort_values(
                    by="image_path", ignore_index=True
                )

                # assign mask paths to anomalous test images
                samples["mask_path"] = ""
                samples.loc[
                    (samples.split == "test")
                    & (samples.label_index == LabelName.ABNORMAL),
                    "mask_path",
                ] = mask_samples.image_path.to_numpy()

                # assert that the right mask files are associated with the right test images
                abnormal_samples = samples.loc[
                    samples.label_index == LabelName.ABNORMAL
                ]
                if (
                    len(abnormal_samples)
                    and not abnormal_samples.apply(
                        lambda x: Path(x.image_path).stem in Path(x.mask_path).stem,
                        axis=1,
                    ).all()
                ):
                    msg = """Mismatch between anomalous images and ground truth masks. Make sure t
                    he mask files in 'ground_truth' folder follow the same naming convention as the
                    anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
                    raise Exception(msg)

        if self.split == Split.TEST:
            keep = ["test", "generated_anomalies"] if use_gen_anomalies else ["test"]
            # include both test and test/generated_anomalies
            self.samples = samples[samples.split.isin(keep)].reset_index(drop=True)
        else:
            self.samples = samples[samples.split == self.split].reset_index(drop=True)

        if self.preload_imgs:
            self.data = [
                self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                for index in range(len(self.samples))
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        # open the image and get the tensor
        if self.preload_imgs:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(self.samples.iloc[index].image_path).convert("RGB")
            )

        if self.gemini_logo_mask is not None:
            image = image * self.gemini_logo_mask

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = self.samples.iloc[index].label_index
            path = self.samples.iloc[index].image_path
            mask_path = self.samples.iloc[index].mask_path

            if label == LabelName.ABNORMAL and mask_path:
                mask = Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size)

            return image, label, mask.int(), path
