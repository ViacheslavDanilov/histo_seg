import logging
import multiprocessing
import os
import random
from glob import glob
from typing import List, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
from clearml import Dataset as cl_dataset
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.utils import CLASS_ID


class OCTDataset(Dataset):
    """The dataset used to process OCT images and corresponding segmentation masks."""

    def __init__(
        self,
        data_dir: str,
        classes: List[str],
        input_size: int = 224,
        use_augmentation: bool = False,
    ):
        self.classes = classes
        mask_paths = glob(f'{data_dir}/mask/*.[pj][np][pge]')
        self.input_size = input_size

        num_cores = multiprocessing.cpu_count()
        check_list = Parallel(n_jobs=num_cores, backend='threading')(
            delayed(self.data_check)(f'{data_dir}/img', mask_id)
            for mask_id in tqdm(mask_paths, desc='image load')
        )

        self.image_paths = list(np.array(check_list)[:, 1])
        self.mask_paths = list(np.array(check_list)[:, 0])
        self.class_values = [CLASS_ID[cl] for _, cl in enumerate(self.classes)]

        self.use_augmentation = use_augmentation

    def __getitem__(self, i: int):
        image = cv2.imread(self.image_paths[i])
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.imread(self.mask_paths[i], 0)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.use_augmentation:
            transform = self.get_img_augmentation(input_size=self.input_size)
            sample = transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image, mask = self.to_tensor(np.array(image)), self.to_tensor(np.array(mask))

        return image, mask

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def data_check(
        img_dir: str,
        ann_id: str,
    ) -> Union[Tuple[str, str], None]:
        img_path = f'{img_dir}/{os.path.basename(ann_id)}'
        if os.path.exists(img_path):
            return ann_id, img_path
        else:
            logging.warning(f'Img path: {img_path} not exist')
            return None

    @staticmethod
    def to_tensor(
        x: np.ndarray,
    ) -> np.ndarray:
        return x.transpose([2, 0, 1]).astype('float32')

    @staticmethod
    def get_img_augmentation(
        input_size: int,
    ) -> albu.Compose:
        transform = [
            albu.HorizontalFlip(
                p=0.5,
            ),  # TODO: sure about the aggressiveness of this augmentation pipeline?
            albu.ShiftScaleRotate(
                scale_limit=0.35,
                rotate_limit=45,
                shift_limit=0.1,
                p=0.65,
                border_mode=0,
            ),
            albu.RandomCrop(
                height=int(random.uniform(0.7, 0.9) * input_size),
                width=int(random.uniform(0.7, 0.9) * input_size),
                always_apply=True,
                p=0.5,
            ),
            albu.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                always_apply=True,
                border_mode=0,
            ),
            albu.GaussNoise(p=0.25),
            albu.Perspective(p=0.5),
            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(transform)


class OCTDataModule(pl.LightningDataModule):
    """A data module used to create training and validation dataloaders with OCT images."""

    def __init__(
        self,
        dataset_name: str,
        project_name: str,
        classes: List[str],
        input_size: int = 224,
        batch_size: int = 2,
        num_workers: int = 2,
        data_location: str = 'local',
    ):
        super().__init__()
        self.data_dir = None
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.classes = classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_location = data_location

    def prepare_data(self):
        if self.data_location == 'local':
            self.data_dir = 'data/final'
        elif self.data_location == 'cl_ml':
            self.data_dir = cl_dataset.get(
                dataset_name=self.dataset_name,
                dataset_project=self.project_name,
            ).get_local_copy()
        else:
            raise ValueError(f'The {self.data_location} method is not yet implemented')

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            self.train_dataloader_set = OCTDataset(
                input_size=self.input_size,
                data_dir=f'{self.data_dir}/train',
                classes=self.classes,
                use_augmentation=True,
            )
            self.val_dataloader_set = OCTDataset(
                input_size=self.input_size,
                data_dir=f'{self.data_dir}/test',
                classes=self.classes,
                use_augmentation=False,
            )
        elif stage == 'test':
            raise ValueError('The "test" method is not yet implemented')
        else:
            raise ValueError(f'Unsupported stage value: {stage}')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataloader_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataloader_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    dataset = OCTDataset(
        data_dir='data/final/test',
        classes=['Arteriole lumen', 'Arteriole media', 'Arteriole adventitia'],
        input_size=224,
        use_augmentation=False,
    )
    for i in range(10):
        img, mask = dataset[i]
