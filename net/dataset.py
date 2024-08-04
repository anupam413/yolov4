# coding:utf-8
from os import path
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree as ET
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.augmentation_utils import Transformer
from utils.annotation_utils import AnnotationReader
from utils.box_utils import corner_to_center_numpy


class VOCDataset(Dataset):
    """ VOC data set"""

    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, root: Union[str, List[str]], image_set: Union[str, List[str]],
                 transformer: Transformer = None, color_transformer: Transformer = None, keep_difficult=False,
                 use_mosaic=False, use_mixup=False, image_size=416):
        """
        Parameters
        -----------
        root: Str or List [Str]
            The root path of the dataset, below must be `Annotations`,` ImageSets` and `JPEGIMAGES` folders

        Image_Set: Str or List [Str]
            The types of data sets can be `Train`,` Val`, `Trainval` or` Test`

        Transformer: Transformer
            Data enhancer used when not using mosaic enhancement

        color_transformer: Transformer
            Data enhancer used when using mosaic enhancement

        Keep_difficulty: Bool
            Whether to keep the sample with Difficult 1

        use_mosaic: Bool
            Whether to enable mosaic data enhancement

        use_mixup: BOOL
            Whether the Mixup data enhances, it only works when `mosaic` is` true`

        Image_size: int
            The image size of the data set output
        """
        super().__init__()
        if isinstance(root, str):
            root = [root]
        if isinstance(image_set, str):
            image_set = [image_set]
        if len(root) != len(image_set):
            raise ValueError("`Root` and `Image_set` must be the same")

        self.root = root
        self.image_set = image_set
        self.image_size = image_size
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup

        self.n_classes = len(self.classes)
        self.keep_difficult = keep_difficult
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        self.transformer = transformer    # Data enhancer
        self.color_transformer = color_transformer
        self.annotation_reader = AnnotationReader(
            self.class_to_index, keep_difficult)

        # Get all the pictures and label file paths of the specified data set
        self.image_names = []
        self.image_paths = []
        self.annotation_paths = []

        for root, image_set in zip(self.root, self.image_set):
            with open(path.join(root, f'ImageSets/Main/{image_set}.txt')) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.image_names.append(line)
                    self.image_paths.append(
                        path.join(root, f'JPEGImages/{line}.jpg'))
                    self.annotation_paths.append(
                        path.join(root, f'Annotations/{line}.xml'))

        # print("imagepath", self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """ Obtain a sample

        Parameters
        -----------
        index: int
            Bidding

        Returns
        --------
        Image: tensor of shape `(3, h, w)`
            Enhanced image data

        target: `np.ndarray` of shape` (N_Objects, 5) `
            Tag data, each line format is `(CX, Cy, W, H, Class)` `
        """
        # 50% Probability for mosaic data enhancement
        if self.use_mosaic and np.random.randint(2):
            image, bbox, label = self.make_mosaic(index)

            # mixup
            if self.use_mixup and np.random.randint(2):
                index_ = np.random.randint(0, len(self))
                # print("-------------------------------------------------------")
                # print("index_", index_)
                image_, bbox_, label_ = self.make_mosaic(index_)
                r = np.random.beta(8, 8)
                image = (image*r+image_*(1-r)).astype(np.uint8)
                bbox = np.vstack((bbox, bbox_))
                label = np.hstack((label, label_))

            # Image enhancement
            if self.color_transformer:
                image, bbox, label = self.color_transformer.transform(
                    image, bbox, label)

        else:
            image, bbox, label = self.read_image_label(index)
            if self.transformer:
                image, bbox, label = self.transformer.transform(
                    image, bbox, label)

        image = image.astype(np.float32)
        image /= 255.0
        bbox = corner_to_center_numpy(bbox)
        target = np.hstack((bbox, label[:, np.newaxis]))

        return torch.from_numpy(image).permute(2, 0, 1), target

    def make_mosaic(self, index: int):
        """ The picture after creating a mosaic enhancement """
        # Select three pictures randomly
        indexes = list(range(len(self.image_paths)))
        choices = random.sample(indexes[:index]+indexes[index+1:], 3)
        choices.append(index)

        # Read the four pictures and labels used to make mosaics and their labels
        images, bboxes, labels = [], [], []
        # print("choices", choices)
        for i in choices:
            image, bbox, label = self.read_image_label(i)
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)

        # Create a mosaic image and select the splicing point
        img_size = self.image_size
        mean = np.array([123, 117, 104])
        mosaic_img = np.ones((img_size*2, img_size*2, 3))*mean
        xc = int(random.uniform(img_size//2, 3*img_size//2))
        yc = int(random.uniform(img_size//2, 3*img_size//2))

        # Stitching image
        for i, (image, bbox, label) in enumerate(zip(images, bboxes, labels)):
            # Reserved/non -reserved proportional scaling image
            ih, iw, _ = image.shape
            s = np.random.choice(np.arange(50, 210, 10))/100
            if np.random.randint(2):
                r = img_size / max(ih, iw)
                if r != 1:
                    image = cv.resize(image, (int(iw*r*s), int(ih*r*s)))
            else:
                image = cv.resize(image, (int(img_size*s), int(img_size*s)))

            # Paste the image to the upper left corner of the stitching point, upper right corner, lower left corner and lower right corner
            h, w, _ = image.shape
            if i == 0:
                # The coordinates in the upper left corner and lower right corner in the mosaic image
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # The coordinates of the upper left corner and lower right corner of the paste part in the original image
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(
                    yc - h, 0), min(xc + w, img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(
                    xc - w, 0), yc, xc, min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, img_size * 2), min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # Reverse the border frame back and translate the coordinates
            dx = x1a - x1b
            dy = y1a - y1b
            bbox[:, [0, 2]] = bbox[:, [0, 2]]*w+dx
            bbox[:, [1, 3]] = bbox[:, [1, 3]]*h+dy

        # Treatment of the boundary frame that exceeds the mosaic image coordinate system
        bbox = np.clip(np.vstack(bboxes), 0, 2*img_size)
        label = np.hstack(labels)

        # Remove the small boundary frame
        bbox_w = bbox[:, 2] - bbox[:, 0]
        bbox_h = bbox[:, 3] - bbox[:, 1]
        mask = np.logical_and(bbox_w > 1, bbox_h > 1)
        bbox, label = bbox[mask], label[mask]
        if len(bbox) == 0:
            bbox = np.zeros((1, 4))
            label = np.array([0])

        # Betalization Boundary Frame
        bbox /= mosaic_img.shape[0]

        return mosaic_img, bbox, label

    def read_image_label(self, index: int):
        """ Read pictures and label data

        Parameters
        -----------
        index: int
            Reading sample index

        Returns
        --------
        Image: `~ np.ndarray` of shape` (h, w, 3) `
                RGB image

        bbox: `~ np.ndarray` of shape` (N_Objects, 4) ``
            Breakfast border frame

        label: `~ np.ndarray` of shape` (N_Objects,) `` `
            Category
        """
        image = cv.cvtColor(
            cv.imread(self.image_paths[index]), cv.COLOR_BGR2RGB)
        target = np.array(self.annotation_reader.read(
            self.annotation_paths[index]))
        bbox, label = target[:, :4], target[:, 4]
        return image, bbox, label


def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]):
    """ Sort the data from DataLoader

    Parameters
    -----------
    BATCH: List of Shape `(n, 2)`
        A batch of data, each in the list includes two elements::
        * Image: tensor of shape `(3, h, w)`
        * target: ~ np.ndarray` of shape `(N_Objects, 5)` `

    Returns
    --------
    Image: tensor of shape `(n, 3, h, w)`
        image

    target: list [tensor]
        Label
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))

    return torch.stack(images, 0), targets


def make_data_loader(dataset: VOCDataset, batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
