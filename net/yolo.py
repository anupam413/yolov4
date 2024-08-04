# coding:utf-8
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from utils.augmentation_utils import ToTensor
from utils.box_utils import draw, rescale_bbox

from .detector import Detector


class Mish(nn.Module):
    """ Mish Activation function """

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CBMBlock(nn.Module):
    """ CBM piece """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        """ Feedforward

        Parameters
        -----------
        x: tensor of shape `(n, in_Channels, h, w)`
            enter

        Returns
        --------
        y: tensor of shape `(n, out_Channels, h, w)`
            Output
        """
        return self.mish(self.bn(self.conv(x)))


class ResidualUnit(nn.Module):
    """ Residual unit """

    def __init__(self, in_channels, hidden_channels=None):
        """
Parameters
        -----------
        in_Channels: int
            Input channel number

        hidden_channels: int
            The number of output channels for the first CBM block
        """
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.block = nn.Sequential(
            CBMBlock(in_channels, hidden_channels, 1),
            CBMBlock(hidden_channels, in_channels, 3),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock(nn.Module):
    """ Residual block """

    def __init__(self, in_channels, out_channels, n_blocks):
        """
        Parameters
        -----------
        in_Channels: int
            Input channel number

        out_Channels: int
            Number of output channels

        n_blocks: int
            Number of internal residual units
        """
        super().__init__()
        self.downsample_conv = CBMBlock(
            in_channels, out_channels, 3, stride=2)

        if n_blocks == 1:
            self.split_conv0 = CBMBlock(out_channels, out_channels, 1)
            self.split_conv1 = CBMBlock(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                ResidualUnit(out_channels, out_channels//2),
                CBMBlock(out_channels, out_channels, 1)
            )
            self.concat_conv = CBMBlock(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = CBMBlock(out_channels, out_channels//2, 1)
            self.split_conv1 = CBMBlock(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[ResidualUnit(out_channels//2) for _ in range(n_blocks)],
                CBMBlock(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = CBMBlock(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        # Residual part
        x0 = self.split_conv0(x)

        # Main part
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        return self.concat_conv(x)


class CSPDarkNet(nn.Module):
    """ CSPDarkNet Main network """

    def __init__(self) -> None:
        super().__init__()
        layers = [1, 2, 8, 8, 4]
        channels = [32, 64, 128, 256, 512, 1024]
        self.conv1 = CBMBlock(3, 32, 3)
        self.stages = nn.ModuleList([
            ResidualBlock(channels[i], channels[i+1], layers[i]) for i in range(5)
        ])

    def forward(self, x):
        """ Feedforward

        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            enter

        Returns
        -------
        x3: Tensor of shape `(N, 256, H/8, W/8)`
        x4: Tensor of shape `(N, 512, H/16, W/16)`
        x5: Tensor of shape `(N, 1024, H/32, W/32)`
        """
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)

        # Output three feature maps
        x3 = self.stages[2](x)
        x4 = self.stages[3](x3)
        x5 = self.stages[4](x4)

        return x3, x4, x5

    def load(self, model_path: Union[Path, str]):
        """ Loading model

        Parameters
        -----------
        Model_path: Str or Path
            Model file path
        """
        self.load_state_dict(torch.load(model_path))

    def set_freezed(self, freeze: bool):
        """ Set whether the model parameter is frozen"""
        for param in self.parameters():
            param.requires_grad = not freeze


class CBLBlock(nn.Module):
    """ CBL piece """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size-1)//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SPPBlock(nn.Module):
    """ SPP piece """

    def __init__(self, sizes=(5, 9, 13)):
        super().__init__()
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(size, 1, size//2) for size in sizes
        ])

    def forward(self, x):
        x1 = [pool(x) for pool in self.maxpools[::-1]]
        x1.append(x)
        return torch.cat(x1, dim=1)


class Upsample(nn.Module):
    """ Sampling """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            CBLBlock(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

    def forward(self, x):
        return self.upsample(x)


def make_three_cbl(channels: list):
    """ Create three CBL blocks """
    return nn.Sequential(
        CBLBlock(channels[0], channels[1], 1),
        CBLBlock(channels[1], channels[2], 3),
        CBLBlock(channels[2], channels[1], 1),
    )


def make_five_cbl(channels: list):
    """ Create five CBL blocks """
    return nn.Sequential(
        CBLBlock(channels[0], channels[1], 1),
        CBLBlock(channels[1], channels[2], 3),
        CBLBlock(channels[2], channels[1], 1),
        CBLBlock(channels[1], channels[2], 3),
        CBLBlock(channels[2], channels[1], 1),
    )


def make_yolo_head(channels: list):
    """ Create YOLO head """
    return nn.Sequential(
        CBLBlock(channels[0], channels[1], 3),
        nn.Conv2d(channels[1], channels[2], 1),
    )


class Yolo(nn.Module):
    """ Yolov4 Neural Networks """

    def __init__(self, n_classes, image_size=416, anchors: list = None, conf_thresh=0.1, nms_thresh=0.45):
        """
Parameters
        -----------
        n_classes: int
            Number of categories

        Image_size: int
            The size of the picture must be the multiple of 32

        anchors: list of shape `(1, 3, n_anchors, 2)` `
            The input image size is 416, a prior box, and the scale has from to large

        conf_thresh: Float
            Confidence threshold

        nms_thresh: float
            The non -extremely large value inhibitory is compared with the threshold, the larger the more reserved prediction box value inhibitory is compared with the threshold, the larger the more reserved prediction box
        """
        super().__init__()
        if image_size <= 0 or image_size % 32 != 0:
            raise ValueError("image_size Must be a 32 multiple")

        # Pioneering box
        anchors = anchors or [
            [[142, 110], [192, 243], [459, 401]],
            [[36, 75], [76, 55], [72, 146]],
            [[12, 16], [19, 36], [40, 28]],
        ]
        anchors = np.array(anchors, dtype=np.float32)
        anchors = anchors*image_size/416
        self.anchors = anchors.tolist()  # type:list

        self.n_classes = n_classes
        self.image_size = image_size

        # Main network
        self.backbone = CSPDarkNet()

        self.conv1 = make_three_cbl([1024, 512, 1024])
        self.SPP = SPPBlock()
        self.conv2 = make_three_cbl([2048, 512, 1024])

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = CBLBlock(512, 256, 1)
        self.make_five_conv1 = make_five_cbl([512, 256, 512])

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = CBLBlock(256, 128, 1)
        self.make_five_conv2 = make_five_cbl([256, 128, 256])

        channel = len(self.anchors[1]) * (5 + n_classes)
        self.yolo_head3 = make_yolo_head([128, 256, channel])

        self.down_sample1 = CBLBlock(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_cbl([512, 256, 512])

        self.yolo_head2 = make_yolo_head([256, 512, channel])

        self.down_sample2 = CBLBlock(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_cbl([1024, 512, 1024])

        self.yolo_head1 = make_yolo_head([512, 1024, channel])

        # detector
        self.detector = Detector(
            self.anchors, image_size, n_classes, conf_thresh, nms_thresh)

    def forward(self, x: torch.Tensor):
        """ Feedforward

        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            enter

        Returns
        -------
        y0: Tensor of shape `(N, 3*(5+n_classes), 13, 13)`
        y1: Tensor of shape `(N, 3*(5+n_classes), 26, 26)`
        y2: Tensor of shape `(N, 3*(5+n_classes), 52, 52)`
        """
        # Main network
        x2, x1, x0 = self.backbone(x)

        # (13, 13, 1024) --> (13, 13, 512)
        P5 = self.conv2(self.SPP(self.conv1(x0)))

        # Switching after the sampling P5, (13, 13, 512) --> (26, 26, 256) --> (26, 26, 512)
        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        # (26, 26, 512) --> (26, 26, 256)
        P4 = self.make_five_conv1(P4)

        # Switch after the sampling P4, (26. 26, 256) --> (52, 52, 128) --> (52, 52, 256)
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        # (52, 52, 256) --> (52, 52, 128)
        P3 = self.make_five_conv2(P3)

        # P3 sampling P3 and then stitch with P4, (52, 52, 128) --> (26, 26, 256) --> (26, 26, 512)
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        # (26, 26, 512) --> (26, 26, 256)
        P4 = self.make_five_conv3(P4)

        # Sample sampling P4 before stitching P5, (26, 26, 256) --> (13, 13, 512) --> (13, 13, 1024)
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        # (13, 13, 1024) --> (13, 13, 512)
        P5 = self.make_five_conv4(P5)

        # Three feature maps output
        y2 = self.yolo_head3(P3)  # (N, 3*(5+n_classes), 52, 52)
        y1 = self.yolo_head2(P4)  # (N, 3*(5+n_classes), 26, 26)
        y0 = self.yolo_head1(P5)  # (N, 3*(5+n_classes), 13, 13)

        return y0, y1, y2

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """ forecast result

        Parameters
        -----------
        x: tensor of shape `(n, 3, h, w)`
            Enter the image, it has been native

        Returns
        --------
        out: list [Dict [IT, Tensor]]]
            All the detection results of the input picture, a element in the list represents the detection result of a picture,
            The keys in the dictionary are category indexes, the value of the detection results of this category, and the last dimension is `(conf, cx, cy, w, h)`,
        """
        return self.detector(self(x))

    def detect(self, image: Union[str, np.ndarray], classes: List[str], use_gpu=True, show_conf=True) -> Image.Image:
        """ Targe detection of the picture

        Parameters
        -----------
        Image: Str of `np.ndarray`
            Picture path or RGB image

        CLASSSES: List [STR]
            Category name list

        use_gpu: BOOL
            Whether to use GPU

        show_conf: BOOL
            Whether to show confidence

        Returns
        --------
        Image: ~ pil.image.image`
            Draw the boundary box, confidence and category image
        """
        if isinstance(image, str):
            if os.path.exists(image):
                image = np.array(Image.open(image).convert('RGB'))
            else:
                raise FileNotFoundError("The picture does not exist, please check the picture path!")

        h, w, channels = image.shape
        if channels != 3:
            raise ValueError('The input must be the RGB image of the three channels')

        x = ToTensor(self.image_size).transform(image)
        if use_gpu:
            x = x.cuda()

        # Forecasting border and confidence, sharing: (n_classes, top_k, 5)
        y = self.predict(x)
        if not y:
            return Image.fromarray(image)

        # Filtering the prediction box with no less confidence than the threshold
        bbox = []
        conf = []
        label = []
        for c, pred in y[0].items():
            # shape: (n_boxes, 5)
            pred = pred.numpy()  # type: np.ndarray

            # Restore the boundary frame of the original size
            boxes = rescale_bbox(pred[:, 1:], self.image_size, h, w)
            bbox.append(boxes)

            conf.extend(pred[:, 0].tolist())
            label.extend([classes[c]] * pred.shape[0])

        if not show_conf:
            conf = None

        image = draw(image, np.vstack(bbox), label, conf)
        return image

    def load(self, model_path: Union[Path, str]):
        """ Loading model

        Parameters
        -----------
        Model_path: Str or Path
            Model file path
        """
        self.load_state_dict(torch.load(model_path))
