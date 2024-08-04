# coding:utf-8
from typing import Tuple, Dict, List
import numpy as np

import torch
from utils.box_utils import decode, center_to_corner
from torchvision.ops import nms


class Detector:
    """ detector """

    def __init__(self, anchors: list, image_size: int, n_classes: int, conf_thresh=0.25, nms_thresh=0.45):
        """
Parameters
        -----------
        Anchors: list of shape `(3, n_anchors, 2)`
            Pioneering box

        Image_size: int
            size of the picture

        n_classes: int
            Number of categories

        conf_thresh: Float
            Confidence threshold

        nms_thresh: float
            The threshold of iOU in NMS operation, the larger the reserved prediction box
        """
        self.anchors = anchors
        self.n_classes = n_classes
        self.image_size = image_size
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

    def __call__(self, preds: Tuple[torch.Tensor]) -> List[Dict[int, torch.Tensor]]:
        """ Treatment of the results of neural network output

        Parameters
        -----------
        Preds: Tuple [tensor]
            Three feature maps output of neural networks, the dimension is `(n, c, 13, 13)`, `(n, c, 26, 26)` and `(n, c, 52, 52)` `` `` `

        Returns
        --------
        out: list [Dict [IT, Tensor]]]
            All the detection results of the input picture, a element in the list represents the detection result of a picture,
            The keys in the dictionary are category indexes, and the value is the test results of this category. The first element of the last dimension of the test result is confidence.
            The latter four elements are boundary frames `(CX, Cy, W, h)`
        """
        N = preds[0].size(0)

        # decoding
        batch_pred = []
        for pred, anchors in zip(preds, self.anchors):
            pred_ = decode(pred, anchors, self.n_classes, self.image_size)

            # Pingping prediction box, shape: (n, n_anchors*h*w, n_classes+5)
            batch_pred.append(pred_.view(N, -1, self.n_classes+5))

        batch_pred = torch.cat(batch_pred, dim=1)

        # Non -big value inhibitory
        out = []
        for pred in batch_pred:
            # 计算得分
            pred[:, 5:] = pred[:, 5:] * pred[:, 4:5]

            # Select the category with the highest confidence of the class as the prediction result of the prediction box, shape: (n_anchors*h*w, 6)
            conf, c = torch.max(pred[:, 5:], dim=1, keepdim=True)
            pred = torch.cat((pred[:, :4], conf, c), dim=1)

            # Filtering the prediction box with too low reliability
            pred = pred[pred[:, 4] >= self.conf_thresh]
            if not pred.size(0):
                continue

            # The predicted category typeicted category type
            classes_pred = pred[:, -1].unique()

            # Filter the prediction box of each category
            detections = {}
            for c in classes_pred:
                mask = pred[:, -1] == c
                boxes = pred[:, :4][mask]
                scores = pred[:, 4][mask]
                keep = nms(center_to_corner(boxes), scores, self.nms_thresh)
                detections[int(c)] = torch.cat(
                    (scores[keep].unsqueeze(1), boxes[keep]), dim=1)

            out.append(detections)

        return out
