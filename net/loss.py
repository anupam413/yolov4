# coding: utf-8
from typing import List
import numpy as np

import torch
from torch import Tensor, nn
from utils.box_utils import match, decode, ciou, iou


class YoloLoss(nn.Module):
    """ Loss function """

    def __init__(self, anchors: list, n_classes: int, image_size: int, overlap_thresh=0.5):
        """
Parameters
        -----------
        Anchors: list of shape `(3, n_anchors, 2)`
            List of ancestral box, size from large to small

        n_classes: int
            Number of categories

        Image_size: int
            Enter the picture size of the neural network

        overlap_thresh: Float
            The IOU threshold deemed to ignore the sample
        """
        super().__init__()
        self.n_anchors = len(anchors[0])
        self.anchors = np.array(anchors).reshape(-1, 2)
        self.n_classes = n_classes
        self.image_size = image_size
        self.overlap_thresh = overlap_thresh

        # The weight of each part of the loss function
        self.balances = [0.4, 1, 4]
        self.lambda_box = 0.05
        self.lambda_obj = 5*(image_size/416)**2
        self.lambda_cls = n_classes / 80

        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, index: int,  pred: Tensor, targets: List[Tensor]):
        """ Calculate the loss of a feature diagram

        Parameters
        -----------
        index: int
            The index of the feature diagram is 0 ~ 2
        preds: tensor
            Each feature diagram of YOLO neural network output, the dimension is:
            * `(N, (n_classes+5)* n_anchors, 13, 13)` ``
            * `(N, (n_classes+5)* n_anchors, 26, 26)` ``
            * `(N, (n_classes+5)* n_anchors, 52, 52)` ``

        targets: list [tensor]
            Label data, the dimension of each label tensor is `(n, n_objects, 5)`, the last dimension is 
            the border frame `(CX, Cy, H, class)` `` `

        Returns
        --------
        Loss: tensor of shape `(1,)` `
            Losses
        """
        loss = 0
        N, _, h, w = pred.shape

        # Decoding the prediction results, shape: (n, n_anchors, h, w, n_classes+5)
        anchor_mask = list(
            range(index*self.n_anchors, (index+1)*self.n_anchors))
        pred = decode(pred, self.anchors[anchor_mask],
                      self.n_classes, self.image_size, False)

        # Matching boundary box
        step_h = self.image_size / h
        step_w = self.image_size / w
        anchors = [[i/step_w, j/step_h] for i, j in self.anchors]
        p_mask, n_mask, gt = match(
            anchors, anchor_mask, targets, h, w, self.n_classes, self.overlap_thresh)
        self.mark_ignore(pred, targets, n_mask)

        p_mask = p_mask.to(pred.device)
        n_mask = n_mask.to(pred.device)
        gt = gt.to(pred.device)

        m = p_mask == 1
        if m.sum() != 0:
            # Position losstion loss
            iou = ciou(pred[..., :4], gt[..., :4])
            m &= torch.logical_not(torch.isnan(iou))
            loss += torch.mean((1-iou)[m])*self.lambda_box

            # Classification loss
            loss += self.bce_loss(pred[..., 5:][m], gt[..., 5:][m])*self.lambda_cls

        # Trust loss of positive samples and negative samples
        mask = n_mask.bool() | m
        loss += self.bce_loss(pred[..., 4]*mask, m.type_as(pred)*mask) * \
            self.lambda_obj*self.balances[index]

        return loss

    def mark_ignore(self, pred: Tensor, targets: List[Tensor], n_mask: Tensor):
        """ Mark a neglect sample

        Parameters
        -----------
        Pred: tensor of shape `(n, n_anchors, h, w, n_classes+5)` `` `` `` `` `` `` `
            Decoding feature diagram

        targets: list [tensor]
            Label data, the dimension of each label tensor is `(n, n_objects, 5)`, the last dimension is the border frame `(CX, Cy, H, class)` `` `

        n_mask: tensor of shape `(n, n, n_anchors, h, w)`
            Negative sample mask
        """
        N, _, h, w, _ = pred.shape
        bbox = pred[..., :4]

        for i in range(N):
            if targets[i].size(0) == 0:
                continue

            # shape: (h*w*n_anchors, 4)
            box = bbox[i].view(-1, 4)
            target = torch.zeros_like(targets[i][..., :4])
            target[:, [0, 2]] = targets[i][:, [0, 2]] * w
            target[:, [1, 3]] = targets[i][:, [1, 3]] * h

            # Calculate the prediction box and the real box
            max_iou, _ = torch.max(iou(target, box), dim=0)
            max_iou = max_iou.view(pred[i].shape[:3])
            n_mask[i][max_iou > self.overlap_thresh] = 0
