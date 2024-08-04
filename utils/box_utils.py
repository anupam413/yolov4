# coding: utf-8
import math
from typing import List, Union

import cmapy
import numpy as np
import torch
from numpy import ndarray
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from math import pi


def iou(bbox1: Tensor, bbox2: Tensor):
    """ Calculate the merger of the two sets of boundary boxes, the four coordinates are `(CX, Cy, W, H)`

    Parameters
    -----------
    bbox1: tensor of shape `(a, 4)`
        The first group of boundary boxes

    bbox2: tensor of shape `(b, 4)`
        The second group of boundary boxes

    Returns
    --------
    iOU: tensor of shape `(a, b)`
        Comparison
    """
    A = bbox1.size(0)
    B = bbox2.size(0)

    bbox1 = center_to_corner(bbox1)
    bbox2 = center_to_corner(bbox2)

    # X, ymax, and xmin, and ymin of XMAX, YMAX, and xmin, and ymin of the first test box make the dimension, (A, B, 2)
    # Calculate the smaller XMAX and Ymin, the larger of Xmin and Ymin, W = XMAX is smaller-xmin is larger, H = ymax is smaller-ymin is larger
    xy_max = torch.min(bbox1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bbox2[:, 2:].unsqueeze(0).expand(A, B, 2))
    xy_min = torch.max(bbox1[:, :2].unsqueeze(1).expand(A, B, 2),
                       bbox2[:, :2].unsqueeze(0).expand(A, B, 2))

    #intersection area
    inter = (xy_max-xy_min).clamp(min=0)
    inter = inter[:, :, 0]*inter[:, :, 1]

    #At calculation of the area of each rectangle
    area_prior = ((bbox1[:, 2]-bbox1[:, 0]) *
                  (bbox1[:, 3]-bbox1[:, 1])).unsqueeze(1).expand(A, B)
    area_bbox = ((bbox2[:, 2]-bbox2[:, 0]) *
                 (bbox2[:, 3]-bbox2[:, 1])).unsqueeze(0).expand(A, B)

    return inter/(area_prior+area_bbox-inter)


def jaccard_overlap_numpy(box: np.ndarray, boxes: np.ndarray):
    """Calculate a border box and multiple boundary frames, the coordinate form is` (xmin, ymin, xmax, ymax) `` `

    Parameters
    -----------
    Box: `~ np.ndarray` of shape` (4,) `` `
        Border

    Boxes: `~ np.ndarray` of shape` (n, 4) `
        Other boundary frame

    Returns
    --------
    iOU: `~ np.ndarray` of shape` (n,) `` `
        Comparison
    """
    #Acgiotic intersection
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    # 计算 iou
    iou = inter/(area_box+area_boxes-inter)  # type: np.ndarray
    return iou


def ciou(bbox1: Tensor, bbox2: Tensor):
    """the corresponding CIOU of the two sets of boundary boxes one by one

    Parameters
    -----------
    bbox1, bbox2: tensor of shape `(..., 4)` `
        The boundary frame with the same dimension, the form is `(CX, Cy, W, H)`
    """
    # Converted to the corner coordinates form
    xy_min1 = bbox1[..., [0, 1]] - bbox1[..., [2, 3]]/2
    xy_max1 = bbox1[..., [0, 1]] + bbox1[..., [2, 3]]/2
    xy_min2 = bbox2[..., [0, 1]] - bbox2[..., [2, 3]]/2
    xy_max2 = bbox2[..., [0, 1]] + bbox2[..., [2, 3]]/2

    # calculateIOU
    xy_max = torch.min(xy_max1, xy_max2)
    xy_min = torch.max(xy_min1, xy_min2)
    inter = (xy_max-xy_min).clamp(min=0)
    inter = inter[..., 0]*inter[..., 1]
    union = bbox1[..., 2]*bbox1[..., 3] + bbox2[..., 2]*bbox2[..., 3] - inter
    iou = inter/(union+1e-7)

    # Calculation center distance
    center_distance = (torch.pow(bbox1[..., :2]-bbox2[..., :2], 2)).sum(dim=-1)

    # Calculate diagonal distance
    xy_max = torch.max(xy_max1, xy_max2)
    xy_min = torch.min(xy_min1, xy_min2)
    diag_distance = torch.pow(xy_max-xy_min, 2).sum(dim=-1)

    # Calculating scale similarity
    v = 4 / (pi**2) * torch.pow(
        torch.atan(bbox1[..., 2]/bbox1[..., 3].clamp(min=1e-6)) -
        torch.atan(bbox2[..., 2]/bbox2[..., 3].clamp(min=1e-6)), 2
    )
    alpha = v / torch.clamp((1 - iou + v), min=1e-6)

    return iou - center_distance/diag_distance.clamp(min=1e-6) - alpha*v


def center_to_corner(boxes: Tensor):
    """The boundary frame of the form of` (cx, cy, w, h) `is converted to the border frame of the form of the form

    Parameters
    -----------
    boxor of shape `(n, 4)`
        Border
    """
    return torch.cat((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), dim=1)


def center_to_corner_numpy(boxes: ndarray) -> ndarray:
    """The boundary frame of the form of` (cx, cy, w, h) `is converted to the border frame of the form of the form

    Parameters
    -----------
    Boxes: `~ np.ndarray` of shape` (n, 4) `
        Border
    """
    return np.hstack((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2))


def corner_to_center(boxes: Tensor):
    """ Xmin, Ymin, XMAX, YMAX) `The boundary frame of forms of form to` (cx, cy, w, h) `

    Parameters
    -----------
    boxor of shape `(n, 4)`
        Border
    """
    return torch.cat(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]), dim=1)


def corner_to_center_numpy(boxes: ndarray) -> ndarray:
    """ Xmin, Ymin, XMAX, YMAX) `The boundary frame of forms of form to` (cx, cy, w, h) `

    Parameters
    -----------
    Boxes: `~ np.ndarray` of shape` (n, 4) `
        Border
    """
    return np.hstack(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]))


def decode(pred: Tensor, anchors: Union[List[List[int]], np.ndarray], n_classes: int, image_size: int, scale=True):
    """Decodes out of prediction boxes

        Parameters
        -----------
        Pred: tensor of shape `(n, (n_classes+5)*n_anchors, h, w)`
            A feature diagram of neural network output

        anchors: list [list [int]] or `np.ndarray` of shape` (n_anchors, 2) `
            A priori box that does not make a zoom according to the characteristics of the feature diagram

        n_classes: int
            Number of categories

        Image_size: int
            Enter the image size of the neural network

        Scale: BOOL
            Whether the prediction box is reduced to the scale of the original image, the `false` is zoomed in the scale of the feature diagram

        Returns
        --------
        out: tensor of shape `(n, n_anchors, h, w, n_classes+5)` `` ``
            Decoding result
    """
    n_anchors = len(anchors)
    N, _, h, w = pred.shape

    # Adjust the characteristics of the characteristic diagram, facilitate the index, the post -adjustment dimension is (n, n_anchors, h, w, n_classes+5)
    pred = pred.view(N, n_anchors, n_classes+5, h,
                     w).permute(0, 1, 3, 4, 2).contiguous().cpu()

    # Zoom a priori box
    step_h = image_size/h
    step_w = image_size/w
    anchors = [[i/step_w, j/step_h] for i, j in anchors]
    anchors = Tensor(anchors)  # type:Tensor

    # broadcast
    cx = torch.linspace(0, w-1, w).repeat(N, n_anchors, h, 1)
    cy = torch.linspace(0, h-1, h).view(h, 1).repeat(N, n_anchors, 1, w)
    pw = anchors[:, 0].view(n_anchors, 1, 1).repeat(N, 1, h, w)
    ph = anchors[:, 1].view(n_anchors, 1, 1).repeat(N, 1, h, w)

    # decoding
    out = torch.zeros_like(pred)
    out[..., 0] = cx + pred[..., 0].sigmoid()
    out[..., 1] = cy + pred[..., 1].sigmoid()
    out[..., 2] = pw*torch.exp(pred[..., 2])
    out[..., 3] = ph*torch.exp(pred[..., 3])
    out[..., 4:] = pred[..., 4:].sigmoid()

    # The absolute size when the scaling prediction box is to the image size is (Image_size, Image_size)
    if scale:
        out[..., [0, 2]] *= step_w
        out[..., [1, 3]] *= step_h

    return out


def match(anchors: list, anchor_mask: list, targets: List[Tensor], h: int, w: int, n_classes: int, overlap_thresh=0.5):
    """ Match the true value of the first test box and the boundary box, mark the positive and negative sample

    Parameters
    -----------
    anchors: list of shape `(n_anchors*3, 2)`
        A retractable box of a zoom according to the size of the feature diagram

    anchor_mask: list [int] of shape `(n_anchors,)` `
        Priority box index corresponding to the feature diagram

    targets: list [tensor] of shape (n, ((n_objects, 4)))
        The label of multiple pictures, the last dimension is `(CX, Cy, W, H, Class)` `

    h: int
        The height of the feature diagram

    w: int
        The width of the feature diagram

    n_classes: int
        Number of categories

    overlap_thresh: Float
        IOU threshold

    Returns
    --------
    p_mask: tensor of shape `(n, n, n_anchors, h, w)` `
        Covering

    n_mask: tensor of shape `(n, n, n_anchors, h, w)`
        Anti -case cover

    GT: tensor of shape `(n, n_anchors, h, w, n_classes+5)` `` `` `` `` `` `` `` ``
        Label, the last dimension is `(CX, CY, W, H, OBJ, C1, C2, ...)` `` `` `
    """
    N = len(targets)
    n_anchors = len(anchor_mask)

    # Initialization return value
    p_mask = torch.zeros(N, n_anchors, h, w)
    n_mask = torch.ones(N, n_anchors, h, w)
    gt = torch.zeros(N, n_anchors, h, w, n_classes+5)

    # Match the first test box and the border box
    anchors = torch.hstack((torch.zeros((len(anchors), 2)), Tensor(anchors)))

    for i in range(N):
        if len(targets[i]) == 0:
            continue

        # Reverse Return One -time Border Frame
        target = torch.zeros_like(targets[i])  # shape:(n_objects, 5)
        target[:, [0, 2]] = targets[i][:, [0, 2]] * w
        target[:, [1, 3]] = targets[i][:, [1, 3]] * h
        target[:, 4] = targets[i][:, 4]
        bbox = torch.cat((torch.zeros(target.size(0), 2), target[:, 2:4]), 1)

        # Calculating the border box and all the prior frames
        best_indexes = torch.argmax(iou(bbox, anchors), dim=1)

        # Iterate every group truth box
        for j, best_i in enumerate(best_indexes):
            if best_i not in anchor_mask:
                continue

            k = anchor_mask.index(best_i)

            # Get label data
            cx, gw = target[j, [0, 2]]
            cy, gh = target[j, [1, 3]]

            # Get the coordinates of the cell in the center of the border frame
            gj, gi = int(cx), int(cy)

            # Mark the positive and opponent
            p_mask[i, k, gi, gj] = 1
            # Except for the case, the intersecting ratio with Group Truth is less than the threshold, but the negative case
            n_mask[i, k, gi, gj] = 0
            # n_mask[i, iou >= overlap_thresh, gi, gj] = 0

            # Calculate the label value
            gt[i, k, gi, gj, 0] = cx
            gt[i, k, gi, gj, 1] = cy
            gt[i, k, gi, gj, 2] = gw
            gt[i, k, gi, gj, 3] = gh
            gt[i, k, gi, gj, 4] = 1
            gt[i, k, gi, gj, 5+int(target[j, 4])] = 1

    return p_mask, n_mask, gt


def nms(boxes: Tensor, scores: Tensor, overlap_thresh=0.45, top_k=100):
    """ Non -large value suppression, remove the excess prediction box

    Parameters
    -----------
    Boxes: Tensor of Shape `(n_boxes, 4)` `
        Forecast box, the coordinate form is `(CX, Cy, W, H)`

    scores: tensor of shape `(n_boxes,)` `
        The confidence of each prediction box

    overlap_thresh: Float
        IOU threshold, part of the predicted box greater than the threshold will be removed. The larger the value of the value, the more

    top_k: int
        The reserved prediction box is the upper limit

    Returns
    --------
    Indexes: LongTensor of Shape `(n,)` `
        The index of the reserved boundary frame
    """
    keep = []
    if boxes.numel() == 0:
        return torch.LongTensor(keep)

    # The area of each prediction box
    boxes = center_to_corner(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2-x1)*(y2-y1)

    # Sort the score to sort and intercept before TOP_K indexes
    _, indexes = scores.sort(dim=0, descending=True)
    indexes = indexes[:top_k]

    while indexes.numel():
        i = indexes[0]
        keep.append(i)

        # Directly exit the loop when the last index
        if indexes.numel() == 1:
            break

        # Other prediction boxes and current prediction box intersections
        right = x2[indexes].clamp(max=x2[i].item())
        left = x1[indexes].clamp(min=x1[i].item())
        bottom = y2[indexes].clamp(max=y2[i].item())
        top = y1[indexes].clamp(min=y1[i].item())
        inter = ((right-left)*(bottom-top)).clamp(min=0)

        # Calculate iOU
        iou = inter/(area[i]+area[indexes]-inter)

        # Keep the boundary box with less than the threshold, and your own iOU is 1
        indexes = indexes[iou < overlap_thresh]

    return torch.LongTensor(keep)


def draw(image: Union[ndarray, Image.Image], bbox: ndarray, label: ndarray, conf: ndarray = None) -> Image.Image:
    """ Draw the boundary box and label on the image

    Parameters
    -----------
    Image: `~ np.ndarray` of shape` (h, w, 3) `or pil.image.image`
        RGB image

    bbox: `~ np.ndarray` of shape` (N_Objects, 4) ``
        Border frame, coordinate form is `(CX, Cy, W, H)`

    label: Iterable of shape `(n_objects,)` ``
        Label

    conf: Iterable of shape `(n_Objects,)` ``
        Confidence
    """
    bbox = center_to_corner_numpy(bbox).astype(np.int)

    if isinstance(image, ndarray):
        image = Image.fromarray(image.astype(np.uint8))  # type:Image.Image

    image_draw = ImageDraw.Draw(image, 'RGBA')
    font = ImageFont.truetype('resource/font/msyh.ttc', size=13)

    label_unique = np.unique(label).tolist()
    color_indexes = np.linspace(0, 255, len(label_unique), dtype=int)

    for i in range(bbox.shape[0]):
        x1 = max(0, bbox[i, 0])
        y1 = max(0, bbox[i, 1])
        x2 = min(image.width-1, bbox[i, 2])
        y2 = min(image.height-1, bbox[i, 3])

        # choose the color
        class_index = label_unique.index(label[i])
        color = to_hex_color(cmapy.color(
            'rainbow', color_indexes[class_index], True))

        # Drawing square
        image_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        y1_ = y1 if y1-23 < 0 else y1-23
        y2_ = y1 if y1_ < y1 else y1+23
        text = label[i] if conf is None else f'{label[i]} | {conf[i]:.2f}'
        l = font.getlength(text) + 3
        right = x1+l if x1+l <= image.width-1 else image.width-1
        left = int(right - l)
        image_draw.rectangle([left, y1_, right, y2_],
                             fill=color+'AA', outline=color+'DD')
        image_draw.text([left+2, y1_+2], text=text,
                        font=font, embedded_color=color)

    return image


def to_hex_color(color):
    """ Convert the color to hexadecimal """
    color = [hex(c)[2:].zfill(2) for c in color]
    return '#'+''.join(color)


def rescale_bbox(bbox: ndarray, image_size: int, h: int, w: int):
    """ The prediction box for the picture after being filled and scaled

    Parameters
    -----------
    bbox: `~ np.ndarray` of shape` (N_Objects, 4) ``
        Forecast box, the coordinate form is `(CX, Cy, W, H)`

    Image_size: int
        The size after the image is scaled

    h: int
        The height of the original image

    w: int
        The width of the original image

    Returns
    --------
    bbox: `~ np.ndarray` of shape` (N_Objects, 4) ``
        Forecast box, the coordinate form is `(CX, Cy, W, H)`
    """
    # Image fill area size
    pad_x = max(h-w, 0)*image_size/max(h, w)
    pad_y = max(w-h, 0)*image_size/max(h, w)

    # Effective image area in images after being scaled
    w_ = image_size - pad_x
    h_ = image_size - pad_y

    # Restore border frame
    bbox = center_to_corner_numpy(bbox)
    bbox[:, [0, 2]] = (bbox[:, [0, 2]]-pad_x/2)*w/w_
    bbox[:, [1, 3]] = (bbox[:, [1, 3]]-pad_y/2)*h/h_
    bbox = corner_to_center_numpy(bbox)
    return bbox
