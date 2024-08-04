# coding:utf-8
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from torch import cuda
from PIL import Image
from prettytable import PrettyTable
from utils.box_utils import jaccard_overlap_numpy, center_to_corner_numpy, rescale_bbox
from utils.augmentation_utils import ToTensor

from .dataset import VOCDataset
from .yolo import Yolo


class EvalPipeline:
    """ Test model assembly line """

    def __init__(self, model_path: str, dataset: VOCDataset, image_size=416, anchors: list = None,
                 conf_thresh=0.05, overlap_thresh=0.5, save_dir='eval', use_07_metric=False, use_gpu=True):
        """
        Parameters
        -----------
        Model_path: STR
            Model file path

        dataset: vocdataset
            data set

        Image_size: int
            Image size

        any: List
            The image size of the image of the neural network is 416,

        conf_thresh: Float
            Confidence threshold

        overlap_thresh: Float
            IOU threshold

        Save_dir: STR
            The preservation directory of test results and prediction results files

        use_07_metric: bool
            Whether to use the AP calculation method of VOC2007

        use_gpu: BOOL
            Whether to use GPUto use GPU
        """
        self.use_gpu = use_gpu
        self.dataset = dataset
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.overlap_thresh = overlap_thresh
        self.use_07_metric = use_07_metric
        self.save_dir = Path(save_dir)

        self.model_path = Path(model_path)
        self.device = 'cuda' if use_gpu and cuda.is_available() else 'cpu'
        self.model = Yolo(self.dataset.n_classes, image_size, anchors)
        self.model.detector.conf_thresh = conf_thresh
        self.model = self.model.to(self.device)
        self.model.load(model_path)
        self.model.eval()

    @torch.no_grad()
    def eval(self):
        """ Test model, get MAP """
        self._predict()
        self._get_ground_truth()
        return self._get_mAP()

    def _predict(self):
        """ What pictures are predicted in which categories exist """
        self.preds = {c: {} for c in self.dataset.classes}
        transformer = ToTensor(self.image_size)

        print('🛸 Being predicted...')
        for i, (image_path, image_name) in enumerate(zip(self.dataset.image_paths, self.dataset.image_names)):
            print(f'\rCurrent progress：{i/len(self.dataset):.0%}', end='')

            # Read the picture
            image = np.array(Image.open(image_path).convert('RGB'))
            h, w, _ = image.shape

            # predict
            x = transformer.transform(image).to(self.device)
            out = self.model.predict(x)
            if not out:
                continue

            for c, pred in out[0].items():
                pred = pred.numpy()
                mask = pred[:, 0] > self.conf_thresh

                # If there is no confidence in the border, it is greater than the threshold.
                if not mask.any():
                    continue

                # Filter the boundary frame that meets the threshold conditions
                conf = pred[:, 0][mask]  # type:np.ndarray
                bbox = rescale_bbox(pred[:, 1:][mask], self.image_size, h, w)
                bbox = center_to_corner_numpy(bbox)

                # Save the prediction results
                self.preds[self.dataset.classes[c]][image_name] = {
                    "bbox": bbox.tolist(),
                    "conf": conf.tolist()
                }

    def _get_ground_truth(self):
        """ What pictures do each type in GROUND TRUTH exist in? """
        self.ground_truths = {c: {} for c in self.dataset.classes}
        self.n_positives = {c: 0 for c in self.dataset.classes}

        print('\n\n🧩 Get the label...')
        for i, (anno_path, img_name) in enumerate(zip(self.dataset.annotation_paths, self.dataset.image_names)):
            print(f'\rCurrent progress：{i/len(self.dataset):.0%}', end='')

            root = ET.parse(anno_path).getroot()

            for obj in root.iter('object'):
                # Get the class and boundary frame contained in the label
                c = obj.find('name').text.lower().strip()
                difficult = int(obj.find('difficult').text)
                bbox = obj.find('bndbox')
                bbox = [
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                ]

                if not self.ground_truths[c].get(img_name):
                    self.ground_truths[c][img_name] = {
                        "bbox": [],
                        "detected": [],
                        "difficult": []
                    }

                # Add a GROUND TRUTH record
                self.ground_truths[c][img_name]['bbox'].append(bbox)
                self.ground_truths[c][img_name]['detected'].append(False)
                self.ground_truths[c][img_name]['difficult'].append(difficult)
                self.n_positives[c] += (1-difficult)

    def _get_mAP(self):
        """ Calculate map """
        result = {}

        print('\n\n🧪 In calculating AP...')
        mAP = 0
        table = PrettyTable(["class", "AP"])
        for c in self.dataset.classes:
            ap, precision, recall = self._get_AP(c)
            result[c] = {
                'AP': ap,
                'precision': precision,
                'recall': recall
            }
            mAP += ap
            table.add_row([c, f"{ap:.2%}"])

        mAP /= len(self.dataset.classes)
        table.add_column("mAP", [f"{mAP:.2%}"] + [""]*(len(self.dataset.classes)-1))
        print(table)

        # Save the evaluation results
        self.save_dir.mkdir(exist_ok=True, parents=True)
        p = self.save_dir / (self.model_path.stem + '_AP.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(result, f)

        return mAP

    def _get_AP(self, c: str):
        """ Calculate an AP of a class

        Parameters
        -----------
        C: STR
            Category name

        Returns
        --------
        AP: Float
            AP, return without predicting this class 0 0

        Precision: List
            Check

        recal: List
            Check the whole rate
        """
        pred = self.preds[c]
        ground_truth = self.ground_truths[c]
        bbox = []
        conf = []
        image_names = []

        # Stitch bbox into a two -dimensional matrix, each behavior of one prediction box
        for image_name, v in pred.items():
            image_names.extend([image_name]*len(v['conf']))
            bbox.append(v['bbox'])
            conf.append(v['conf'])

        # This class is not predicted in any picture
        if not bbox:
            return 0, 0, 0

        bbox = np.vstack(bbox)  # type:np.ndarray
        conf = np.hstack(conf)  # type:np.ndarray
        image_names = np.array(image_names)

        # Sorting prediction box according to the confidence order
        index = np.argsort(-conf)
        bbox = bbox[index]
        conf = conf[index]
        image_names = image_names[index]

        # Calculate TP and FP
        tp = np.zeros(len(image_names))  # type:np.ndarray
        fp = np.zeros(len(image_names))  # type:np.ndarray
        for i, image_name in enumerate(image_names):
            # Get a group in the picture in the picture
            record = ground_truth.get(image_name)

            # FP+1 in the group of this picture without this class, FP+1
            if not record:
                fp[i] = 1
                continue

            bbox_pred = bbox[i]  # shape:(4, )
            bbox_gt = np.array(record['bbox'])  # shape:(n, 4)
            difficult = np.array(record['difficult'], np.bool)  # shape:(n, )

            # Calculation
            iou = jaccard_overlap_numpy(bbox_pred, bbox_gt)
            iou_max = iou.max()
            iou_max_index = iou.argmax()

            if iou_max < self.overlap_thresh:
                fp[i] = 1
            elif not record['difficult'][iou_max_index]:
                # The boundary frame of the prediction box has been matched, and the forecast box cannot be matched again
                if not record['detected'][iou_max_index]:
                    tp[i] = 1
                    record['detected'][iou_max_index] = True
                else:
                    fp[i] = 1

        # 查全率和查准率
        tp = tp.cumsum()
        fp = fp.cumsum()
        n_positives = self.n_positives[c]
        recall = tp / n_positives  # type:np.ndarray
        precision = tp / (tp + fp)  # type:np.ndarray

        # calculate AP
        if not self.use_07_metric:
            rec = np.concatenate(([0.], recall, [1.]))
            prec = np.concatenate(([0.], precision, [0.]))

            # Calculate the package line of the PR curve
            for i in range(prec.size-1, 0, -1):
                prec[i - 1] = np.maximum(prec[i - 1], prec[i])

            # Find out the index when the change of recall
            i = np.where(rec[1:] != rec[:-1])[0]

            # Use the interval between recall
            ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
        else:
            ap = 0
            for r in np.arange(0, 1.1, 0.1):
                if np.any(recall >= r):
                    ap += np.max(precision[recall >= r])/11

        return ap, precision.tolist(), recall.tolist()
