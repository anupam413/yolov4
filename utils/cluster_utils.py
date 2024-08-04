# coding:utf-8
import glob
from xml.etree import ElementTree as ET
from random import choice

import numpy as np

from utils.box_utils import jaccard_overlap_numpy as iou


class AnchorKmeans:
    """ Phase box cluster"""

    def __init__(self, annotation_dir: str):
        self.annotation_dir = annotation_dir
        self.bbox = self.get_bbox()

    def get_bbox(self) -> np.ndarray:
        """ Get all the boundary frames """
        bbox = []

        for path in glob.glob(f'{self.annotation_dir}/*xml'):
            root = ET.parse(path).getroot()

            # The width and height of the image
            w = int(root.find('size/width').text)
            h = int(root.find('size/height').text)

            if w==0:
                print(path)

            # Get all boundary boxes
            for obj in root.iter('object'):
                box = obj.find('bndbox')

                # Normalized coordinates
                xmin = int(box.find('xmin').text)/w
                ymin = int(box.find('ymin').text)/h
                xmax = int(box.find('xmax').text)/w
                ymax = int(box.find('ymax').text)/h

                bbox.append([0, 0, xmax-xmin, ymax-ymin])

        return np.array(bbox)

    def get_cluster(self, n_clusters=9, metric=np.median):
        """ Get cluster results

        Parameters
        -----------
        n_clusters: int
            Cluster

        Metric: Callace
            How to select the cluster center point
        """
        rows = self.bbox.shape[0]

        if rows < n_clusters:
            raise ValueError("n_clusters Can not be greater than the number of sample sample samples")

        last_clusters = np.zeros(rows)
        clusters = np.ones((n_clusters, 2))
        distances = np.zeros((rows, n_clusters))  # type:np.ndarray

        # Select a few points randomly as a cluster center
        np.random.seed(1)
        clusters = self.bbox[np.random.choice(rows, n_clusters, replace=False)]

        # Start a cluster
        while True:
            # Calculation distance
            distances = 1-self.iou(clusters)

            # Drive each boundary box into a cluster
            nearest_clusters = distances.argmin(axis=1)

            # If the cluster center no longer changes, exit
            if np.array_equal(nearest_clusters, last_clusters):
                break

            # Re -select the cluster center
            for i in range(n_clusters):
                clusters[i] = metric(self.bbox[nearest_clusters == i], axis=0)

            last_clusters = nearest_clusters

        return clusters[:, 2:]

    def average_iou(self, clusters: np.ndarray):
        """ Calculate IOU average

        Parameters
        -----------
        clusters: `~ np.ndarray` of shape` (n_clusters, 2) `
            Cluster center
        """
        clusters = np.hstack((np.zeros((clusters.shape[0], 2)), clusters))
        return np.mean([np.max(iou(bbox, clusters)) for bbox in self.bbox])

    def iou(self, clusters: np.ndarray):
        """ Calculate the combination of all boundary boxes to all cluster centers

        Parameters
        -----------
        clusters: ~ np.ndarray` of shape `(n_clusters, 4)` `
            Cluster center

        Returns
        --------
        iOU: `~ np.ndarray` of shape` (n_bbox, n_clusters) ``
            Comparison
        """
        bbox = self.bbox
        A = self.bbox.shape[0]
        B = clusters.shape[0]

        xy_max = np.minimum(bbox[:, np.newaxis, 2:].repeat(B, axis=1),
                            np.broadcast_to(clusters[:, 2:], (A, B, 2)))
        xy_min = np.maximum(bbox[:, np.newaxis, :2].repeat(B, axis=1),
                            np.broadcast_to(clusters[:, :2], (A, B, 2)))

        # Calculate intersection area
        inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
        inter = inter[:, :, 0]*inter[:, :, 1]

        # Calculate the area of each matrix
        area_bbox = ((bbox[:, 2]-bbox[:, 0])*(bbox[:, 3] -
                     bbox[:, 1]))[:, np.newaxis].repeat(B, axis=1)
        area_clusters = ((clusters[:, 2] - clusters[:, 0])*(
            clusters[:, 3] - clusters[:, 1]))[np.newaxis, :].repeat(A, axis=0)

        return inter/(area_bbox+area_clusters-inter)


if __name__ == '__main__':
    root = './dataset/root-roadsign/Annotations'
    model = AnchorKmeans(root)
    clusters = model.get_cluster(9)
    clusters = np.array(sorted(clusters.tolist(), key=lambda i: i[0]*i[1], reverse=True))

    # Restore the first test box to the original size
    print('Cluster:\n', (clusters*416).astype(int))
    print('average IOU:', model.average_iou(clusters))
