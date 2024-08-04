# coding: utf-8
from typing import Dict
from xml.etree import ElementTree as ET



class AnnotationReader:
    """ annotation coverter for xml format """

    def __init__(self, class_to_index: Dict[str, int], keep_difficult=False):
        """
        Parameters
        ----------
        class_to_index: Dict[str, int]
             category encoding dictionary
        keep_difficulty: bool
            Whether to keep samples with difficult as 1
        """
        self.class_to_index = class_to_index
        self.keep_difficult = keep_difficult

    def read(self, file_path: str):
        """ parse xml tag file

        Parameters
        ----------
        file_path: str
            file path

        Returns
        -------
        target: List[list] of shape `(n_objects, 5)`
            A list of labels, the first four elements of each label are the normalized bounding box, and the last label is the encoded category,
            e.g. `[[xmin, ymin, xmax, ymax, class], ...]`
        """
        root = ET.parse(file_path).getroot()

        # image size
        img_size = root.find('size')
        w = int(img_size.find('width').text)
        h = int(img_size.find('height').text)

        # extract all tags
        target = []
        for obj in root.iter('object'):
            # Is the sample unpredictable
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # normalized box position
            points = ['xmin', 'ymin', 'xmax', 'ymax']
            data = []
            for i, pt in enumerate(points):
                pt = int(bbox.find(pt).text) - 1
                pt = pt/w if i % 2 == 0 else pt/h
                data.append(pt)

            # Check if the data is legal
            if data[0] >= data[2] or data[1] >= data[3]:
                p = [int(bbox.find(pt).text) for pt in points]
                raise ValueError(f"{file_path} 存在脏数据：object={name}, bbox={p}")

            data.append(self.class_to_index[name])
            target.append(data)

        return target


