# coding:utf-8
from typing import List

import cv2 as cv
import numpy as np
from imutils.video import FPS, WebcamVideoStream
from net import Yolo


def image_detect(model_path: str, image_path: str, classes: List[str], image_size=416,
                 anchors=None, conf_thresh=0.6, nms_thresh=0.45, use_gpu=True, show_conf=True):
    """ Detect the target in the image

    Parameters
    -----------
    Model_path: STR
        Model path

    Image_path: STR
        Picture path

    CLASSSES: List [STR]
        Category name list

    Image_size: int
        Enter the picture size of the neural network, which must be the multiple of 32

    anchors: list of shape `(1, 3, n_anchors, 2)` `
        The image size of the input neural network is 416, the size is from large to small

    conf_thresh: Float
        The confidence threshold, the prediction box with less than this threshold will not be displayed

    nms_thresh: float
        The non -extremely large value inhibitory threshold, the smaller the value reserved, the less

    use_gpu: BOOL
        Whether to use GPU acceleration detection

    show_conf: BOOL
        Whether to show confidence

    Returns
    --------
    Image: ~ pil.image.image`
        Draw an image of the boundary box, category and confidence
    """
    # Create a model
    model = Yolo(len(classes), image_size, anchors, conf_thresh, nms_thresh)
    if use_gpu:
        model = model.cuda()

    # Loading model
    model.load(model_path)
    model.eval()

    # Detection goals
    return model.detect(image_path, classes, use_gpu, show_conf)


def camera_detect(model_path: str, classes: List[str], image_size: int = 416, anchors: list = None,
                  camera_src=0, conf_thresh=0.6, nms_thresh=0.45, use_gpu=True):
    """ Detect objects from real -time detection from the camera

    Parameters
    -----------
    Model_path: STR
        Model path

    CLASSSES: List [STR]
        Category

    Image_size: int
        Enter the picture size of the neural network, which must be the multiple of 32

    any: List
        The image size of the input neural network is 416, first check the frame

    Camera_src: int
        Camera source, 0 represents the default camera

    conf_thresh: Float
        The confidence threshold, the prediction box with less than this threshold will not be displayed

    nms_thresh: float
        The threshold in which non -extremely large value suppression, the larger the more reserved prediction box, the more

    use_gpu: BOOL
        Whether to use GPU acceleration detection
    """
    # Create a model
    model = Yolo(len(classes), image_size, anchors, conf_thresh, nms_thresh)
    if use_gpu:
        model = model.cuda()

    # Loading model
    model.load(model_path)
    model.eval()

    # Create frame rate statistics device
    fps = FPS().start()

    print('📸 In the detection object, press Q to exit...')

    # Open the camera
    stream = WebcamVideoStream(src=camera_src).start()
    while True:
        image = stream.read()
        image = np.array(model.detect(image, classes, use_gpu))
        fps.update()

        # Display detection results
        cv.imshow('camera detection', image)

        # exit the program
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print(f'The test is over, frame rate：{fps.fps()} FPS')
