# coding:utf-8
import time
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from torch import cuda
from torch.backends import cudnn
from tqdm import tqdm
from utils.log_utils import LossLogger, Logger
from utils.datetime_utils import time_delta
from utils.lr_schedule_utils import WarmUpCosLRSchedule, determin_lr, get_lr
from utils.optimizer_utils import make_optimizer

from .dataset import VOCDataset, make_data_loader
from .loss import YoloLoss
from .yolo import Yolo


def exception_handler(train_func):
    """ Treatment of abnormalities in the training process and save the model """
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            return train_func(train_pipeline, *args, **kwargs)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                Logger("error").error(f"{e.__class__.__name__}: {e}", True)

            train_pipeline.save()

            # Clear GPU cache
            cuda.empty_cache()

            exit()

    return wrapper


class TrainPipeline:
    """ Training model assembly line """

    def __init__(self, n_classes: int, image_size: int, anchors: list, dataset: VOCDataset, darknet_path: str = None,
                 yolo_path: str = None, lr=0.01, momentum=0.9, weight_decay=4e-5, warm_up_ratio=0.02, freeze=True,
                 batch_size=4, freeze_batch_size=8, num_workers=4, freeze_epoch=20, start_epoch=0, max_epoch=60,
                 save_frequency=5, use_gpu=True, save_dir='model', log_file: str = None, log_dir='log'):
        """
        Parameters
        -----------
        n_classes: int
            Number of categories

        Image_size: int
            Enter the picture size of the YOLO neural network

        Anchors: list of shape `(3, n_anchors, 2)`
            The picture size of the input neural network is 416.

        dataset: dataset
            Training data set

        darknet_path: STR
            Pre -trained Darknet53 model file path

        YOLO_PATH: Union [Str, None]
            YOLO model file path, there are two options:
            * If it is not `none`, initialize the parameters in the model file` yolo`
            * If it is `None`, the various layers of the various layers of darknet53 will be randomized

        LR: Float
            Learning rate

        Momentum: Float
            Impulse

        weight_decay: Float
            Weighted attenuation

        Warm_up_ratio: Float
            The generation of warm start in the proportion of the whole generation

        Freeze: BOOL
            Whether to use frozen training

        BATCH_SIZE: int
            The batch size of the non -freezing training process training set

        freeze_batch_size: int
            The batch size in the process of freezing training

        num_Workers: int
            Number of threads of loading data

        freeze_epoch: int
            Frozen training generation

        start_epoch: int
            The parameters containing the YOLO model file are the results of how many generations have been trained

        max_epoch: int
            How many generations iterate

        save_frequency: int
            How many iterations are preserved once a model

        use_gpu: BOOL
            Whether to use GPU acceleration training

        Save_dir: STR
            Save the SSD model folder

        log_file: STR
            Training loss data history record file, the requirements are JSON files

        Save_dir: STR
            Files preserved by training loss data
        """
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.use_gpu = use_gpu
        self.save_frequency = save_frequency
        self.freeze_batch_size = freeze_batch_size
        self.batch_size = batch_size

        self.lr = lr
        self.freeze = freeze
        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.free_epoch = freeze_epoch

        if use_gpu and cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # Create a model
        self.model = Yolo(n_classes, image_size, anchors).to(self.device)
        if yolo_path:
            self.model.load(yolo_path)
            print('🧪 Successfully loaded YOLO model：' + yolo_path)
        elif darknet_path:
            self.model.backbone.load(darknet_path)
            print('🧪 Successfully loaded Darknet53 model：' + darknet_path)
        else:
            raise ValueError("The pre -trained Darknet53 model file path must be specified")

        self.model.backbone.set_freezed(freeze)

        # Create optimizers and loss functions
        bs = freeze_batch_size if freeze else batch_size
        lr_fit, lr_min_fit = determin_lr(lr, bs)
        self.criterion = YoloLoss(anchors, n_classes, image_size)
        self.optimizer = make_optimizer(
            self.model, lr_fit, momentum, weight_decay)
        self.lr_schedule = WarmUpCosLRSchedule(
            self.optimizer, lr_fit, lr_min_fit, max_epoch, warm_up_ratio)

        # Data set loader
        self.num_worksers = num_workers
        self.n_batches = len(self.dataset)//bs
        self.data_loader = make_data_loader(self.dataset, bs, num_workers)

        # Training loss recorder
        self.logger = LossLogger(log_file, log_dir)

    def save(self):
        """ Save the model and training loss data"""
        self.pbar.close()
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Save the model
        self.model.eval()
        path = self.save_dir/f'Yolo_{self.current_epoch+1}.pth'
        torch.save(self.model.state_dict(), path)

        # Save training loss data
        self.logger.save(f'train_losses_{self.current_epoch+1}')

        print(f'\n🎉 Save the current model tocurrent model to {path.absolute()}\n')

    @exception_handler
    def train(self):
        """ Training modelning model """
        t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        self.save_dir = self.save_dir/t
        self.logger.save_dir = self.logger.save_dir/t

        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        print('🚀 Start training！')

        is_unfreezed = False
        for e in range(self.start_epoch, self.max_epoch):
            self.current_epoch = e

            # Thawing training
            if self.freeze and e >= self.free_epoch and not is_unfreezed:
                print('\n🧊 Start thawing training!\n')
                is_unfreezed = True
                self.lr_schedule.set_lr(*determin_lr(self.lr, self.batch_size))
                self.data_loader = make_data_loader(
                    self.dataset, self.batch_size, self.num_worksers)
                self.n_batches = len(self.dataset)//self.batch_size
                self.model.backbone.set_freezed(False)

            self.model.train()

            # Create progress bar
            self.pbar = tqdm(total=self.n_batches, bar_format=bar_format)
            self.pbar.set_description(f"\33[36m💫 Epoch {(e+1):5d}/{self.max_epoch}")
            start_time = datetime.now()

            loss_value = 0
            for iter, (images, targets) in enumerate(self.data_loader, 1):
                # Probability of the border frame, confidence and conditional category
                preds = self.model(images.to(self.device))

                # Error reverse propagation
                self.optimizer.zero_grad()
                loss = 0
                for i, pred in enumerate(preds):
                    loss += self.criterion(i, pred, targets)

                loss.backward()
                self.optimizer.step()

                # Record error
                loss_value += loss.item()

                # Update progress bar
                cost_time = time_delta(start_time)
                self.pbar.set_postfix_str(
                    f'loss: {loss_value/iter:.4f}, lr: {get_lr(self.optimizer):.5f}, time: {cost_time}\33[0m')
                self.pbar.update()

            # Write errors into logs
            self.logger.record(loss_value/iter)

            # Close progress bar
            self.pbar.close()

            # Learning rate annealing
            self.lr_schedule.step(e)

            # Turn off the mosaic data enhancement
            if e == self.max_epoch - self.lr_schedule.no_aug_epoch:
                self.dataset.use_mosaic = False

            # Stay on a regular basis
            if e > self.start_epoch and (e+1-self.start_epoch) % self.save_frequency == 0:
                self.save()

        self.save()
