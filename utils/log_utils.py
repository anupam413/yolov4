# coding:utf-8
import json
import logging
import os
import weakref
from pathlib import Path


class LossLogger:
    """ Loss recorder """

    def __init__(self, log_file: str = None, save_dir='log'):
        """
Parameters
        -----------
        log_file: STR
            Loss data history record file, the requirements are JSON files

        Save_dir: STR
            The folder saved by the loss data
        """
        self.log_file = log_file
        self.save_dir = Path(save_dir)
        self.losses = []

        # Load historical data
        if log_file:
            self.load(log_file)

    def record(self, loss: float):
        """ Add a loss record """
        self.losses.append(loss)

    def load(self, file_path: str):
        """ Loading historical record data """
        if not os.path.exists(file_path):
            raise FileNotFoundError("损失历史纪录文件不存在，请检查文件路径！")

        try:
            with open(file_path, encoding='utf-8') as f:
                self.losses = json.load(f)  # type:list
        except:
            raise Exception("json 文件损坏，无法正确读取内容！")

    def save(self, file_name: str):
        """ Save the recorded data

        Parameters
        -----------
        file_name: STR
            File name, does not include `.json` suffix
        """
        self.save_dir.mkdir(exist_ok=True, parents=True)
        with open(self.save_dir/f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.losses, f)


_loggers = weakref.WeakValueDictionary()


def loggerCache(cls):
    """ decorator for caching logger """

    def wrapper(name, *args, **kwargs):
        if name not in _loggers:
            instance = cls(name, *args, **kwargs)
            _loggers[name] = instance
        else:
            instance = _loggers[name]

        return instance

    return wrapper


@loggerCache
class Logger:
    """ 日志记录器 """

    log_folder = Path('log')

    def __init__(self, fileName: str):
        """
        Parameters
        ----------
        fileName: str
            log filename which doesn't contain `.log` suffix
        """
        self.log_folder.mkdir(exist_ok=True, parents=True)
        self._log_file = self.log_folder/(fileName+'.log')
        self._logger = logging.getLogger(fileName)
        self._console_handler = logging.StreamHandler()
        self._file_handler = logging.FileHandler(
            self._log_file, encoding='utf-8')

        # set log level
        self._logger.setLevel(logging.DEBUG)
        self._console_handler.setLevel(logging.DEBUG)
        self._file_handler.setLevel(logging.DEBUG)

        # set log format
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self._console_handler.setFormatter(fmt)
        self._file_handler.setFormatter(fmt)

        self._logger.addHandler(self._console_handler)
        self._logger.addHandler(self._file_handler)

    def info(self, msg):
        self._logger.info(msg)

    def error(self, msg, exc_info=False):
        self._logger.error(msg, exc_info=exc_info)

    def debug(self, msg):
        self._logger.debug(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def critical(self, msg):
        self._logger.critical(msg)
