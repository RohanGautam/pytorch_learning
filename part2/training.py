# import argparse
import sys
import os
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from utils.log_config import logging  # noqa
from utils.dsets import getCandidateInfo, getCt, LunaDataset  # noqa


# print(sys.path)
log = logging.getLogger(__name__)

log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class LunaTrainingApp:
    def __init__(self, sys_argv=None) -> None:
        if sys_argv is None:
            # get arguments, excluding the file name
            sys_argv = sys.argv[1:]
        # define a parser for the arguments collected above
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--num-workers', help='Number of workers for data loading', default=8, type=int)
        parser.add_argument(
            '--batch-size', help='Number of workers for data loading', default=32, type=int)
        parser.add_argument(
            '--epochs', help='Number of workers for data loading', default=1, type=int)
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        # using dataloaders is great because it eagerly loads the,
        # eg: if the first one is consumed, then it loads the next batch for consumption in the next iteration
        self.train_dl = self.initTrainDataLoader()
        self.val_dl = self.initValDataLoader()

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                # distribute within gpu's on this machine.
                # look at `DistributedDataParallel` if you want to do it b/w machines
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        # we need to move the model to the GPU *before* we initialize the optimizer
        # "momentum" means the weight updates will incur accelleration based on the gradient of the loss function
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDataLoader(self):
        train_ds = LunaDataset(
            val_stride=10,
            is_validation_bool=False  # get the training data
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        # will add a "batch" dimension at the 0th indexs
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,  # will parallelize data loading
            pin_memory=self.use_cuda
        )

    def initValDataLoader(self):
        train_ds = LunaDataset(
            val_stride=10,
            is_validation_bool=True  # get the validation data
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        # will add a "batch" dimension at the 0th indexs
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )


class LunaModel:
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    LunaTrainingApp().main()
