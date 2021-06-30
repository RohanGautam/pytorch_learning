# import argparse
import sys
import os
import argparse
import datetime
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import DataLoader

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# 'noqa' makes autopep8 ignore the line in formatting
from utils.log_config import logging  # noqa
from utils.dsets import getCandidateInfo, getCt, LunaDataset  # noqa


# print(sys.path)
log = logging.getLogger(__name__)

log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3
METRICS_SIZE = 3


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

        self.totalTrainingSamples_count = 0

    def main(self):
        '''We define the main training loop here!
        - load batch
        - classify batch
        - calc loss, record metrics
        - update weights
        '''
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        # using dataloaders is great because it eagerly loads the,
        # eg: if the first one is consumed, then it loads the next batch for consumption in the next iteration
        train_dl = self.initTrainDataLoader()
        val_dl = self.initValDataLoader()
        log.info("loaded data")
        for epoch_num in range(1, self.cli_args.epochs+1):
            train_metrics = self.doTraining(epoch_num, train_dl)
            self.logMetrics(epoch_num, 'trn', train_metrics)
            val_metrics = self.doValidation(epoch_num, val_dl)
            self.logMetrics(epoch_num, 'val', val_metrics)

    def logMetrics(self, epoch_ndx, mode_str, metrics, classificationThreshold=0.5):
        # masks are basically boolean arrays which can be used as an _index_ to filter out arrays
        neg_label_masks = metrics[METRICS_LABEL_NDX] <= classificationThreshold
        neg_pred_masks = metrics[METRICS_PRED_NDX] <= classificationThreshold
        pos_label_masks = ~neg_label_masks
        pos_pred_masks = ~neg_pred_masks

        # collect some statistics
        neg_count = int(neg_label_masks.sum())
        pos_count = int(pos_label_masks.sum())
        neg_correct = int((neg_label_masks & neg_pred_masks).sum())
        pos_correct = int((pos_label_masks & pos_pred_masks).sum())

        metrics_dict = {}
        # computing a per class loss helps narrow down if one class is harder to classify
        metrics_dict['loss/all'] = metrics[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics[METRICS_LOSS_NDX,
                                           neg_label_masks].mean()
        metrics_dict['loss/pos'] = metrics[METRICS_LOSS_NDX,
                                           pos_label_masks].mean()

        metrics_dict['correct/all'] = ((pos_correct+neg_correct) /
                                       np.float32(metrics.shape[1])) * 100
        metrics_dict['correct/neg'] = (neg_correct / np.float32(neg_count))*100
        metrics_dict['correct/pos'] = (pos_correct / np.float32(pos_count))*100
        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

    def doTraining(self, epoch_num, train_dl):
        self.model.train()  # set to training mode. things like batchnorm not used during evaluation
        metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        for i, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()  # clear accumulated gradients
            loss_batch = self.computeBatchLoss(
                i, batch_tuple, train_dl.batch_size, metrics)
            loss_batch.backward()  # calculate gradients
            self.optimizer.step()  # update weights

        self.totalTrainingSamples_count += len(train_dl.dataset)  # ?? wtf ??
        return metrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()  # evaluation mode (no weight updates)
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            for batch_ndx, batch_tup in enumerate(val_dl):
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
                # no updates whatsoever boi, thats the whole point of a validation set

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_idx, batch_tuple, batch_size, metrics_g):
        # at each index, we have numerous batches, hence the plural variable names.
        input_matrices, labels, _, _ = batch_tuple

        inputs_gpu = input_matrices.to(self.device, non_blocking=True)
        labels_gpu = labels.to(self.device, non_blocking=True)

        # calling the instantiated `LunaModel` like this executes the foward pass
        # log.debug(f"Input shape is f{inputs_gpu.shape}")
        logits, out_probability = self.model(inputs_gpu)
        # so we get a tensor which we average later on. This is for tracking metrics per sample
        # log.debug(f"got the results")
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        # log.debug(f"labels: {labels_gpu[:, 1]}")
        # log.debug(f"logits: {logits}")

        # [:,-1] is just getting the labels into a 1d form (i think)
        loss = loss_fn(logits, labels_gpu[:, 1])

        start_ndx = batch_idx * batch_size
        end_ndx = start_ndx + labels.size(0)

        # can use per-instance metric to see which image is the problem
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            labels_gpu[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            out_probability[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss.detach()
        return loss.mean()

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


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        # convolutions are here because we're learning their weights as well
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=conv_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=conv_channels,
                               out_channels=conv_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, input_batch):
        out = F.relu(self.conv1(input_batch), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        return F.max_pool3d(out, 2, 2)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8) -> None:
        super().__init__()
        # normalizes inputs. 1 is the number of channels, and since images are single intensity (unlike RGB), that value is 1
        self.batchnorm = nn.BatchNorm3d(1)
        # each block ends with a 2x2 pool, so resolution is reduces 16x by the end of the 4 blocks
        # ie, a 32x48x48 image becomes 2x3x3 with 8*8 =64 channels at the end
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels*2)
        self.block3 = LunaBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaBlock(conv_channels*4, conv_channels*8)

        # two output features
        self.linear_fc = nn.Linear(2*3*3*64, 2)
        # dimension along which to calculate softmax
        # every slice per dimension will sum to 1
        # here dim=1 points to the channel dimension, as the index 0 is the batch
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, input_batch):
        out = self.batchnorm(input_batch)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # we now have to flatten the above into a 1D vector so we can pass it into the linear layer
        flattened_tensor = out.view(
            out.size(0),  # batch size
            -1  # flatten the remaining
        )
        linear_output = self.linear_fc(flattened_tensor)

        return (
            linear_output,  # for caclulating crossentropy loss while training
            self.softmax(linear_output)  # for actually calssifying
        )

    def _init_weights(self):
        '''blackbox way to initialize stuff well'''
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


if __name__ == '__main__':
    LunaTrainingApp().main()
