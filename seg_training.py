import numpy as np

from model import UNetWrapper, SegmentationAugmentation
from dsets import WhiteDotsSegDataset, WhiteDotsClsDataset, WhiteDotsChunkSegDataset

import torch.nn as nn
import torch
from torch.optim import Adam
import hashlib
import shutil

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F

import os
import sys
import argparse
import time
import PIL.Image as Image
from logconf import logging
import datetime
from utils import enumerateWithEstimate

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
METRICS_FN_LOSS_NDX = 2
METRICS_ALL_LOSS_NDX = 3
METRICS_PTP_NDX = 4
METRICS_PFN_NDX = 5
METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10


class SegmentationTrainingApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument("--worker_num", type=int, default=2, help="number of workers of dataloader")
        parser.add_argument("--batch_size", type=int, default=8, help="batch size of input")
        parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

        parser.add_argument("--begin_epoch", type=int, default=1, help="begin_epoch")
        parser.add_argument("--end_epoch", type=int, default=101, help="end_epoch")

        # controller for the argument
        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument("--need_test", type=bool, default=True, help="need to test")
        parser.add_argument("--test_interval", type=int, default=10, help="interval of test")
        parser.add_argument("--need_save", type=bool, default=True, help="need to save")
        parser.add_argument("--save_interval", type=int, default=10, help="interval of save weights")

        parser.add_argument("--img_height", type=int, default=1024, help="size of image height")
        parser.add_argument("--img_width", type=int, default=1024, help="size of image width")
        parser.add_argument(
            "--chunk_size",
            type=int, default=256,
            help="size of chunk for seg training"
            )

        parser.add_argument('--tb_prefix',
                            default='white_dot_seg',
                            help='Data prefix to use for tensorboard run')
        parser.add_argument('--comment', help='comment suffix for Tensorboard run', nargs='?', default='_none')
        parser.add_argument('--loss_func', default='dice', const='dice', nargs='?',
                            choices=['BCEwl', 'MSE', 'dice'], help='method of loss function')

        self.opt = parser.parse_args(sys_argv)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')


        self.augmentation_dict = {}
        if self.opt.augmented or self.opt.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.opt.augmented or self.opt.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.opt.augmented or self.opt.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.opt.augmented or self.opt.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.opt.augmented or self.opt.augment_noise:
            self.augmentation_dict['noise'] = 25.0


        ## initialize the model
        self.segmentation_model, self.augmentation_model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        segmentation_model = UNetWrapper(
            in_channels=1,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def init_optimizer(self):
        return Adam(self.segmentation_model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

    def init_train_dl(self):
        train_ds = WhiteDotsChunkSegDataset(
            val_stride=5,
            isValSet_bool=False,
            chunk_size=self.opt.chunk_size
        )

        batch_size = self.opt.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.opt.worker_num,
            shuffle=False,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def init_val_dl(self):
        val_ds = WhiteDotsSegDataset(
            val_stride=5,
            isValSet_bool=True
        )

        batch_size = self.opt.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.opt.worker_num,
            shuffle=False,
            pin_memory=self.use_cuda
        )

        return val_dl

    def init_tensorboard_writer(self):

        if self.trn_writer is None:
            log_dir = os.path.join('run_seg', self.opt.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir+'_trn_seg'+self.opt.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir+'_val_seg'+self.opt.comment
            )

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.opt))

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        best_score = 0.0

        self.validation_cadence = 5

        for epoch_ndx in range(self.opt.begin_epoch, self.opt.end_epoch):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.opt.end_epoch,
                len(train_dl),
                len(val_dl),
                self.opt.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
                )
            )

            trnMetrics_t = self.do_training(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:

                valMetrics_t = self.do_validation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.save_model('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl, self.opt.chunk_size)
                print('successfully logged img from training')
                self.logImages(epoch_ndx, 'val', val_dl, self.opt.img_width)

        self.trn_writer.close()
        self.val_writer.close()

    def do_training(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        # train_dl.dataset.shuffleSamples()
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_dict in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(batch_ndx, batch_dict, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                'E{}  validation set'.format(epoch_ndx),
                start_ndx=val_dl.num_workers
            )

            for batch_ndx, batch_dict in batch_iter:
                self.compute_batch_loss(batch_ndx, batch_dict, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_dict, batch_size, metrics_g,
                           classificationThreshold=0.5):

        input_t = batch_dict['img']
        label_t = batch_dict['mask']
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        dice_loss_g = self.dice_loss(prediction_g, label_g)
        fn_loss_g = self.dice_loss(prediction_g * label_g, label_g)
        loss = dice_loss_g
        final_loss = dice_loss_g.mean() + fn_loss_g.mean() * 8

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32)

            tp = (predictionBool_g * label_g.to(torch.long)).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g.to(torch.long)).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (1 - label_g.to(torch.long))).sum(dim=[1, 2, 3])

            start_ndx = batch_ndx * batch_size
            end_ndx = start_ndx + input_t.size(0)

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return final_loss

    def dice_loss(self, prediction, label, epsilon=1):

        dice_pred = prediction.sum(dim=[1, 2, 3])
        dice_label = label.sum(dim=[1, 2, 3])
        dice_correct = (prediction*label).sum(dim=[1, 2, 3])
        dice_ratio = (2*dice_correct+epsilon)/(dice_pred+dice_label+epsilon)

        return 1-dice_ratio

    def logImages(self, epoch_ndx, mode_str, dl, image_size):

        self.segmentation_model.eval()

        for sample_ndx in range(6):
            sample_dict = dl.dataset.__getitem__(sample_ndx)
            transform_img = transforms.Compose([
                transforms.Resize((self.opt.img_height // 2, self.opt.img_width // 2)),
                transforms.ToTensor(),
                ]
            )

            input_t = sample_dict['img']
            label_t = sample_dict['mask']

            input_g = input_t.to(self.device).unsqueeze(0)
            label_g = label_t.to(self.device).unsqueeze(0)

            prediction_g = self.segmentation_model(input_g)
            prediction_a = prediction_g.to('cpu').detach().numpy()[0][0] > 0.5
            label_a = label_g.to('cpu').numpy()[0][0] > 0.5
            input_a = input_t[0].numpy()
            # input_img = Image.fromarray(input_a)   # transform need a PIL image
            # image_raw = transform_img(input_img)   # shape = (1, height, width)
            image_a = np.zeros((image_size, image_size, 3), dtype=np.float32)

            image_a[:, :, :] = input_a.reshape(image_size, image_size, 1)
            image_a[:, :, 0] += prediction_a & (1 - label_a)
            image_a[:, :, 0] += (1 - prediction_a) & label_a
            image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5

            image_a[:, :, 1] += prediction_a & label_a
            image_a *= 0.5
            image_a.clip(0, 1, image_a)

            writer = getattr(self, mode_str + '_writer')

            writer.add_image(
                f'{mode_str}_prediction_{sample_ndx}',
                image_a,
                self.totalTrainingSamples_count,
                dataformats='HWC',
            )

            if epoch_ndx == 1:
                image_a = np.zeros((image_size, image_size, 3), dtype=np.float32)
                image_a[:, :, :] = input_a.reshape(image_size, image_size, 1)
                image_a[:, :, 1] += label_a  # Green

                image_a *= 0.5
                image_a[image_a < 0] = 0
                image_a[image_a > 1] = 1
                writer.add_image(
                    '{}_label_{}'.format(
                        mode_str,
                        sample_ndx,
                    ),
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

            writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info('E{} {}'.format(
            epoch_ndx,
            type(self).__name__
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
                                                   / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] \
                                             / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
                                      / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.init_tensorboard_writer()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def save_model(self, type_str, epoch_ndx, isBest=False):

        file_path = os.path.join(
            'saved_models',
            self.opt.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.opt.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }

        torch.save(state, file_path)

        log.info('saved model params to {}'.format(file_path))

        if isBest:
            best_path = os.path.join(
                'saved_models',
                self.opt.tb_prefix,
                f'{type_str}_{self.time_str}_{self.opt.comment}.best.state')
            shutil.copyfile(file_path, best_path)

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__=='__main__':

    seg_train = SegmentationTrainingApp()
    seg_train.main()