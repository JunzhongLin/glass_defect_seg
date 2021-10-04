import numpy as np

from model import DecisionNet,  weights_init_normal
from dsets import WhiteDotsClsDataset

import torch.nn as nn
import torch
from torch.optim import Adam
import hashlib
import shutil
from matplotlib import pyplot as plt

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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PRED_P_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4


class ClassificationTrainingApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")
        parser.add_argument("--batch_size", type=int, default=10, help="batch size of input")
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

        parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
        parser.add_argument("--end_epoch", type=int, default=101, help="end_epoch")

        parser.add_argument("--need_test", type=bool, default=True, help="need to test")
        parser.add_argument("--test_interval", type=int, default=10, help="interval of test")
        parser.add_argument("--need_save", type=bool, default=True, help="need to save")
        parser.add_argument("--save_interval", type=int, default=10, help="interval of save weights")

        parser.add_argument("--img_height", type=int, default=704, help="size of image height")
        parser.add_argument("--img_width", type=int, default=256, help="size of image width")

        parser.add_argument('--finetune', help="Start finetuning from this model.", default='',)
        parser.add_argument('--finetune-depth',
                            help="Number of blocks (counted from the head) to include in finetuning",
                            type=int, default=1,)
        parser.add_argument('--tb-prefix', default='white_dots_cls',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",)
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='_',)
        parser.add_argument('seg_model_path',
                            help='define the path for the state of seg_model',
                            nargs='?', default=' ',)

        self.opt = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.decision_model = self.init_cls_model()
        self.root_path = r'./Data'

        self.optimizer = self.init_optimizer()

    def init_cls_model(self):

        decision_model = DecisionNet()
        if self.use_cuda:
            log.info('using CUDA; {} devices'.format(torch.cuda.device_count()))
            decision_model = nn.DataParallel(decision_model)
        decision_model = decision_model.to(self.device)
        decision_model.apply(weights_init_normal)

        return decision_model

    def init_optimizer(self):
        return Adam(self.decision_model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

    def init_train_dl(self):

        train_dl = DataLoader(
            WhiteDotsClsDataset(
                val_stride=5,
                isValSet_bool=False,
                ratio_int=1
            ),
            batch_size=self.opt.batch_size,
            num_workers=self.opt.worker_num,
            shuffle=False,
            pin_memory=self.use_cuda
        )

        return train_dl

    def init_val_dl(self):

        val_dl = DataLoader(
            WhiteDotsClsDataset(
                val_stride=5,
                isValSet_bool=True,
                ratio_int=1
            ),
            batch_size=self.opt.batch_size,
            num_workers=self.opt.worker_num,
            shuffle=False,
            pin_memory=self.use_cuda
        )

        return val_dl

    def init_tensorboard_writer(self):

        if self.trn_writer is None:
            log_dir = os.path.join('run_cls', self.opt.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir+'_trn_cls'+self.opt.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir+'_val_cls'+self.opt.comment
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
            self.log_metrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                valMetrics_t = self.do_validation(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)
                self.save_model('seg', epoch_ndx, score == best_score)
                self.log_metrics(epoch_ndx, 'val', valMetrics_t)

        self.trn_writer.close()
        self.val_writer.close()

    def do_training(self, epoch_ndx, train_dl):

        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        self.decision_model.train()

        train_dl.dataset.shuffleSamples()
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_dict in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_dict,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def do_validation(self, epoch_ndx, val_dl):

        with torch.no_grad():
            self.decision_model.eval()

            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )

        for batch_ndx, batch_dict in batch_iter:
            self.compute_batch_loss(
                batch_ndx, batch_dict, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_dict, batch_size, metrics_g,):

        input_t = batch_dict['chunk']
        label_t = batch_dict['pos']

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.decision_model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],
        )

        with torch.no_grad():
            start_ndx = batch_ndx * batch_size
            end_ndx = start_ndx + input_t.size(0)

            metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].view(-1)
            metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = (probability_g > 0.5)[:, 1].view(-1)
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.view(-1)
            metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:, 1].view(-1)

        return loss_g.mean()

    def log_metrics(self, epoch_ndx, mode_str, metrics_t):

        log.info('E{} {}'.format(
            epoch_ndx,
            type(self).__name__
        ))

        self.init_tensorboard_writer()

        metrics_a = metrics_t.detach().numpy()

        negLabel_mask = metrics_a[METRICS_LABEL_NDX] == 0
        posLabel_mask = ~ negLabel_mask
        negPred_mask = metrics_a[METRICS_PRED_NDX] == 0
        posPred_mask = ~ negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float64(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float64(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0)
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:]-fpr[:-1]
        tp_avg  = (tpr[1:]+tpr[:-1])/2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score, "
                 + "{auc:.4f} auc"
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
                mode_str + '_' + 'neg',
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
                mode_str + '_' + 'pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        fig = plt.figure()
        plt.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)

        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        bins = np.linspace(0, 1)

        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        score = metrics_dict['auc']

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

        model = self.decision_model
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
    example = ClassificationTrainingApp()
    example.main()


