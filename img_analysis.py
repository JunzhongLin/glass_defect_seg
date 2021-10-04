import numpy as np
from itertools import compress
import pandas as pd
import datetime
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim

from scipy.ndimage import morphology, measurements

import argparse
import glob
import os
import sys
from logconf import logging
from model import UNetWrapper
import model

from dsets import WhiteDotsClsDataset, WhiteDotsSegDataset, getCandidateInfoList, Img, CandidateInfoTuple, \
    getCandidateInfoDict
from utils import enumerateWithEstimate

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger('dsets').setLevel(logging.WARNING)


def print_confusion(label, confusions):
    col_labels = [
        'Miss detected (FN)',
        'Detected by Seg',
        'Dec+Seg Net (TP)', # Correctly detected by Dec+Seg Net (TP)
        'Dec+Seg Net (FP)', # incorrectly detected by Dec+Seg Net (FP)
    ]
    cell_width = 20
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    print(' | '.join([f.format(i) for i in confusions[0]]))


def match_and_score(detections, seg_detections, truth, threshold=0.5):
    # return 1X4 confusion matrix
    # Rows: white dots
    # Cols (counts): Miss detected (FN), Detected by Seg , Correctly detected by Dec+Seg Net (TP),
    #                incorrectly detected by Dec+Seg Net (FP)
    # If one detection matches several white-dot annotations, it counts for all of them

    true_wd = [c for c in truth if c.isPit]  # wd denotes white dots
    truth_diams = np.array([c.radius_pixel for c in true_wd])
    truth_xyz = np.array([c.center_xyz[1:3] for c in true_wd])

    detected_xyz = np.array([c.center_xyz[1:3] for c in detections])

    confusion = np.zeros((1, 4), dtype=int)

    confusion[0, 1] = len(seg_detections)

    if len(detected_xyz) == 0:
        confusion[0, 0] = len(truth_xyz)

    elif len(truth_xyz) == 0:
        confusion[0, 3] = len(detected_xyz)

    else:
        normalized_dists = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / \
                           truth_diams[:, None]
        matches = (normalized_dists < 0.7)
        unmatched_detections = np.ones(len(detections), dtype=bool)
        matched_true_wd = np.zeros(len(true_wd), dtype=int)
        for i_twd, i_detection in zip(*matches.nonzero()):
            matched_true_wd[i_twd] = 1
            unmatched_detections[i_detection] = False

        confusion[0, 0] = len(true_wd) - len(matched_true_wd.nonzero()[0])
        confusion[0, 2] = len(matched_true_wd.nonzero()[0])
        confusion[0, 3] = len(~unmatched_detections.nonzero()[0])

    return confusion


class ImgAnalysisAPP:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--worker_num", type=int, default=2, help="number of workers of dataloader"
        )
        parser.add_argument(
            "--batch_size", type=int, default=2, help="batch size of input"
        )
        parser.add_argument(
            '--run_validation',
            help='Run over validation rather than a single image',
            action='store_true',
            default=False,
        )

        parser.add_argument(
            '--analysis_stage',
            help='Run over fresh images rather than the ones from golden samples',
            action='store_true',
            default=False
        )

        parser.add_argument(
            '--include_train',
            help='Include data that was in the training set. (default: validation data only)',
            action='store_true',
            default=False
        )

        parser.add_argument(
            '--segmentation_path',
            help='path to the saved segmentation model',
            nargs='?',
            default=None
        )

        parser.add_argument(
            '--classification_path',
            help="Path to the saved classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument(
            '--cls_model',
            help='What to model class name to use for the classifier.',
            action='store',
            default='DecisionNet'
        )

        parser.add_argument(
            '--tb_prefix',
            default='img_analysis',
            help="Data prefix to use for Tensorboard run."
        )

        parser.add_argument(
            '--image_id',
            default='SNAP-133102-0007.jpg,SNAP-133228-0014.jpg',
            help='image id to use'
        )

        parser.add_argument(
            '--output_prefix',
            default='testing images',
            help='prefix to help define the output summary of white dots found on the glass'
        )

        parser.add_argument(
            '--output_seg_mask',
            default=True,
            action='store_true',
            help='Export the final segmentation masks'
        )

        self.opt = parser.parse_args(sys_argv)

        if not (bool(self.opt.image_id) ^ self.opt.run_validation):
            raise Exception("One and only one of image_id and --run_validation should be given")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.opt.segmentation_path:
            self.opt.segmentation_path = self.init_model_path('seg')

        if not self.opt.classification_path:
            self.opt.classification_path = self.init_model_path('cls')

        self.seg_model, self.cls_model = self.init_models()

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    def init_model_path(self, type_str):
        local_path = os.path.join(
            './saved_models',
            'white_dots_{}'.format(type_str),
            '*.best.state'
        )

        file_list = glob.glob(local_path)

        if not file_list:
            pretrained_path = os.path.join(
                './saved_models',
                'white_dots_{}'.format(type_str),
                '*.*.state'
            )
            file_list = glob.glob(pretrained_path)

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise

    def init_models(self):
        log.debug(self.opt.segmentation_path)
        seg_dict = torch.load(self.opt.segmentation_path, map_location=self.device)
        seg_model = UNetWrapper(
            in_channels=1,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        log.debug(self.opt.classification_path)
        cls_dict = torch.load(self.opt.classification_path, map_location=self.device)

        model_cls = getattr(model, self.opt.cls_model)
        cls_model = model_cls()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        return seg_model, cls_model

    def init_seg_dl(self, image_id):

        ## need some modification for the fresh data
        seg_ds = WhiteDotsSegDataset(
            val_stride=0,
            image_id=image_id
        )

        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.opt.batch_size*(torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.opt.worker_num,
            pin_memory=self.use_cuda
        )

        return seg_dl

    def init_cls_dl(self, candidateInfo_list):

        cls_ds = WhiteDotsClsDataset(
            val_stride=0,
            analysis_stage=True,
            candidateInfo_list=candidateInfo_list
        )

        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.opt.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.opt.worker_num,
            pin_memory=self.use_cuda
        )

        return cls_dl

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.opt))
        val_ds = WhiteDotsClsDataset(
            val_stride=5,
            isValSet_bool=True
        )

        val_set = set(
            candidateInfo_tup.image_id
            for candidateInfo_tup in val_ds.candidateInfo_list
        )

        if self.opt.image_id:
            images_set = set(self.opt.image_id.split(','))
        else:
            images_set = set(
                candidateInfo_tup.image_id
                for candidateInfo_tup in getCandidateInfoList()
            )

        if self.opt.include_train:
            train_list = sorted(images_set-val_set)
        else:
            train_list = []
        val_list = sorted(images_set & val_set)
        # print(val_list, train_list)
        seg_dl = self.init_seg_dl(train_list + val_list)
        seg_res, image_id_list, image_list = self.segment_img(seg_dl, threshold=0.5)
        candidateInfo_list, label_a_list, candidate_count_list = \
            self.group_seg_output(image_id_list, image_list, seg_res)
        assert np.array(candidate_count_list).sum()==len(candidateInfo_list), repr(candidate_count_list)+\
            ' num of candidates: '+repr(len(candidateInfo_list))
        cls_dl = self.init_cls_dl(candidateInfo_list)
        cls_res = self.classify_candidates(cls_dl)

        candidateInfo_list_final = self.write_summary_csv(candidateInfo_list, cls_res)

        if self.opt.output_seg_mask:
            self.log_images(cls_res, label_a_list, candidate_count_list, image_id_list)

        candidateInfo_dict = getCandidateInfoDict()

        all_confusion = np.zeros((1, 4), dtype=int)

        for image_id in image_id_list:
            detection_list = [c for c in candidateInfo_list_final if c.image_id == image_id]
            truth = candidateInfo_dict[image_id]
            one_confusion = match_and_score(detection_list, candidateInfo_list, truth)
            all_confusion += one_confusion

        print_confusion('Total counts: ', all_confusion)


        ### export the final segmentation mask

    def classify_candidates(self, cls_dl):
        # log.info('staring candidate classification of total {} candidates'.format(len(cls_dl.dataset)))

        cls_batch_iter = enumerateWithEstimate(
            cls_dl,
            'Image_analysis at {} '.format('classification stage')
        )
        cls_res = []
        with torch.no_grad():
            for batch_ndx, batch_dict in cls_batch_iter:
                input_t = batch_dict['chunk']
                input_g = input_t.to(self.device)
                _, probability_g = self.cls_model(input_g)
                for prob in probability_g.cpu().numpy()[:, 1] > 0.5:
                    cls_res.append(prob)

        return cls_res

    def segment_img(self, img_dl, threshold=0.5):
        # log.info('starting image segmentation of total {} images'.format(len(img_dl.dataset)))

        seg_batch_iter = enumerateWithEstimate(
            img_dl,
            'Image_analysis at {} '.format('Segmentation stage')
        )
        seg_res = []
        image_list = []
        image_id_list = []
        with torch.no_grad():
            for batch_ndx, batch_dict in seg_batch_iter:
                input_g = batch_dict['img'].to(self.device)
                prediction_g = self.seg_model(input_g)
                for pred_item, img_name, img in zip(
                        prediction_g.cpu().numpy(),
                        batch_dict['id'],
                        batch_dict['img']
                ):
                    mask_a = pred_item > threshold
                    mask_a[0] = morphology.binary_erosion(mask_a[0], iterations=1)
                    seg_res.append(mask_a)
                    image_id_list.append(img_name)
                    image_list.append(img.cpu().numpy())
        return seg_res, image_id_list, image_list

    def group_seg_output(self, image_id_list, image_list, seg_res):
        candidateInfo_list = []
        label_a_list = []
        candidate_count_list= []
        for image_id, image, mask_a in zip(image_id_list, image_list, seg_res):
            candidateLabel_a, candidate_count = measurements.label(mask_a)
            # img = Img(image_id=image_id)
            center_xyz_list = measurements.center_of_mass(
                image,
                labels=candidateLabel_a,
                index=np.arange(1, candidate_count+1)
            )

            label_a_list.append(candidateLabel_a)
            candidate_count_list.append(candidate_count)

            for i, center_xyz in enumerate(center_xyz_list):
                diameter = np.sqrt(len(np.where(candidateLabel_a==i+1)[0])/np.pi)*2
                # print(center_xyz)

                # beware that the sequence of x, y is inverted with the annotation csv file
                candidateInfo_tup = CandidateInfoTuple(
                    True, diameter, (3.0, center_xyz[2], center_xyz[1], diameter), image_id
                )

                candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list, label_a_list, candidate_count_list

    def log_res(self):

        pass

    def write_summary_csv(self, candidateInfo_list, cls_res):

        # export summary of white_dots found

        # filter out the miss assigned white dots by segmentation model
        # print(candidateInfo_list, cls_res)
        candidateInfo_list_final = list(compress(candidateInfo_list, cls_res))
        summary_df = pd.DataFrame.from_records(
            candidateInfo_list_final,
            columns=CandidateInfoTuple._fields
        )
        output_path = os.path.join(
            '.\output',
            self.opt.output_prefix,
            self.time_str+'\\',
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(os.path.join(output_path, 'summary.csv'))

        return candidateInfo_list_final

    def log_images(self, cls_res, seg_res, candidate_count_list, image_id_list):
        output_path = os.path.join(
            '.\output',
            'segmentation',
            self.time_str+'\\'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        i=0
        for image_id, seg_a, candidate_count in zip(image_id_list, seg_res, candidate_count_list):
            print(seg_a)
            for j in range(1, candidate_count+1):
                if not cls_res[i+j-1]:
                    seg_a[np.where(seg_a == j)] = 0
            seg_a_final = (seg_a > 0).astype(np.int64)
            print(seg_a_final.nonzero())
            i += candidate_count
            seg_im = Image.fromarray((seg_a_final[0]*255).astype(np.uint8))
            seg_im.save(os.path.join(output_path, image_id+'_seg.png'))


if __name__ == '__main__':
    img_analysis_proc = ImgAnalysisAPP()
    img_analysis_proc.main()