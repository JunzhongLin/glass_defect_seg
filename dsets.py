import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from disk import getCache
from logconf import logging

from PIL import Image, ImageOps

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('white_dots_raw')
CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isPit, radius_pixel, center_xyz, image_id')


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):

    image_list = glob.glob('./data/raw/*/*.jpg')
    presentOnDisk_set = {os.path.split(p)[-1] for p in image_list}

    candidateInfo_list = []
    with open('./data/candidates_info.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            image_id = row[0][2:-2]
            if image_id not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isPit_bool = bool(int(row[1]))
            candidateCenter_xyz = tuple([float(x) for x in row[3][1:-1].split(',')])

            if candidateCenter_xyz[1] >= 1024 or candidateCenter_xyz[2] >= 1024:   # image is cropped to suit with Unet
                continue

            annotationRadius_pixel = float(row[3][1:-1].split(',')[-1])
            candidateInfo_list.append(
                CandidateInfoTuple(
                    isPit_bool,
                    annotationRadius_pixel,
                    candidateCenter_xyz,
                    image_id
                )
            )
    candidateInfo_list.sort(reverse=True)

    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.image_id,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict


class Img:

    def __init__(self, image_id, analysis_stage=False, chunk_size=64):
        image_path = glob.glob('./data/raw/*/{}'.format(image_id))[0]
        img_rgb = Image.open(image_path).convert('RGB')
        img = ImageOps.grayscale(img_rgb)


        self.img_a = np.array(img)
        self.image_id = image_id

        candidateInfo_list = copy.copy(getCandidateInfoDict())[self.image_id]

        self.positiveInfo_list = [
            candidate_tup for candidate_tup in candidateInfo_list if candidate_tup.isPit
        ]
        self.positive_mask = self.buildAnnotationMask()
        self.chunk_size = chunk_size

    def buildAnnotationMask(self,):

        boundingBox_a = np.zeros_like(self.img_a, dtype=bool)

        for candidateInfo_tup in self.positiveInfo_list:
            center_x = candidateInfo_tup.center_xyz[2]
            center_y = candidateInfo_tup.center_xyz[1]
            X, Y = np.ogrid[:self.img_a.shape[0], :self.img_a.shape[1]]   # confusion about axes
            distance_to_center = np.sqrt((X-center_x)**2 + (Y-center_y)**2)
            mask = distance_to_center < candidateInfo_tup.radius_pixel
            boundingBox_a = boundingBox_a | mask

        return boundingBox_a

    def gen_blank_candidate(self,):
        blank_candidate_list = []

        exclude_area_mask = np.zeros_like(self.img_a, dtype=np.bool)

        mask_left = exclude_area_mask.copy()
        mask_top = exclude_area_mask.copy()
        mask_left[:400, :] = True
        mask_top[:, :400] = True
        exclude_area_mask = mask_top & mask_left
        exclude_area_mask[:self.chunk_size, :] = True
        exclude_area_mask[-self.chunk_size:, :] = True
        exclude_area_mask[:, :self.chunk_size] = True
        exclude_area_mask[:, -self.chunk_size:] = True

        for candidate_tup in self.positiveInfo_list:
            c_x = candidate_tup.center_xyz[2]
            c_y = candidate_tup.center_xyz[1]
            x_range = (
                int(c_x-self.chunk_size) if (c_x-self.chunk_size)>0 else 0,
                int(c_x-self.chunk_size) if (c_x+self.chunk_size)<self.img_a.shape[1] else self.img_a.shape[1]
            )
            y_range = (
                int(c_y-self.chunk_size) if (c_y-self.chunk_size)>0 else 0,
                int(c_y+self.chunk_size) if (c_y+self.chunk_size)<self.img_a.shape[0] else self.img_a.shape[0]
            )
            exclude_area_mask[x_range[0]:x_range[1], y_range[0]:y_range[1]] = True

        blank_pool = np.where(exclude_area_mask == 0)
        indices = np.random.randint(0, len(blank_pool[0]), size=len(self.positiveInfo_list))
        for i in indices:
            blank_candidate_list.append(
                {
                    'file_list': [self.image_id], 'flags': 0, 'temporal_coordinates': [],
                    'spatial_coordinates': [3., blank_pool[1][i], blank_pool[0][i], 0.],   # notice the axes
                    'metadata': {}
                }
            )
        # print(blank_candidate_list)
        return blank_candidate_list

    def getChunkCandidate(self, center_xyz):
        c_x = center_xyz[2]
        c_y = center_xyz[1]

        slice_list = []
        for axis, central_val in enumerate([c_x, c_y]):
            start_pixel = int(round(central_val - self.chunk_size/2))
            end_pixel   = int(round(central_val + self.chunk_size/2))

            assert central_val >= 0 and central_val <= self.img_a.shape[axis], repr([self.image_id, center_xyz])

            if start_pixel < 0:
                start_pixel = 0
                end_pixel=int(self.chunk_size)

            if end_pixel > self.img_a.shape[axis]:
                end_pixel = self.img_a.shape[axis]
                start_pixel = int(self.img_a.shape[axis] - self.chunk_size)

            slice_list.append(slice(start_pixel, end_pixel))

        img_chunk_a = self.img_a[tuple(slice_list)]
        mask_chunk_a = self.positive_mask[tuple(slice_list)]

        return img_chunk_a, mask_chunk_a


@functools.lru_cache(1, typed=True)
def getImg(image_id, chunk_size=64):
    return Img(image_id, chunk_size=chunk_size)


@raw_cache.memoize(typed=True)
def getImgChunkCandidate(image_id, center_xyz, chunk_size=64):
    # image = Img(image_id)
    image = getImg(image_id, chunk_size=chunk_size)
    img_chunk_a, mask_chunk_a = image.getChunkCandidate(center_xyz)
    return img_chunk_a, mask_chunk_a


@raw_cache.memoize(typed=True)
def getImgSegCandidate(image_id):
    image = Img(image_id)
    img_a = image.img_a
    mask = image.buildAnnotationMask()
    return img_a, mask, image.image_id


class WhiteDotsSegDataset(Dataset):

    def __init__(self, val_stride=0, isValSet_bool=None, image_id=None,
                 transforms_=None, transforms_mask=None, ):

        if image_id:
            self.image_list = image_id
        else:
            self.image_list = sorted(copy.copy(getCandidateInfoDict()).keys())

        #  deal with the split of training set and the test set
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.image_list = self.image_list[::val_stride]
            assert self.image_list
        elif val_stride > 0:
            del self.image_list[::val_stride]
            assert self.image_list

#         images_set = set(self.image_list)

#         self.candidateInfo_list = getCandidateInfoList()
#
#         self.candidateInfo_list = [cit for cit in self.candidateInfo_list if cit.image_id
#                                    in images_set]
#         self.pos_list = [nt for nt in self.candidateInfo_list if nt.isPit]

        if transforms_ is None:
            self.transforms_ = transforms.Compose(
                [transforms.ToTensor()]
            )
        else:
            self.transforms_ = transforms_

        if transforms_mask is None:
            self.transforms_mask = self.transforms_

        else:
            self.transforms_mask = transforms_mask

        log.info("{!r}: {} {} images".format(
            self,
            len(self.image_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
        ))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, ndx):

        image_id = self.image_list[ndx % len(self.image_list)]
        img_a, mask, _ = getImgSegCandidate(image_id)
        img_t = self.transforms_(img_a)
        mask_t = self.transforms_mask(mask)
        return {'img': img_t[:, :1024, :1024], 'mask': mask_t[:, :1024, :1024], 'id': image_id}


class WhiteDotsClsDataset(Dataset):

    def __init__(self, val_stride=5, isValSet_bool=None, image_id=None, candidateInfo_list=None,
                 transforms_=None, transforms_mask=None, analysis_stage=False, ratio_int=0):

        self.ratio_int = ratio_int

        if image_id:
            self.image_list = image_id
        else:
            self.image_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.image_list = self.image_list[::val_stride]
        elif val_stride > 0:
            del self.image_list[::val_stride]

        images_set = set(self.image_list)
        if candidateInfo_list:   # in the image_analysis app, the the use_cache will be turned off
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False

        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        self.candidateInfo_list = [cit for cit in self.candidateInfo_list if cit.image_id
                                    in images_set]

        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isPit]
        self.neg_list = [nt for nt in self.candidateInfo_list if not nt.isPit]

        if not analysis_stage:
            assert self.neg_list and self.pos_list, '\n length of neg_list: ' + repr(len(self.neg_list)) + \
                                                    '\n length of pos_list: ' + repr(len(self.pos_list))

        '''
        ### possibly data leakage
        
        images_set = set(self.image_list)

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False

        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        self.candidateInfo_list = [cit for cit in self.candidateInfo_list if cit.image_id
                                    in images_set]

        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isPit]
        self.neg_list = [nt for nt in self.candidateInfo_list if not nt.isPit]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.pos_list = self.pos_list[::val_stride]
            self.neg_list = self.neg_list[::val_stride]
            assert self.pos_list and self.neg_list
        elif val_stride > 0:
            del self.neg_list[::val_stride]
            del self.pos_list[::val_stride]
            assert self.neg_list and self.pos_list, '\n length of neg_list: '+ repr(len(self.neg_list)) + \
                                                    '\n length of pos_list: ' + repr(len(self.pos_list))
        '''

        if transforms_ is None:
            self.transforms_ = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transforms_ = transforms_

        if transforms_mask is None:
            self.transforms_mask = self.transforms_

        else:
            self.transforms_mask = transforms_mask

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.neg_list)+len(self.pos_list),
            "validation" if isValSet_bool else "training" if not analysis_stage else 'image_analysis',
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 2000
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            '''
            The relationship between the dataset index and the positive index is simple: divide
            the dataset index by 3 and then round down. The negative index is slightly more complicated,
            in that we have to subtract 1 from the dataset index and then subtract the
            most recent positive index as well.
            '''
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidateInfo_tup = self.neg_list[neg_ndx]

            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]

        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        if self.use_cache:
            candidate_a, mask_a = getImgChunkCandidate(candidateInfo_tup.image_id, candidateInfo_tup.center_xyz)

        else:
            img = getImg(candidateInfo_tup.image_id)
            candidate_a, mask_a = img.getChunkCandidate(candidateInfo_tup.center_xyz)

        candidate_t = self.transforms_(candidate_a)
        mask_t = self.transforms_mask(mask_a)

        pos_t = torch.tensor([
            not candidateInfo_tup.isPit,
            candidateInfo_tup.isPit
            ], dtype=torch.long
        )

        return {'chunk': candidate_t, 'pos': pos_t,
                'id': candidateInfo_tup.image_id,
                'mask': mask_t,
                'loc': candidateInfo_tup.center_xyz}


class WhiteDotsChunkSegDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, image_id=None,
                 transforms_=None, transforms_mask=None, chunk_size=256, use_cache=True):

        if image_id:
            self.image_list = image_id
        else:
            self.image_list = sorted(copy.copy(getCandidateInfoDict()).keys())

        self.use_cache = use_cache
        self.chunk_size = chunk_size

        #  deal with the split of training set and the test set

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.image_list = self.image_list[::val_stride]
            assert self.image_list
        elif val_stride > 0:
            del self.image_list[::val_stride]
            assert self.image_list
        images_set = set(self.image_list)
        self.candidateInfo_list = getCandidateInfoList()
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list if cit.image_id
                                   in images_set]
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isPit]

        if transforms_ is None:
            self.transforms_ = transforms.Compose(
                [transforms.ToTensor()]
            )
        else:
            self.transforms_ = transforms_

        if transforms_mask is None:
            self.transforms_mask = self.transforms_

        else:
            self.transforms_mask = transforms_mask

    def __len__(self):
        return len(self.pos_list)

    def __getitem__(self, ndx):

        candidateInfo_tup = self.pos_list[ndx]

        if self.use_cache:
            candidate_a, mask_a = getImgChunkCandidate(
                candidateInfo_tup.image_id,
                candidateInfo_tup.center_xyz,
                chunk_size=self.chunk_size
            )

        else:
            img = getImg(candidateInfo_tup.image_id, chunk_size=self.chunk_size)
            candidate_a, mask_a = img.getChunkCandidate(candidateInfo_tup.center_xyz)

        img_t = self.transforms_(candidate_a)
        mask_t = self.transforms_mask(mask_a)

        return {'img': img_t, 'mask': mask_t, 'id': candidateInfo_tup.image_id}


class AnalysisSegDataset(Dataset):

    def __init__(self, image_id=None, candidateInfo_list=None, ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

class PrepcacheWhiteDotsDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    seg_dataset = WhiteDotsSegDataset()
    cls_dataset = WhiteDotsClsDataset()
    seg_chunk_dataset = WhiteDotsChunkSegDataset()

    seg_loader = DataLoader(
        seg_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
                            )
    cls_loader = DataLoader(
        cls_dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0
    )
    seg_chunk_loader = DataLoader(
        seg_chunk_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    test_img = seg_loader.dataset.__getitem__(1)['img']
    test_img_chunk = seg_chunk_loader.dataset.__getitem__(1)['img']
