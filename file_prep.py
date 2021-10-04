import glob
from PIL import Image
import pandas as pd
import cv2
import torch
import os
import json
from dsets import getCandidateInfoList, getCandidateInfoDict, Img, getImgChunkCandidate, WhiteDotsClsDataset
from matplotlib import pyplot as plt
import numpy as np

raw_data_path = r'data/raw/*/*.jpg'

file_list = glob.glob(raw_data_path)

'''
### prepare the candidates_list
anno_info_path = r'data/raw/*/*.xlsx'
anno_info_list = glob.glob(anno_info_path)
anno_info_list.sort()
exp_1 = pd.read_excel(anno_info_list[0], engine='openpyxl')
candidates_list = []


for i in range(len(exp_1)):
    candidate_dict = {}
    candidate = exp_1.iloc[i]
    label_dict = json.loads(candidate.region_attributes)
    if label_dict['type'] == 'pits' or label_dict['type'] == 'pit':
        print(label_dict['type'])
        candidate_dict['file_list'] = [candidate.filename]
        candidate_dict['flags'] = 0
        candidate_dict['temporal_coordinates'] = []
        xyr_dict = json.loads(candidate.region_shape_attributes)
        candidate_dict['spatial_coordinates'] = [3., xyr_dict['cx'], xyr_dict['cy'], xyr_dict['r']]
        candidate_dict['metadata'] = {}
        candidates_list.append(candidate_dict)

df = pd.DataFrame(candidates_list)
df.to_excel('./data/raw/raw_data_anno_1/anno_1_edited.xlsx')
'''

### prepare the mask file

'''
candidateInfo_list = getCandidateInfoList()
candidateInfo_dict = getCandidateInfoDict()

blank_list = []

for img_id in list(candidateInfo_dict.keys()):

    image = Img(img_id)
    mask = image.positive_mask
    mask.dtype = np.uint8
    temp_png = Image.fromarray(mask*255)
    temp_png.save(glob.glob('./data/raw/*/{}'.format(img_id))[0][:-4]+'_mask.png')
'''

### prepare the neg_list
'''
image_list = sorted(getCandidateInfoDict().keys())
neg_list = []
num = 0
for img_id in image_list:
    num +=1
    print(num)
    img = Img(img_id)
    blank_list = img.gen_blank_candidate()
    # print(blank_list)
    for i in range(len(blank_list)):
        neg_list.append(blank_list[i])

df_neg = pd.DataFrame(neg_list)
df_neg.to_csv('./data/neg_candidates_info.csv')
'''




## export samples for the cls_model training

cls_dset = WhiteDotsClsDataset()

for i in range(len(cls_dset)):
    chunk = cls_dset.__getitem__(i)
    img_png = Image.fromarray((chunk['chunk'].numpy()[0]*255).astype(np.uint8))

    if chunk['pos'][1]:
        img_png.save('./data/cls_data/pos/{}_{}_{}.png'.format(i, chunk['id'][:-4], chunk['loc']))
    else:
        img_png.save('./data/cls_data/neg/{}_{}_{}.png'.format(i, chunk['id'][:-4], chunk['loc']))



