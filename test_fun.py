import glob
from PIL import Image, ImageOps
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from dsets import Img
from dsets import WhiteDotsClsDataset

# image_list = glob.glob('./data/raw/*/SNAP-163019-0295.jpg')
# img_rgb = Image.open(image_list[0]).convert('RGB')
# img = ImageOps.grayscale(img_rgb)
#
# trans_ = transforms.Compose([
#     transforms.ToTensor()
# ])
#
#
# a = np.array(img)
#
# b = np.random.randint(2, size=(4,4))
# IMG = Img('SNAP-163019-0295.jpg')
# # plt.imshow(IMG.positive_mask)

datset_cls = WhiteDotsClsDataset(
    val_stride=10,
    isValSet_bool=False,
    ratio_int=1
)
