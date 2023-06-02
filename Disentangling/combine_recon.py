#%%
import numpy as np
import torch
from PIL import Image
import glob
from torchvision.utils import make_grid, save_image
import re
import sys
import os

# %%
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# %%
def combineImages(name_initial, indices):
    padding = 2
    images = [np.rollaxis(np.asarray(Image.open(f))[padding:-padding, padding:-padding, :], 2, 0) / 255.0 \
        for f in sorted(glob.glob('{}_[0-9].jpg'.format(name_initial)) \
            + glob.glob('{}_[0-9][0-9].jpg'.format(name_initial)), key=natural_keys)]
    width = images[0].shape[1] + padding
    steps = len(images)
    allt = []
    for index in indices:
        for image in images:
            allt.append(image[:, :, (index * width):((index + 1) * width - padding)])
    save_image(tensor=torch.tensor(allt), fp='result/{}_combined.eps'.format(name_initial), nrow=steps, pad_value=1)
    

# %%
# combineImages('fixed_heart', [2, 4, 5, 6, 9, 8])
# label_lst = ['z1 (y)','z2','z3','z4','z5','z6 (Orientation)','z7 (Shape)',\
                # 'z8 (x)','z9','z10 (Scale)','total KL']

# %%
if __name__ == '__main__':
    directory = 'result'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # combineImages(sys.argv[1], [int(x) for x in sys.argv[2:]])
    # combineImages('fixed_ellipse', [0,1,2,3,4,5,6,7,8,9])
    combineImages('fixed_ellipse', [0,6])
    