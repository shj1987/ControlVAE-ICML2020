#%%
import numpy as np
import torch
from PIL import Image
import glob
from torchvision.utils import make_grid, save_image
import re
import sys,os

# %%
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# %%
def combineImages(name_initial, indices):
    padding = 2

    recon = np.rollaxis(np.asarray(Image.open('recon.jpg'))[padding:-padding, padding:-padding, :], 2, 0) / 255.0
    recon_allwidth = recon.shape[1]
    origin = recon[:, :, 0:recon_allwidth]
    recon = recon[:, :, (recon_allwidth + padding):]
    grid_size = 8
    recon_width = recon_allwidth / grid_size

    images = [np.rollaxis(np.asarray(Image.open(f))[padding:-padding, padding:-padding, :], 2, 0) / 255.0 \
        for f in sorted(glob.glob('{}_[0-9].jpg'.format(name_initial)) \
            + glob.glob('{}_[0-9][0-9].jpg'.format(name_initial)), key=natural_keys)]
    width = images[0].shape[1] + padding
    steps = len(images)
    allt = []
    for i in range(steps):
        hfrom = int((i // grid_size) * recon_width)
        wfrom = int((i % grid_size) * recon_width)
        allt.append(origin[:, hfrom:(hfrom + width - padding), wfrom:(wfrom + width - padding)])
    for i in range(steps):
        hfrom = int((i // grid_size) * recon_width)
        wfrom = int((i % grid_size) * recon_width)
        allt.append(recon[:, hfrom:(hfrom + width - padding), wfrom:(wfrom + width - padding)])
    for index in indices:
        for image in images:
            allt.append(image[:, :, (index * width):((index + 1) * width - padding)])
    print(allt[0].shape)
    save_image(tensor=torch.tensor(allt), filename='result/{}_combined.eps'.format(name_initial), nrow=steps, pad_value=1)
    

# %%
# combineImages('fixed_heart', [2, 4, 5, 6, 9, 8])

# %%
if __name__ == '__main__':
    directory = 'result'
    if not os.path.exists(directory):
        os.makedirs(directory)
    combineImages(sys.argv[1], [int(x) for x in sys.argv[2:]])
    
# %%
