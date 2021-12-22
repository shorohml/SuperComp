import enum
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread


STEPS = (5, 10, 15)
GRID_RES = (257, 257, 257)
IMGS_DIR = Path('../layers/imgs')


def main():
    font = {'family' : 'DejaVu Sans',
            'size'   : 25}
    matplotlib.rc('font', **font)

    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for i, step in enumerate(STEPS):
        img_1 = imread(IMGS_DIR / f'analytical_{step}_{GRID_RES[0]}_{GRID_RES[1]}_{GRID_RES[2]}.png')
        img_2 = imread(IMGS_DIR / f'layer_{step}_{GRID_RES[0]}_{GRID_RES[1]}_{GRID_RES[2]}.png')
        ax[i, 0].imshow(img_1)
        ax[i, 1].imshow(img_2)
        ax[i, 0].set_title(r'Analytical at $ t_{' + str(step) + r'} $')
        ax[i, 1].set_title(r'Computed at $ t_{' + str(step) + r'} $')
        for j in range(2):
            ax[i, j].tick_params(left = False, right = False , labelleft = False ,
                                    labelbottom = False, bottom = False)
            ax[i, j].axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.03, wspace=0, hspace=0)
    plt.savefig(IMGS_DIR / 'cmp.png', dpi=200)

    _, ax = plt.subplots(3, 1, figsize=(10, 30))
    for i, step in enumerate(STEPS):
        img_3 = imread(IMGS_DIR / f'errs_{step}_{GRID_RES[0]}_{GRID_RES[1]}_{GRID_RES[2]}.png')
        ax[i].imshow(img_3)
        ax[i].set_title(r'Absolute error at $ t_{' + str(step) + r'} $')
        ax[i].tick_params(left = False, right = False , labelleft = False ,
                                labelbottom = False, bottom = False)
        ax[i].axis('off')
    plt.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.03, wspace=0, hspace=0)
    plt.savefig(IMGS_DIR / 'errs.png', dpi=200)


if __name__ == '__main__':
    main()
