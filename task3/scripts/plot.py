from pathlib import Path
import configparser
import argparse
from typing import Tuple

import numpy as np
from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction
from pyface.api import GUI
from skimage.io import imsave


OPACITY = 0.02
FIG_SIZE = (1024, 1024)


def render_volume(volume_path: Path, img_path: Path, grid_size: Tuple[int, int, int]):
    data = np.fromfile(volume_path, dtype=np.float64)
    data = data.reshape(grid_size)

    fig = mlab.figure(size=FIG_SIZE)

    vol = mlab.pipeline.volume(
        mlab.pipeline.scalar_field(data),
        vmin=data.min(),
        vmax=data.max(),
    )
    vol._volume_property.shade = 0

    otf = PiecewiseFunction()
    otf.add_point(-1.0, OPACITY)
    otf.add_point(1.0, OPACITY)
    vol._otf = otf
    vol._volume_property.set_scalar_opacity(otf)

    cam = fig.scene.camera
    cam.zoom(0.7)

    ax = mlab.axes(
        ranges=(0, grid_size[0] - 1, 0, grid_size[1] - 1, 0, grid_size[2] - 1))
    ax.axes.label_format = '%.0f'
    mlab.outline()
    mlab.colorbar(orientation='vertical')

    # mlab.test_plot3d()
    GUI().process_events()

    # fig = mlab.gcf()
    fig.scene._lift()

    arr = mlab.screenshot(figure=fig)
    imsave(img_path, arr)
    mlab.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path2config', type=Path, help='path to config.ini')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.path2config)

    grid_size = (
        int(config['Solver']['N_x']) + 1,
        int(config['Solver']['N_y']) + 1,
        int(config['Solver']['N_z']) + 1,
    )
    save_step = int(config['Solver']['save_step'])
    n_steps = int(config['Solver']['K']) // save_step

    layers_path = Path(config['Solver']['layers_path'])
    layers_path = args.path2config.parent / layers_path
    imgs_path = layers_path / 'imgs'
    imgs_path.mkdir(exist_ok=True)

    for i in range(1, n_steps + 1):
        i *= save_step
        grid_size_str = f'{grid_size[0]}_{grid_size[1]}_{grid_size[2]}'
        render_volume(
            layers_path / f'layer_{i}_{grid_size_str}.bin',
            imgs_path / f'layer_{i}_{grid_size_str}.png',
            grid_size,
        )
        render_volume(
            layers_path / f'errs_{i}_{grid_size_str}.bin',
            imgs_path / f'errs_{i}_{grid_size_str}.png',
            grid_size,
        )
        render_volume(
            layers_path / f'analytical_{i}_{grid_size_str}.bin',
            imgs_path / f'analytical_{i}_{grid_size_str}.png',
            grid_size,
        )


if __name__ == '__main__':
    main()
