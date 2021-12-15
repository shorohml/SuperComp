from pathlib import Path

import numpy as np
from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction
from pyface.api import GUI
from skimage.io import imsave


OPACITY = 0.02
FIG_SIZE = (1024, 1024)
GRID_SIZE = (129, 129, 129)


def render_volume(volume_path: Path, img_path: Path):
    data = np.fromfile(volume_path, dtype=np.float64)
    data = data.reshape(GRID_SIZE)

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

    ax = mlab.axes(ranges=(0, GRID_SIZE[0] - 1, 0, GRID_SIZE[1] - 1, 0, GRID_SIZE[2] - 1))
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
    for i in range(1, 5):
        i *= 100
        render_volume(Path(f'layer{i}.bin'), Path(f'layer{i}.png'))
        render_volume(Path(f'errs{i}.bin'), Path(f'errs{i}.png'))


if __name__ == '__main__':
    main()
