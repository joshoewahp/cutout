import os
import click
import logging
import matplotlib
import time
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from astropy.time import Time

from astroutils.logger import setupLogger
from astroutils.io import FITSException
from cutout import Cutout, ContourCutout


logger = logging.getLogger(__name__)

radius = 1/30
cmap = plt.cm.gray_r

params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
    'axes.titlesize': 10,
    'font.size': 10,  # was 10
    'legend.fontsize': 6,  # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    # 'figure.figsize': [6.825, 23.625],
    'font.family': 'serif',
}

matplotlib.rcParams.update(params)
offsets = True

def colorbar(cutout, ax, sign, pol, vmin, vmax, fig):
    divider = make_axes_locatable(ax)
    width = ax.get_position().width
    height = ax.get_position().height

    cax = divider.append_axes("right", size="3%", pad=0, axes_class=matplotlib.axes._axes.Axes)

    cbar = fig.colorbar(cutout.im, cax=cax, ticks=range(vmin, vmax+1))
    cbar.ax.set_yticklabels([f'{v}' for v in range(vmin, vmax+1)])
    cbar.ax.set_ylabel(r"mJy PSF$^{-1}$", labelpad=0)
   
    return


@click.command()
@click.option('-n', '--name', type=str, default=None,
              help='Name of object')
@click.option('-s', '--stokes', type=click.Choice(['I', 'V']))
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Enable verbose logging.')
@click.option('-S', '--savedir', type=click.Path(), default=None)
@click.argument('ra')
@click.argument('dec')
def main(name, stokes, verbose, savedir, ra, dec):

    setupLogger(verbose)

    ncols = 4
    stretchfactor = 5
    N = ncols * stretchfactor

    unit = u.hourangle if ':' in ra or 'h' in ra else u.deg
    position = SkyCoord(ra=ra, dec=dec, unit=(unit, u.deg))

    fig = plt.figure(figsize=(7.5, 7))
    gs = GridSpec(N, N+1, figure=fig)
    surveys = [s + stokes for s in [
        'racs',
        'vastp1',
        'vastp2',
        'vastp3x',
        'vastp4x',
        'vastp5x',
        'vastp6x',
        'vastp7x',
        'vastp8',
        'vastp9',
        'vastp10x',
        'vastp11x',
        'vastp12',
    ]]
    
    vmax = 2

    for grididx, survey in enumerate(surveys):

        rowidx = (grididx // ncols) * stretchfactor
        colidx = (grididx % ncols) * stretchfactor
        
        gridslice = gs[rowidx:rowidx+stretchfactor,
                       colidx:colidx+stretchfactor]

        try:
            sign = -1 if survey == 'racsV' else 1
            cutout = Cutout(survey, position, radius=radius, bar=False, title=False, grid=False,
                            coords='compact', corner=True, psf=True, stokes=stokes.lower(),
                            sign=sign, obfuscate=True, vmin=-vmax, vmax=vmax)

            ax = fig.add_subplot(gridslice, projection=cutout.wcs)
            cutout.plot(fig, ax)

            # Add survey name annotation
            text = AnchoredText(survey.replace(stokes, '').upper().replace('X', 'x').replace('vastp', 'VASTP '),
                                loc='lower right',
                                frameon=False,
                                prop={'size': 11, 'color': 'firebrick',
                                      'family': 'sans-serif', 'alpha': 0.7,
                                      'weight': 'heavy', 'usetex': False})
            ax.add_artist(text)

        except (FITSException, AssertionError) as e:
            logger.warning(f"{survey} failed: {e}")
            pass

    bigax = fig.add_subplot(gs[:])
    bigax.axis('off')

    colorbar(cutout, bigax, 1, stokes, -vmax, vmax, fig)

    if name:
        bigax.set_title(name, size=20)
    
    fig.subplots_adjust(left=0.075, right=0.925, top=0.90, bottom=0.05,
                        hspace=0.25, wspace=0.1)
        
    plt.show()

if __name__ == '__main__':
    main()
