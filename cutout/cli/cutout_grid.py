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

def colorbar(cutout, ax, sign, pol, vmin, vmax, fig, null=False):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1,
                              axes_class=matplotlib.axes._axes.Axes
                              )
    if null:
        cax.axis('off')
    else:
        if pol == 'I' or sign > 0:
            cbar = fig.colorbar(cutout.im, cax=cax, ticks=[vmin, vmax])
            cbar.ax.set_yticklabels([f'{vmin}', f'{vmax}'])
        else:
            cbar = fig.colorbar(cutout.im, cax=cax, ticks=[-vmax, -vmin])
            cbar.ax.set_yticklabels([f'{vmax*sign}', f'{vmin*sign}'])

        cbar.ax.set_ylabel(r"mJy PSF$^{-1}$", labelpad=-12)

    return


def cutout(survey, position, radius, vmax=None, vmin=None, contour=False, sign=1, pm=False, data=False):
    if contour:
        c = ContourCutout(survey, position, radius, cmap=cmap, contours=contour, title=False,
                          corner=True, grid=False, psf=True, coords='compact', bar=False,
                          contourwidth=1, neighbours=False, pm=pm, offset=True, 
                          data=data)
    else:
        c = Cutout(survey, position, radius * 2, cmap=cmap, vmin=vmin, vmax=vmax, title=False,
                   corner=True, grid=False, psf=True, bar=False, coords='compact',
                   verbose=False, neighbours=False, offset=True)
    return c

def opt_cutout(stardict, fig, r_pos):
    sign = stardict['sign']
    epoch = f"{stardict['epoch']}I"

    # cutout of optical data at radio position (just to get PM parameters)
    try:
        crad = cutout('panstarrs', r_pos, radius, contour=epoch, sign=sign, pm=True)
        survey = 'panstarrs'
        code = 'PS '
    except FITSException as e:
        crad = cutout('skymapper', r_pos, radius, contour=epoch, sign=sign, pm=True)
        survey = 'skymapper'
        code = 'SM '

    try:
        pmra = crad.pm_coord.pm_ra
    except AttributeError as e:
        pmra = crad.pm_coord.pm_ra_cosdec
        logger.error(e)
        logger.error("Astropy for some reason can't decide on calling this pm_ra or pm_ra_cosdec")

    # position of star at radio epoch (want optical data centred here)
    coord = SkyCoord(
        ra=crad.ra * u.deg,
        dec=crad.dec * u.deg,
        frame='icrs',
        distance=crad.pm_coord.distance,
        pm_ra_cosdec=pmra,
        pm_dec=crad.pm_coord.pm_dec,
        obstime=Time(crad.cmjd, format='mjd'))
    newpos = coord.apply_space_motion(Time(crad.mjd, format='mjd'))
    cO = cutout(survey, newpos, radius, contour=stardict['epoch']+'I', sign=sign)
    sep = newpos.separation(r_pos).arcsec
    logger.info(f"Originally at <{r_pos.ra:.3f}, {r_pos.dec:.3f}>, moved {sep:.2f} arsec to <{newpos.ra:.3f}, {newpos.dec:.3f}>")

    # radio centred image with optical data inserted
    crad = cutout(survey, r_pos, radius, contour=epoch, sign=sign, data=cO, pm=True)
    crad.correct_pm = False

    return crad, newpos, code


def add_cutout(stardict, row, fig, gs, keepxlabel=False, offsets=True):
    pos = SkyCoord(stardict['ra_deg_cont'], stardict['dec_deg_cont'], unit=u.deg)
    sign = stardict['sign']
    epoch = stardict['epoch']

    Vmax = int(np.ceil(stardict['V_flux_peak'])) 
    Vvmin, Vvmax = -Vmax, Vmax
    Ivmin, Ivmax = -1, int(np.ceil(stardict['I_flux_peak']))
    
    cI = cutout(f'{epoch}I', pos, radius, Ivmax, Ivmin)
    cV = cutout(f'{epoch}V', pos, radius, Vvmax * sign, Vvmin * sign)
    logger.info(pos)
    try:
        cO, newpos, code = opt_cutout(stardict, fig, pos)
    except FITSException as e:
        raise
        logger.info(f'\n{stardict}')
        logger.error(e)

    if offsets:
        cI.switch_to_offsets()
        cV.switch_to_offsets()
        cO.switch_to_offsets()

    cO.align_image_to_contours()
    cO.correct_pm = False

    axI = fig.add_subplot(gs[row, 0], projection=cI.wcs)
    axV = fig.add_subplot(gs[row, 1], projection=cV.wcs)
    axO = fig.add_subplot(gs[row, 2], projection=cO.wcs)

    cI.plot(fig, axI)
    cV.plot(fig, axV)
    cO.plot(fig, axO)

    if not offsets:
        cO.ax.coords[1].set_ticklabel_position('r')
        cO.ax.coords[1].set_auto_axislabel(False)

        dyear = Time(cI.mjd, format='mjd').decimalyear
        xlabel = f'RA (J{dyear:.1f})'
        ylabel = f'Dec (J{dyear:.1f})'

    else:
        for ax in [axI, axV, axO]:
            ax.coords[0].set_coord_type('longitude', coord_wrap=180)
            ax.coords[0].set_major_formatter('s')
            ax.coords[1].set_major_formatter('s')

        xlabel = 'RA Offset'
        ylabel = 'Dec Offset'

    if not keepxlabel:
        cI.ax.coords[0].set_auto_axislabel(False)
        cV.ax.coords[0].set_auto_axislabel(False)
        cO.ax.coords[0].set_auto_axislabel(False)
    else:
        cI.set_xlabel(xlabel)
        cV.set_xlabel(xlabel)
        cO.set_xlabel(xlabel)

    cI.set_ylabel(ylabel)
    cV.ax.coords[1].set_auto_axislabel(False)
    cO.ax.coords[1].set_auto_axislabel(False)

    textI = AnchoredText(' I ', loc='upper right', frameon=False,
                         prop={'size': 12, 'color': 'firebrick',
                               'family': 'sans-serif', 'alpha': 0.7,
                               'weight': 'heavy', 'usetex': False})
    textV = AnchoredText(' V ', loc='upper right', frameon=False,
                         prop={'size': 12, 'color': 'firebrick',
                               'family': 'sans-serif', 'alpha': 0.7,
                               'weight': 'heavy', 'usetex': False})
    textO = AnchoredText(code, loc='upper right', frameon=False,
                         prop={'size': 12, 'color': 'firebrick',
                               'family': 'sans-serif', 'alpha': 0.7,
                               'weight': 'bold', 'usetex': False})
    
    cI.ax.add_artist(textI)
    cV.ax.add_artist(textV)
    cO.ax.add_artist(textO)
    
    
    colorbar(cI, axI, sign, 'I', Ivmin, Ivmax, fig)
    colorbar(cV, axV, sign, 'V', Vvmin, Vvmax, fig)
    colorbar(cO, axO, sign, 'I', Ivmin, Ivmax, fig, null=True)

    axV.set_title(stardict["name"])

    return cI, cV, cO, axI, axV, axO


@click.command()
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Enable verbose logging.')
@click.argument('sources')
@click.argument('savedir', type=click.Path())
def main(verbose, sources, savedir):

    setupLogger(verbose)

    if os.path.exists(savedir):
        os.system(f"rm -r {savedir}")
    os.makedirs(savedir)

    df = pd.read_csv(sources)
    df['fig'] = df.apply(lambda x: x.name // 5 + 1, axis=1)

    t_marg = 0.04
    b_marg = 0.05
    l_marg = 0.08
    r_marg = 0.00
    wspace = 0.00
    hspace = 0.32
    
    for fignum, subdf in df.groupby('fig'):

        logger.info(f"Plotting Figure {fignum}: {subdf.name.values}")

        row_frac = len(subdf) / 5
        
        # 3 row figure requires a constant extra fig height padding offset
        x = .425
        height = (9-x)*row_frac + x

        t_margin = t_marg / row_frac
        b_margin = b_marg / row_frac

        fig = plt.figure(figsize=(6.5, height))
        gs = GridSpec(len(subdf), 3, figure=fig)

        if len(subdf) == 5:

            GI, GV, GO, ax1, ax2, ax3 = add_cutout(subdf.iloc[0], 0, fig, gs)
            EI, EV, EO, ax4, ax5, ax6 = add_cutout(subdf.iloc[1], 1, fig, gs)
            HI, HV, HO, ax7, ax8, ax9 = add_cutout(subdf.iloc[2], 2, fig, gs)
            RI, RV, RO, ax10, ax11, ax12 = add_cutout(subdf.iloc[3], 3, fig, gs)
            OI, OV, OO, ax13, ax14, ax15 = add_cutout(subdf.iloc[4], 4, fig, gs, keepxlabel=True)

        else:

            GI, GV, GO, ax1, ax2, ax3 = add_cutout(subdf.iloc[0], 0, fig, gs, keepxlabel=True)
            # EI, EV, EO, ax4, ax5, ax6 = add_cutout(subdf.iloc[1], 1, fig, gs)
            # HI, HV, HO, ax7, ax8, ax9 = add_cutout(subdf.iloc[2], 2, fig, gs)
            # BI, BV, BO, ax10, ax11, ax12 = add_cutout(subdf.iloc[3], 3, fig, gs, keepxlabel=True)

        fig.subplots_adjust(left=l_marg, right=1-r_marg, top=1-t_margin, bottom=b_margin,
                            hspace=hspace, wspace=wspace)
        
        fig.savefig(f'{savedir}/cutout-fig{fignum}.png', format='png', dpi=300)
        fig.savefig(f'{savedir}/cutout-fig{fignum}.pdf', format='pdf', dpi=300)

        # sleep to avoid requests.exceptions.ConnectionError
        time.sleep(20)
        
    plt.show()

if __name__ == '__main__':
    main()
