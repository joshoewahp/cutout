#!/suphys/jpri6587/bin/miniconda3/bin/python

import sys
import click
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tools.cutout import Cutout, ContourCutout, FITSException
from tools.logger import Logger

SURVEYS = pd.read_json('./config/surveys.json')
SURVEYS.set_index('survey', inplace=True)


@click.command()
@click.option('-r', '--radius', default=None, help="Size of the cutout in degrees.", type=float)
@click.option('-P', '--sign', is_flag=True, default=False, help="Invert polarisation sign.")
@click.option('-c', '--contours', type=str, default=None,
              help="Survey data to use for contours.")
@click.option('-l', '--clabels', is_flag=True, help="Display contour level labels.", default=False)
@click.option('--pm/--no-pm', default=False,
              help="Trigger proper motion correction for nearby stars.")
@click.option('-e', '--epoch', type=float, default=2020.0,
              help="Epoch in either decimalyear or mjd format.")
@click.option('-p', '--psf', is_flag=True, help="Display the PSF alongside cutout.")
@click.option('-s', '--source', is_flag=True, help="Display location of located source.")
@click.option('-L', '--corner', is_flag=True, default=False,
              help="Use corner marker instead of source ellipse.")
@click.option('-n', '--neighbors', is_flag=True,
              help="Display location of neighboring sources.")
@click.option('-t', '--annotation', type=str, default=None,
              help="Annotated text.")
@click.option('-B', '--basesurvey', default='racsI',
              help="Name of survey to use for primary positional accuracy and time parameters.")
@click.option('-C', '--cmap', type=str, default='gray_r',
              help="Colorbar selection to contour plots")
@click.option('-m', '--maxnorm', is_flag=True, help="Use data max for normalisation.")
@click.option('-b', '--band', type=click.Choice(['g', 'r', 'i', 'z', 'y']), default='g',
              help="PanSTARRS filter band.")
@click.option('-o', '--obfuscate', is_flag=True,
              help="Remove coordinates and axes from output.")
@click.option('-v', '--verbose', is_flag=True, help="Report source and neighbor properties.")
@click.option('-S', '--save', type=click.Path(), default=None)
@click.option('-F', '--savefits', type=click.Path(), default=None)
@click.argument('RA', type=str)
@click.argument('Dec', type=str)
@click.argument('Survey', type=str)
def main(radius, contours, clabels, pm, epoch, sign, psf, source, corner, neighbors, annotation,
         basesurvey, cmap, maxnorm, band, obfuscate, verbose, save, savefits, ra, dec, survey):
    """Generate SED and lightcurve from radio survey data."""

    level = 'DEBUG' if verbose else 'INFO'
    logger = Logger(__name__, streamlevel=level).logger
    
    if ':' in ra:
        unit = u.hourangle
    else:
        unit = u.deg

    position = SkyCoord(ra=ra, dec=dec, unit=(unit, u.deg))
    psign = -1 if sign else 1

    s = SURVEYS.loc[survey]
    if not radius:
        radius = s.radius

    try:
        if s.radio and not contours:
            cutout = Cutout(survey, position, radius=radius, psf=psf, source=source, corner=corner,
                            neighbors=neighbors, annotation=annotation, basesurvey=basesurvey,
                            band=band, maxnorm=maxnorm, cmap=cmap, obfuscate=obfuscate,
                            verbose=verbose, sign=psign, pm=pm, epoch=epoch)
        elif contours:
            cutout = ContourCutout(survey, position, radius=radius, contours=contours,
                                clabels=clabels, psf=psf, source=source, corner=corner,
                                neighbors=neighbors, annotation=annotation,
                                basesurvey=basesurvey, band=band, cmap=cmap, maxnorm=maxnorm,
                                obfuscate=obfuscate, verbose=verbose, sign=psign, pm=pm,
                                epoch=epoch)
        else:
            cutout = ContourCutout(survey, position, radius=radius, contours='racsI', psf=psf,
                                source=source, corner=corner, neighbors=neighbors,
                                annotation=annotation, basesurvey=basesurvey, band=band,
                                cmap=cmap, maxnorm=maxnorm, obfuscate=obfuscate, verbose=verbose,
                                sign=psign, pm=pm, epoch=epoch)

        cutout.plot()
        logger.debug(f'RMS: {np.sqrt(np.mean(np.square(cutout.data))):.2f} mJy')
    except FITSException as e:
        logger.error(e)

    if save:
        cutout.save(save)

    if savefits:
        cutout.savefits(savefits)

    plt.show()


if __name__ == '__main__':
    main()
