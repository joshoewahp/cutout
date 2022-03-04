#!/usr/bin/env python

import click
import logging
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from cutout import Cutout, ContourCutout
from astroutils.logger import setupLogger
from astroutils.io import FITSException, get_surveys


SURVEYS = get_surveys()
SURVEYS.set_index('survey', inplace=True)

logger = logging.getLogger(__name__)


@click.command()
@click.option('-r', '--radius', default=None, help="Size of the cutout in degrees.", type=float)
@click.option('-s', '--stokes', type=click.Choice(['i', 'v']), default='i',
              help="Stokes parameter.")
@click.option('-P', '--sign', is_flag=True, default=False, help="Invert polarisation sign.")
@click.option('-c', '--contours', type=str, default=None,
              help="Survey data to use for contours.")
@click.option('-l', '--clabels', is_flag=True, help="Display contour level labels.", default=False)
@click.option('--pm/--no-pm', default=False,
              help="Trigger proper motion correction for nearby stars.")
@click.option('-e', '--epoch', type=float, default=None,
              help="Epoch in either decimalyear or mjd format.")
@click.option('-p', '--psf', is_flag=True, help="Display the PSF alongside cutout.")
@click.option('-L', '--corner', is_flag=True, default=False,
              help="Use corner marker instead of source ellipse.")
@click.option('--neighbours/--no-neighbours', default=True,
              help="Display location of neighbouring sources.")
@click.option('-t', '--annotation', type=str, default=None,
              help="Annotated text.")
@click.option('-H', '--header', is_flag=True, default=False,
              help="Display FITS header of target data.")
@click.option('-B', '--basesurvey', default='racs-low',
              help="Name of survey to use for primary positional accuracy and time parameters.")
@click.option('-C', '--cmap', type=str, default='gray_r',
              help="Name of colormap to use.")
@click.option('-m', '--maxnorm', is_flag=True, help="Use data max for normalisation.")
@click.option('-n', '--vmax', type=float, default=None, help="Specify vmax for flux normalisation.")
@click.option('-d', '--vmin', type=float, default=None, help="Specify vmin for flux normalisation.")
@click.option('-b', '--band', type=click.Choice(['g', 'r', 'i', 'z', 'y']), default='g',
              help="Filter band for optical surveys (e.g. PanSTARRS, DECam).")
@click.option('-o', '--obfuscate', is_flag=True, default=False,
              help="Remove coordinates and axes from output.")
@click.option('-v', '--verbose', is_flag=True, help="Enable verbose logging mode")
@click.option('-S', '--save', type=click.Path(), default=None,
              help="Save path for cutout in PNG format.")
@click.option('-F', '--savefits', type=click.Path(), default=None,
              help="Save path for cutout in FITS format.")
@click.argument('RA', type=str)
@click.argument('Dec', type=str)
@click.argument('Survey', type=str)
def main(radius, contours, clabels, pm, epoch, stokes, sign, psf, corner, neighbours, annotation, header,
         basesurvey, cmap, maxnorm, vmax, vmin, band, obfuscate, verbose, save, savefits, ra, dec, survey):
    """Generate image cutout from multi-wavelength survey data.

    Available surveys:

    \b
    Radio
    ---------------------------------
    gw1             s190814bv Epoch 1
    gw2             s190814bv Epoch 2
    gw3             s190814bv Epoch 3
    gw4             s190814bv Epoch 4
    swagx                      SWAG-X
    racs-low                 RACS Low
    racs-mid                 RACS Mid
    vastp1                    VAST P1
    vastp2                    VAST P2
    vastp3x                  VAST P3x
    vastp4x                  VAST P4x
    vastp5x                  VAST P5x
    vastp6x                  VAST P6x
    vastp7x                  VAST P7x
    vastp8                    VAST P8
    vastp9                    VAST P9
    vastp10x                VAST P10x
    vastp11x                VAST P11x
    vastp12                  VAST P12
    vlass                       VLASS
    sumss                       SUMSS
    nvss                         NVSS
    tgss                         TGSS
    gleam                       GLEAM
    ---------------------------------
    Infra-red
    ---------------------------------
    2massh                    2MASS H
    2massj                    2MASS J
    2massk                    2MASS K
    wise3_4                  WISE 3.4
    wise4_6                  WISE 4.6
    wise12                    WISE 12
    wise22                    WISE 22
    ---------------------------------
    Optical
    ---------------------------------
    decam                    DECam LS
    panstarrs               PanSTARRS
    skymapper               Skymapper
    dss                           DSS
    ---------------------------------
    UV / X-ray / Gamma-ray
    ---------------------------------
    galex_nuv           GALEX Near UV
    fermi             Fermi 3-300 GeV
    rass_soft       RASS Soft (ROSAT)
    swift_xrtcnt         SWIFT Counts
    """

    setupLogger(verbose)

    unit = u.hourangle if ':' in ra or 'h' in ra else u.deg

    position = SkyCoord(ra=ra, dec=dec, unit=(unit, u.deg))
    psign = -1 if sign else 1

    s = SURVEYS.loc[survey]
    if not radius:
        radius = s.radius

    try:
        if s.radio and not contours:
            cutout = Cutout(survey, position, radius=radius, psf=psf, corner=corner,
                            neighbours=neighbours, annotation=annotation, stokes=stokes,
                            basesurvey=basesurvey, band=band, maxnorm=maxnorm, vmax=vmax,
                            vmin=vmin, cmap=cmap, obfuscate=obfuscate, verbose=verbose,
                            sign=psign, pm=pm, epoch=epoch)
        elif contours:
            cutout = ContourCutout(survey, position, radius=radius, contours=contours,
                                   clabels=clabels, psf=psf, corner=corner, stokes=stokes,
                                   neighbours=neighbours, annotation=annotation,
                                   basesurvey=basesurvey, band=band, cmap=cmap,
                                   maxnorm=maxnorm, vmax=vmax, vmin=vmin, obfuscate=obfuscate,
                                   verbose=verbose, sign=psign, pm=pm, epoch=epoch)
        else:
            cutout = ContourCutout(survey, position, radius=radius, contours='racs-low',
                                   stokes=stokes, psf=psf, corner=corner, neighbours=neighbours,
                                   annotation=annotation, basesurvey=basesurvey, band=band,
                                   cmap=cmap, maxnorm=maxnorm, vmax=vmax, vmin=vmin, sign=psign,
                                   obfuscate=obfuscate, verbose=verbose, pm=pm, epoch=epoch)

        cutout.plot()

        if header:
            logger.info(cutout.header.items)

        if save:
            cutout.save(save)

        if savefits:
            cutout.savefits(savefits)

    except FITSException as e:
        if verbose:
            logger.exception(e)
        else:
            logger.error(e)

    plt.show()


if __name__ == '__main__':
    main()
