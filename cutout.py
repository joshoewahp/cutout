#!/usr/bin/env python

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
@click.option('-L', '--corner', is_flag=True, default=False,
              help="Use corner marker instead of source ellipse.")
@click.option('--neighbours/--no-neighbours', default=True,
              help="Display location of neighbouring sources.")
@click.option('-t', '--annotation', type=str, default=None,
              help="Annotated text.")
@click.option('-B', '--basesurvey', default='racsI',
              help="Name of survey to use for primary positional accuracy and time parameters.")
@click.option('-C', '--cmap', type=str, default='gray_r',
              help="Colorbar selection to contour plots.")
@click.option('-m', '--maxnorm', is_flag=True, help="Use data max for normalisation.")
@click.option('-b', '--band', type=click.Choice(['g', 'r', 'i', 'z', 'y']), default='g',
              help="Filter band for optical surveys (e.g. PanSTARRS, DECam).")
@click.option('-o', '--obfuscate', is_flag=True,
              help="Remove coordinates and axes from output.")
@click.option('-v', '--verbose', is_flag=True, help="Report source and neighbour properties.")
@click.option('-S', '--save', type=click.Path(), default=None,
              help="Save path for cutout in PNG format.")
@click.option('-F', '--savefits', type=click.Path(), default=None,
              help="Save path for cutout in FITS format.")
@click.argument('RA', type=str)
@click.argument('Dec', type=str)
@click.argument('Survey', type=str)
def main(radius, contours, clabels, pm, epoch, sign, psf, corner, neighbours, annotation,
         basesurvey, cmap, maxnorm, band, obfuscate, verbose, save, savefits, ra, dec, survey):
    """Generate image cutout from multi-wavelength survey data.

    Available surveys:

    \b
    Radio
    ---------------------------------
    gw1I        s190814bv Epoch 1 (I)
    gw2I        s190814bv Epoch 2 (I)
    gw3I        s190814bv Epoch 3 (I)
    gw4I        s190814bv Epoch 4 (I)
    racsI                    RACS (I)
    racsV                    RACS (V)
    vastp1I               VAST P1 (I)
    vastp1V               VAST P1 (V)
    vastp2I               VAST P2 (I)
    vastp2V               VAST P2 (V)
    vastp3xI             VAST P3x (I)
    vastp3xV             VAST P3x (V)
    vastp4xI             VAST P4x (I)
    vastp4xV             VAST P4x (V)
    vastp5xI             VAST P5x (I)
    vastp5xV             VAST P5x (V)
    vastp6xI             VAST P6x (I)
    vastp6xV             VAST P6x (V)
    vastp7xI             VAST P7x (I)
    vastp7xV             VAST P7x (V)
    vastp8I               VAST P8 (I)
    vastp8V               VAST P8 (V)
    vastp9I               VAST P9 (I)
    vastp9V               VAST P9 (V)
    vastp10xI           VAST P10x (I)
    vastp10xV           VAST P10x (V)
    vastp11xI           VAST P11x (I)
    vastp11xV           VAST P11x (V)
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
            cutout = Cutout(survey, position, radius=radius, psf=psf, corner=corner,
                            neighbours=neighbours, annotation=annotation, basesurvey=basesurvey,
                            band=band, maxnorm=maxnorm, cmap=cmap, obfuscate=obfuscate,
                            verbose=verbose, sign=psign, pm=pm, epoch=epoch)
        elif contours:
            cutout = ContourCutout(survey, position, radius=radius, contours=contours,
                                   clabels=clabels, psf=psf, corner=corner, neighbours=neighbours,
                                   annotation=annotation, basesurvey=basesurvey, band=band,
                                   cmap=cmap, maxnorm=maxnorm, obfuscate=obfuscate,
                                   verbose=verbose, sign=psign, pm=pm, epoch=epoch)
        else:
            cutout = ContourCutout(survey, position, radius=radius, contours='racsI', psf=psf,
                                   corner=corner, neighbours=neighbours, annotation=annotation,
                                   basesurvey=basesurvey, band=band, cmap=cmap, maxnorm=maxnorm,
                                   obfuscate=obfuscate, verbose=verbose, sign=psign, pm=pm,
                                   epoch=epoch)

        cutout.plot()
    except FITSException as e:
        if verbose:
            logger.exception(e)
        else:
            logger.error(e)

    if save:
        cutout.save(save)

    if savefits:
        cutout.savefits(savefits)

    plt.show()


if __name__ == '__main__':
    main()
