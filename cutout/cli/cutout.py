#!/usr/bin/env python

import logging
import os

import astropy.units as u
import click
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astroutils.io import FITSException, get_surveys
from astroutils.logger import setupLogger

from cutout import CornerMarker, Cutout

SURVEYS = get_surveys()
SURVEYS.set_index("survey", inplace=True)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-r",
    "--size",
    default=None,
    type=float,
    help="Size of the cutout in degrees. This is the edge length, not a cone search size.",
)
@click.option(
    "-s",
    "--stokes",
    type=click.Choice(["i", "v"]),
    default="i",
    help="Stokes parameter.",
)
@click.option(
    "-P",
    "--sign",
    is_flag=True,
    default=False,
    help="Invert polarisation sign.",
)
@click.option(
    "-c",
    "--contours",
    type=str,
    default=None,
    help="Survey data to use for contours.",
)
@click.option(
    "-Q",
    "--query-simbad",
    is_flag=True,
    default=False,
    help="Query SIMBAD and apply proper motion corrections.",
)
@click.option(
    "-e",
    "--epoch",
    type=float,
    default=None,
    help="Epoch in either decimalyear or mjd format.",
)
@click.option(
    "-f",
    "--fieldname",
    type=str,
    default=None,
    help="Fieldname (e.g. 1200+00) to pick.",
)
@click.option(
    "-B",
    "--sbid",
    type=str,
    default=None,
    help="SBID (e.g. SB32155) to pick.",
)
@click.option(
    "-p",
    "--psf",
    is_flag=True,
    help="Display the PSF alongside cutout.",
)
@click.option(
    "-L",
    "--corner",
    is_flag=True,
    default=False,
    help="Display corner marker at central cutout position.",
)
@click.option(
    "-V",
    "--selavy",
    type=click.Path(),
    default=None,
    help="Path to selavy file for overlay of source components.",
)
@click.option(
    "--neighbours/--no-neighbours",
    default=True,
    help="Display location of neighbouring sources.",
)
@click.option(
    "-t",
    "--annotation",
    type=str,
    default=None,
    help="Annotated text.",
)
@click.option(
    "-T",
    "--tiletype",
    type=click.Choice(["TILES", "COMBINED"]),
    default="TILES",
    help="Produce cutout from tile images or combined mosaics.",
)
@click.option(
    "-H",
    "--header",
    is_flag=True,
    default=False,
    help="Display FITS header of target data.",
)
@click.option(
    "-C",
    "--cmap",
    type=str,
    default="gray_r",
    help="Name of colormap to use.",
)
@click.option(
    "-m",
    "--maxnorm",
    is_flag=True,
    help="Use data max for normalisation.",
)
@click.option(
    "-n",
    "--vmax",
    type=float,
    default=None,
    help="Specify vmax for flux normalisation.",
)
@click.option(
    "-d",
    "--vmin",
    type=float,
    default=None,
    help="Specify vmin for flux normalisation.",
)
@click.option(
    "-b",
    "--band",
    type=click.Choice(["g", "r", "i", "z", "y"]),
    default="g",
    help="Filter band for optical surveys (e.g. PanSTARRS, DECam).",
)
@click.option(
    "-o",
    "--obfuscate",
    is_flag=True,
    default=False,
    help="Remove coordinates and axes from output.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging mode",
)
@click.option(
    "-S",
    "--save",
    type=click.Path(),
    default=None,
    help="Save path for cutout in PNG format.",
)
@click.option(
    "-F",
    "--savefits",
    type=click.Path(),
    default=None,
    help="Save path for cutout in FITS format.",
)
@click.argument("RA", type=str)
@click.argument("Dec", type=str)
@click.argument("Survey", type=str)
def main(
    size,
    contours,
    query_simbad,
    epoch,
    fieldname,
    sbid,
    stokes,
    sign,
    psf,
    corner,
    selavy,
    neighbours,
    annotation,
    tiletype,
    header,
    cmap,
    maxnorm,
    vmax,
    vmin,
    band,
    obfuscate,
    verbose,
    save,
    savefits,
    ra,
    dec,
    survey,
):
    """Generate image cutout from multi-wavelength survey data.

    Available surveys:

    \b
    ---------------------------------
    Radio
    ---------------------------------
    racs-low                 RACS Low
    racs-mid                 RACS Mid
    vastp[Nx]   VAST Pilot Epoch N[x]
    gw[N]           s190814bv Epoch N
    swagx                      SWAG-X
    dwf-ngc[N]    DWF '21 NGC Epoch N
    dwf-frb[N]    DWF '21 FRB Epoch N
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

    unit = u.hourangle if ":" in ra or "h" in ra else u.deg

    position = SkyCoord(ra=ra, dec=dec, unit=(unit, u.deg))
    psign = -1 if sign else 1

    # If passing raw FITS in, use RACS Low parameters
    s = SURVEYS.loc["racs-low"] if os.path.isfile(survey) else SURVEYS.loc[survey]
    if not size:
        size = s.cutout_size

    size *= u.deg

    try:
        cutout = Cutout(
            survey,
            position,
            size=size,
            stokes=stokes,
            tiletype=tiletype,
            epoch=epoch,
            fieldname=fieldname,
            sbid=sbid,
            sign=psign,
            cmap=cmap,
            selavy=selavy,
            neighbours=neighbours,
            band=band,
            vmin=vmin,
            vmax=vmax,
            maxnorm=maxnorm,
            compact=True,
        )

        cutout.plot()

        if query_simbad:
            simbad, newpos = cutout.query_simbad()
            logger.info(f"SIMBAD results:\n {simbad.head()}")
            cutout.add_markers(
                newpos,
                color="dodgerblue",
                marker="x",
                s=100,
            )

        if corner:
            span = len(cutout.data) / 4
            offset = len(cutout.data) / 8
            corner = CornerMarker(
                position,
                cutout.wcs,
                colour="r",
                span=span,
                offset=offset,
            )
            cutout.add_cornermarker(corner)

        if contours:
            cutout.add_contours(contours, position)

        if psf:
            cutout.add_psf()

        if annotation:
            cutout.add_annotation(annotation, location="upper left")

        if obfuscate:
            cutout.hide_coords()

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


if __name__ == "__main__":
    main()
