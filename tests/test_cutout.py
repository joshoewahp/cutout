import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroutils.io import FITSException
from astroutils.logger import setupLogger

from cutout import CornerMarker, get_simbad

logger = logging.getLogger()

setupLogger(True)


@pytest.mark.parametrize(
    "ra, dec, in_pixel_range",
    [
        (12.68, -25.3, True),
        (12.61, -25.3, False),
    ],
)
def test_cornermarker_in_pixel_range(ra, dec, in_pixel_range, image_products):
    """Test pixel range predicate and WCS."""

    position = SkyCoord(ra=ra, dec=dec, unit="deg")
    wcs = image_products.get("wcs")
    data = image_products.get("data")

    marker = CornerMarker(
        position,
        wcs,
        colour="r",
        span=2.5,
        offset=0.5,
    )

    # Test data position methods
    assert marker.in_pixel_range(0, len(data)) == in_pixel_range


@pytest.mark.parametrize(
    "ra, dec",
    [
        (12.68, -25.3),
        (12.61, -25.3),
    ],
)
def test_cornermarker_line_parameters(ra, dec, image_products):
    """Test construction of RA/Dec lines with offset/span parameters."""

    position = SkyCoord(ra=ra, dec=dec, unit="deg")
    wcs = image_products.get("wcs")

    offset = 0.5
    span = 2.5

    marker = CornerMarker(
        position,
        wcs,
        colour="r",
        span=2.5,
        offset=0.5,
    )

    x = marker.datapos[0] - 1
    y = marker.datapos[1] - 1

    assert marker.raline._xorig == [x - offset, x - span]
    assert marker.raline._yorig == [y, y]
    assert marker.decline._xorig == [x, x]
    assert marker.decline._yorig == [y + offset, y + span]


def test_get_simbad(position):
    pos = position("no_pm")
    simbad = get_simbad(pos)

    assert isinstance(simbad, pd.DataFrame)


def test_get_simbad_no_results():
    pos = SkyCoord(ra=191, dec=-80, unit="deg")
    simbad = get_simbad(pos)

    assert simbad is None


def test_getattr_overload(cutout):
    """Need to check as we overload __getattr__ to avoid a RecursionError."""
    c = cutout()

    with pytest.raises(AttributeError):
        c.invalid_attr


def test_plot_setup(cutout):
    c = cutout()
    c.plot()


def test_plot_setup_external_fig(cutout):
    c = cutout()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=c.wcs)

    c.plot(fig=fig, ax=ax)


@pytest.mark.parametrize(
    "options",
    [
        {"compact": True},
        {"maxnorm": True},
        {"vmin": -5},
        {"vmax": 5},
        {"epoch": 2020.5},
        {"stokes": "v"},
    ],
)
def test_cutout_plot_options(cutout, options):
    c = cutout(options=options)
    c.plot()


@pytest.mark.parametrize(
    "options",
    [
        {"rmslevels": True},
        {"peaklevels": True},
        {"clabels": True},
        {"bar": True},
    ],
)
def test_contourcutout_plot_options(cutout, options):
    c = cutout(options=options, contours=True)
    c.plot(**options)


datekeys = [
    ("MJD-OBS", 58712),
    ("MJD", 58712),
    ("DATE-OBS", "2021-12-28T10:27:18.4"),
    ("DATE", "2021-12-28T10:27:18.4"),
]


@pytest.mark.parametrize("datekey", datekeys)
def test_alternate_date_header_keys(cutout, datekey):
    c = cutout()

    # Remove all other datekeys
    for key, date in datekeys:
        if key in c.header.keys():
            del c.header[key]

    key, date = datekey
    c.header[key] = date

    c.plot()


def test_add_cornermarker(cutout, params):
    c = cutout()

    ra, dec, _ = params

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    marker = CornerMarker(
        position,
        c.wcs,
        colour="r",
        span=len(c.data) / 4,
        offset=len(c.data) / 8,
    )

    c.plot()
    c.add_cornermarker(marker)


def test_add_cornermarker_out_of_range_raises_error(cutout, params, caplog):
    c = cutout()

    ra, _, _ = params

    position = SkyCoord(ra=ra, dec=-30, unit="deg")

    marker = CornerMarker(
        position,
        c.wcs,
        colour="r",
        span=len(c.data) / 4,
        offset=len(c.data) / 8,
    )

    c.plot()
    c.add_cornermarker(marker)

    assert "Cornermarker will be disabled" in caplog.text


def test_add_annotation(cutout):
    c = cutout()

    c.plot()
    c.add_annotation("test")


@pytest.mark.parametrize("frameon", [False, True])
def test_add_psf(cutout, frameon):
    c = cutout()

    c.plot()
    c.add_psf(frameon=frameon)


def test_add_psf_missing_beam_header_logs_warning(cutout, caplog):
    c = cutout()
    del c.header["BMAJ"]

    c.plot()
    c.add_psf()

    assert "disabling PSF marker" in caplog.text


def test_add_psf_missing_cdelt_header_uses_cd1_1(cutout, caplog):
    c = cutout()
    c.header["CD1_1"] = c.header["CDELT1"]
    del c.header["CDELT1"]

    c.plot()
    c.add_psf()


@pytest.mark.parametrize("axis", ["x", "y", "both"])
@pytest.mark.parametrize("ticks", [True, False])
@pytest.mark.parametrize("labels", [True, False])
def test_hide_coords(cutout, axis, ticks, labels):
    c = cutout()

    c.plot()
    c.hide_coords(axis=axis, ticks=ticks, labels=labels)


@pytest.mark.parametrize("neighbours", [True, False])
@pytest.mark.parametrize("source", [True, False])
@pytest.mark.parametrize("contours", [True, False])
@pytest.mark.parametrize("pos_err", [1 * u.arcsec, 0 * u.arcsec])
def test_switch_to_offsets(cutout, contours, source, neighbours, pos_err):
    options = {
        "neighbours": neighbours,
        "source": source,
        "pos_err": pos_err,
    }
    c = cutout(options=options, contours=contours)

    c.plot()
    c.switch_to_offsets()


@pytest.mark.parametrize("coord", ["no_pm", "no_pm_north"])
@pytest.mark.parametrize("align", [True, False])
def test_set_ylabel(cutout, coord, align):
    c = cutout(coord=coord)

    label = "test label"

    c.plot()
    c.set_ylabel(label, align=align)


def test_save(cutout, cleanup_cache):
    c = cutout()
    c.plot()

    cache_path = "tests/data/cache/"

    c.save(f"{cache_path}/test.png")
    c.savefits(f"{cache_path}/test.fits")

    assert os.path.exists(f"{cache_path}/test.png")
    assert os.path.exists(f"{cache_path}/test.fits")


@pytest.mark.parametrize(
    "mjd, coord",
    [
        (None, "pm"),
        (58000, "pm_offset"),
    ],
)
def test_contourcutout_correct_proper_motion(
    mjd,
    coord,
    cutout,
    mock_simbad,
    mocker,
):
    simbad = mock_simbad([coord])
    mocker.patch("cutout.cutout.get_simbad", return_value=simbad)

    c = cutout(contours=True, coord="pm")
    c.plot()

    c.correct_proper_motion(mjd=mjd)


def test_contourcutout_correct_proper_motion_no_contour_mjd(cutout):
    c = cutout(contours=True)
    c.mjd = None

    c.plot()

    with pytest.raises(FITSException):
        c.correct_proper_motion()


def test_contourcutout_correct_proper_motion_no_radio_mjd(cutout):
    c = cutout(contours=True)
    c.radio.mjd = None

    c.plot()

    with pytest.raises(FITSException):
        c.correct_proper_motion()


def test_contourcutout_correct_proper_motion_no_simbad(
    cutout,
    mocker,
    caplog,
):
    c = cutout(contours=True)

    mocker.patch("cutout.cutout.get_simbad", return_value=None)

    c.plot()
    c.correct_proper_motion()

    assert "No high proper-motion objects" in caplog.text


@pytest.mark.parametrize("mjd", [None, 57000])
def test_contourcutout_add_pm_location(
    mjd,
    cutout,
    mock_simbad,
    mocker,
):
    c = cutout(contours=True)

    simbad = mock_simbad(["pm"])
    mocker.patch("cutout.cutout.get_simbad", return_value=simbad)

    c.plot()
    c.correct_proper_motion(mjd)
    c.add_pm_location()


def test_contourcutout_add_pm_location_bad_order_raises_error(
    cutout,
    mock_simbad,
    mocker,
):
    c = cutout(contours=True)

    simbad = mock_simbad(["pm"])
    mocker.patch("cutout.cutout.get_simbad", return_value=simbad)

    with pytest.raises(FITSException):
        c.plot()
        c.add_pm_location()


@pytest.mark.parametrize(
    "shift_epoch",
    [Time(2020.5, format="decimalyear"), None],
)
def test_add_contours(shift_epoch, cutout, position):
    c = cutout(contours=True)

    c.plot()
    c.add_contours(
        "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
        position("pm"),
        shift_epoch=shift_epoch,
    )


def test_shift_coordinate_grid(cutout, position):
    c = cutout(contours=True)

    c.plot()
    c.shift_coordinate_grid(
        position("pm"),
        Time(2020.5, format="decimalyear"),
    )
