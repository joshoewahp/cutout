import os

import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from cutout import ContourCutout, Cutout


@pytest.fixture
def params():
    ra = 12.6833
    dec = -25.304
    size = 0.05 * u.deg

    return ra, dec, size


@pytest.fixture
def position(params):
    ra, dec, _ = params
    dist = Distance(parallax=100 * u.mas)

    def _position(coord):
        no_pm = SkyCoord(ra=ra, dec=dec, unit="deg")
        no_pm_north = SkyCoord(ra=ra, dec=5.304, unit="deg")
        pm = SkyCoord(
            ra=ra,
            dec=dec,
            unit=("deg", "deg"),
            distance=dist,
            pm_ra_cosdec=1000 * u.mas / u.yr,
            pm_dec=1000 * u.mas / u.yr,
            obstime=Time(58712, format="mjd"),
        )
        pm_offset = SkyCoord(
            ra=ra,
            dec=dec - 0.05,
            unit=("deg", "deg"),
            distance=dist,
            pm_ra_cosdec=1000 * u.mas / u.yr,
            pm_dec=1000 * u.mas / u.yr,
            obstime=Time(58712, format="mjd"),
        )

        c = {
            "no_pm": no_pm,
            "no_pm_north": no_pm_north,
            "pm": pm,
            "pm_offset": pm_offset,
        }

        return c[coord]

    return _position


@pytest.fixture
def cleanup_cache():
    cache_path = "tests/data/cache/"
    os.system(f"mkdir -p {cache_path}")

    yield

    cache_path = "tests/data/cache/"

    os.system(f"rm -r {cache_path}")


@pytest.fixture(autouse=True)
def cleanup_pyplot():
    yield

    plt.close("all")


@pytest.fixture
def mock_simbad(position):
    def _simbad(objects):
        simbad = pd.DataFrame(
            {
                "Object": ["no_pm", "pm", "pm_offset"],
                "j2000pos": [
                    position("no_pm"),
                    position("pm"),
                    position("pm_offset"),
                ],
            }
        )
        simbad = simbad[simbad.Object.isin(objects)]
        return simbad

    return _simbad


@pytest.fixture()
def image_products():
    with fits.open("tests/data/image.i.SB9602.cont.taylor.0.restored.fits") as hdul:
        header, data = hdul[0].header, hdul[0].data
        wcs = WCS(header, naxis=2)

    products = {
        "data": data,
        "header": header,
        "wcs": wcs,
    }

    return products


@pytest.fixture
def data_path():
    def _data_path(coord):
        im_paths = {
            "no_pm": "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
            "no_pm_north": "tests/data/test_cutout_racsmid_north.fits",
            "pm": "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
            "pm_offset": "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
        }
        sel_paths = {
            "no_pm": "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt",
            "no_pm_north": None,
            "pm": "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt",
            "pm_offset": "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt",
        }

        return im_paths[coord], sel_paths[coord]

    return _data_path


@pytest.fixture
def cutout(position, data_path):
    def _cutout(contours=None, coord="no_pm", options=None):
        if not options:
            options = dict()

        pos = position(coord)

        impath, selpath = data_path(coord)

        if contours:
            contours = impath

            CutoutClass = ContourCutout
        else:
            CutoutClass = Cutout

        c = CutoutClass(
            impath,
            pos,
            size=0.01 * u.deg,
            selavy=selpath,
            contours=contours,
            **options,
        )

        return c

    return _cutout


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skipper = pytest.mark.skip(reason="Only run when --run-slow is given")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skipper)
