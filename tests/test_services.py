from pathlib import Path

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astroutils.io import FITSException, get_surveys
from astroutils.source import SelavyCatalogue

from cutout import ContourCutout, Cutout

SURVEYS = get_surveys()

cache_path = "tests/data/cache/"

no_selavy_case = (
    "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
    None,
    0.05 * u.deg,
    0.5 * u.arcsec,
)
no_selavy_case_naxis4 = (
    "tests/data/image.naxis4.i.SB9602.cont.taylor.0.restored.fits",
    None,
    0.05 * u.deg,
    0.5 * u.arcsec,
)
selavy_case = (
    "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
    "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt",
    0.05 * u.deg,
    0.5 * u.arcsec,
)
selavy_no_neighbours_case = (
    "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
    "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt",
    0.0004 * u.deg,
    0.5 * u.arcsec,
)
selavy_no_source_case = (
    "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
    "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt",
    0.01 * u.deg,
    0.1 * u.arcsec,
)


@pytest.mark.parametrize(
    "im_path, selavy_path, size, pos_err",
    [
        no_selavy_case,
        no_selavy_case_naxis4,
        selavy_case,
        selavy_no_neighbours_case,
        selavy_no_source_case,
    ],
)
def test_raw_cutout(im_path, selavy_path, size, pos_err, params):
    ra, dec, _ = params

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    cutout = Cutout(
        im_path,
        position,
        size,
        selavy=selavy_path,
        pos_err=pos_err,
    )

    template = "Cutout('{}', SkyCoord(ra={:.4f}, dec={:.4f}, unit='deg'), size={:.4f})"
    rep = template.format(
        im_path,
        ra,
        dec,
        size,
    )
    assert repr(cutout) == rep

    return


invalid_image_case = (
    "tests/data/invalid_image_path.fits",
    None,
)

invalid_selavy_case = (
    "tests/data/image.i.SB9602.cont.taylor.0.restored.fits",
    "tests/data/invalid_selavy_path.txt",
)


@pytest.mark.parametrize(
    "image_path, selavy_path",
    [invalid_image_case, invalid_selavy_case],
)
def test_raw_cutout_invalid_path_raises_error(image_path, selavy_path, params):
    ra, dec, size = params

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    with pytest.raises(FITSException):
        Cutout(
            image_path,
            position,
            size,
            selavy=selavy_path,
        )


field1 = pd.DataFrame(
    {
        "field": ["SB9602"],
        "sbid": ["SB9602"],
        "i_path": ["tests/data/image.i.SB9602.cont.taylor.0.restored.fits"],
        "v_path": [""],
        "cr_ra_pix": [4661.0],
        "cr_dec_pix": [-3255.0],
        "bmaj": [12.348672481212551],
        "bmin": [9.999193689545292],
        "dist_field_centre": [0.018464],
    }
)


field2 = pd.DataFrame(
    {
        "field": ["0618+12"],
        "sbid": ["SB8567"],
        "i_path": ["tests/data/iphas_racs_contours.fits"],
        "v_path": [""],
        "cr_ra_pix": [3884.0],
        "cr_dec_pix": [2364.0],
        "bmaj": [16.356075],
        "bmin": [15.137407],
        "dist_field_centre": [2.381194],
    }
)

mocked_fields = [field1, field2]


@pytest.mark.parametrize(
    "fieldname, survey",
    [
        ("SB9602", "gw1"),
        (None, "gw1"),
        ("SB9602", "vlass1"),
    ],
)
def test_local_cutout(fieldname, survey, params, mocker):
    ra, dec, size = params

    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[0])

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    Cutout(
        survey,
        position,
        size,
        fieldname=fieldname,
        sbid=fieldname,
    )


def test_local_nofields(params, mocker):
    ra, dec, size = params

    mocker.patch("cutout.services.find_fields", return_value=pd.DataFrame())

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    with pytest.raises(FITSException):
        Cutout(
            "gw1",
            position,
            size,
        )


mocked_surveys = SURVEYS[SURVEYS.survey == "mwats"].copy()
mocked_surveys["data_path"] = "tests/data/mwats"
mocked_surveys.set_index("survey", inplace=True)


def test_mwats_cutout(params, mocker):
    ra, dec, size = params

    mwats = SelavyCatalogue.from_aegean("tests/data/mwats/mwats_test.parq")
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[0])
    mocker.patch("cutout.services.SURVEYS", new=mocked_surveys)
    mocker.patch("cutout.services.SelavyCatalogue.from_aegean", return_value=mwats)

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    Cutout(
        "mwats",
        position,
        size,
    )


def test_mwats_cutout_out_of_zone(params, mocker):
    ra, _, size = params
    dec = 55

    mwats = SelavyCatalogue.from_aegean("tests/data/mwats/mwats_test.parq")
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[0])
    mocker.patch("cutout.services.SelavyCatalogue.from_aegean", return_value=mwats)

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    with pytest.raises(FITSException):
        Cutout(
            "mwats",
            position,
            size,
        )


@pytest.mark.parametrize("components", [True, False])
def test_local_cutout_selavy_ellipses(components, params, mocker):
    """Test that local cutouts with a valid selavy file have components set."""

    ra, dec, size = params

    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[0])

    if components:
        selavy = SelavyCatalogue(
            "tests/data/selavy-image.i.SB9602.cont.taylor.0.restored.components.txt"
        )
        mocker.patch("cutout.services.SelavyCatalogue.from_params", return_value=selavy)

    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    cutout = Cutout("gw1", position, size)

    if components:
        assert isinstance(cutout.components, pd.DataFrame)
    else:
        assert cutout.components is None


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::astropy.wcs.FITSFixedWarning")
@pytest.mark.parametrize(
    "survey, ra, dec, field_idx",
    [
        ("skymapper", 12.68, -25.3, 0),
        ("panstarrs", 12.68, -25.3, 0),
        ("decam", 12.68, -25.3, 0),
        ("iphas", 96.9, 11.5, 1),
        # 2MASS to test SkyView
        ("2massj", 12.68, -25.3, 0),
    ],
)
def test_api_contour_cutouts(
    survey,
    ra,
    dec,
    field_idx,
    mocker,
    cleanup_cache,
):
    """Test cutout services that depend upon external APIs.

    APIs in this category include SkyView, Skymapper, DECam, iPHAS, and PanSTARRS.
    """
    size = 0.05 * u.deg
    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[field_idx])

    ContourCutout(
        survey,
        position,
        size,
        contours="gw1",
    )


@pytest.mark.parametrize(
    "survey, ra, dec, field_idx",
    [
        ("skymapper", 12.68, 55.3, 0),
        ("panstarrs", 12.68, 55.3, 0),
        ("decam", 12.68, 55.3, 0),
        ("iphas", 12.38, -25.3, 0),
    ],
)
def test_api_contour_cutouts_out_of_zone(
    survey,
    ra,
    dec,
    field_idx,
    mocker,
    cleanup_cache,
):
    """Test cutout services that depend upon external APIs.

    APIs in this category include SkyView, Skymapper, DECam, iPHAS, and PanSTARRS.
    """
    size = 0.05 * u.deg
    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[field_idx])

    with pytest.raises(FITSException):
        ContourCutout(
            survey,
            position,
            size,
            contours="gw1",
        )


@pytest.mark.parametrize(
    "ra, dec, survey, field_index",
    [
        (10.6833, -25.304, "gw1", 0),
        (12.68, -25.3, "iphas", 0),
    ],
)
def test_bad_position_raises_error(
    ra,
    dec,
    survey,
    field_index,
    mocker,
    cleanup_cache,
):
    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[field_index])

    with pytest.raises(FITSException):
        Cutout(
            survey,
            position,
            size=0.05 * u.deg,
            contours="gw1",
        )


def test_skyview_invalid_survey_raises_error(mocker, cleanup_cache):
    position = SkyCoord(ra=12.68, dec=-25.3, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[0])

    with pytest.raises(FITSException):
        ContourCutout(
            "2massq",
            position,
            size=0.05 * u.deg,
            contours="gw1",
        )


def test_skyview_invalid_data_raises_error(mocker, cleanup_cache):
    position = SkyCoord(ra=0, dec=-0.005, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))

    with pytest.raises(FITSException):
        cutout = Cutout(
            "tests/data/fermi_baddata.fits",
            position,
            size=0.01 * u.deg,
        )
        cutout.plot()


@pytest.mark.slow
def test_skyview_out_of_zone_raises_error(mocker, cleanup_cache):
    position = SkyCoord(ra=12.68, dec=55.3, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[0])

    with pytest.raises(FITSException):
        Cutout(
            "sumss",
            position,
            size=0.05 * u.deg,
        )
