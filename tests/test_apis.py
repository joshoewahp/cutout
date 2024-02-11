from pathlib import Path

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from cutout import Cutout

cache_path = "tests/data/cache/"


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
def test_api_cutouts(
    survey,
    ra,
    dec,
    field_idx,
    mocker,
    mocked_fields,
    cleanup_cache,
):
    """Test cutout services that depend upon external APIs.

    APIs in this category include SkyView, Skymapper, DECam, iPHAS, and PanSTARRS.
    """
    size = 0.05 * u.deg
    position = SkyCoord(ra=ra, dec=dec, unit="deg")

    mocker.patch("cutout.services.cutout_cache", new=Path(cache_path))
    mocker.patch("cutout.services.find_fields", return_value=mocked_fields[field_idx])

    Cutout(
        survey,
        position,
        size,
    )
