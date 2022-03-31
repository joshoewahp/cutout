import pytest
import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

from cutout.services import (
    CutoutService,
    LocalCutout,
)

class MockCutout:

    def __init__(self, survey, position, size):
        self.survey = survey
        self.position = position
        self.size = size
        self.band = 'g'
        self.ra = position.ra
        self.dec = position.dec

@pytest.mark.parametrize(
    "survey, ra, dec",
    [
        ("racs-low", 0, 0),
    ]
)
def test_local_cutout_service(survey, ra, dec, mocker):
    
    surveys = pd.DataFrame({
        'survey': ['racs-low', 'racs-mid', 'vastp8', 'gw3', 'dwf-ngc-10'],
        'pos_err': [4, 4, 4, 4, 4],
    })
    # mocker.patch("cutout.cutout.services.get_surveys", return_value=surveys)


    position = SkyCoord(ra=ra, dec=dec, unit='deg')
    cutout = MockCutout(survey, position, size=1*u.arcmin)

    # cutout = LocalCutout(cutout)
    # print(cutout)
