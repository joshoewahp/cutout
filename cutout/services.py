import io
import os
import logging
import requests
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from abc import ABC, abstractmethod
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.time import Time
from astropy.table import Table
from astropy.nddata import Cutout2D
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from astropy.wcs.utils import proj_plane_pixel_scales
from pathlib import Path
from urllib.error import HTTPError

from astroutils.io import table2df, find_fields, FITSException, get_surveys, get_config
from astroutils.source import SelavyCatalogue

warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

config = get_config()
aux_path = Path(config['DATA']['aux_path'])
vlass_path = Path(config['DATA']['vlass_path'])
cutout_cache = aux_path / 'cutouts'

SURVEYS = get_surveys()
SURVEYS.set_index('survey', inplace=True)


logger = logging.getLogger(__name__)


class CutoutService(ABC):

    def __init__(self, cutout):
        self.plot_source = False
        self.plot_neighbours = False

        self.fetch_data(cutout)
        self._make_cutout(cutout)

    def _set_cache_path(self, cutout):
        band = f'_{cutout.band}' if 'band' in cutout.__dict__ else '' 
        path = f"{cutout.size*6:.4f}arcmin_{cutout.ra:.4f}_{cutout.dec:.4f}{band}.fits"
        self.filepath = cutout_cache / cutout.survey / path

    @abstractmethod
    def fetch_data(self, cutout, *args):
        """Fetch cutout data from source and set filepath attribute accordingly."""

    def _find_neighbours(self, cutout, components):
        """Set source and neighbour attributes from component list."""

        components.sort_values('d2d', inplace=True)

        # Set nearest component within positional uncertainty
        # as the target source and all others as neighbours
        if components.iloc[0].d2d < SURVEYS.loc[cutout.survey].pos_err:
            self.source = components.iloc[0]
            self.neighbours = components.iloc[1:]
            self.plot_source = True
        else:
            self.source = None
            self.neighbours = components

        self.plot_neighbours = cutout.options.get('neighbours', True)

        if self.source is not None:
            logger.debug(f'Source: \n{self.source}')

        if not self.neighbours.empty:
            nn = self.neighbours.iloc[0]
            neighbour_view = self.neighbours[[
                'ra_deg_cont',
                'dec_deg_cont',
                'maj_axis',
                'min_axis',
                'flux_peak',
                'flux_int',
                'rms_image',
                'd2d'
            ]]
            logger.debug(f'Nearest neighbour coordinates: \n{nn.ra_deg_cont} {nn.dec_deg_cont}')
            logger.debug(f'Nearest 5 Neighbours \n{neighbour_view.head()}')


    def _make_cutout(self, cutout):

        with fits.open(self.filepath) as hdul:
            logger.debug(f"Making cutout from FITS image located at:\n{self.filepath}")

            hdulindex = 1 if cutout.survey == 'iphas' else 0
            header, data = hdul[hdulindex].header, hdul[hdulindex].data
            wcs = WCS(header, naxis=2)

        try:
            cutout2d = Cutout2D(data[0, 0, :, :], cutout.position, cutout.size * u.deg, wcs=wcs)
        except IndexError:
            cutout2d = Cutout2D(data, cutout.position, cutout.size * u.deg, wcs=wcs)

        self.data = cutout2d.data * 1000
        self.wcs = cutout2d.wcs
        self.header = self.wcs.to_header()


class RawCutout(CutoutService):
    """Fetch cutout data from specific local FITS image path."""

    def fetch_data(self, cutout):
        self.filepath = cutout.survey


class LocalCutout(CutoutService):
    """Fetch cutout data via local FITS images (e.g. RACS / VLASS)."""

    def fetch_data(self, cutout):

        fields = find_fields(cutout.position, cutout.survey)

        if fields.empty:
            raise FITSException(f"No fields located at {cutout.position.ra:.2f}, {cutout.position.dec:.2f}")

        self.closest = fields.sort_values('dist_field_centre').iloc[0]
        self.filepath = self.closest[f'{cutout.stokes}_path']

    def fetch_sources(self, cutout):
        """Locate target and neighbouring selavy components within positional uncertainty and FoV."""

        # VLASS source catalogues not yet available
        if cutout.survey == 'vlass':
            return

        try:
            selavy = SelavyCatalogue.from_params(
                epoch=cutout.survey,
                field=self.closest.field,
                stokes=cutout.stokes
            )
            components = selavy.cone_search(cutout.position, 0.5 * cutout.size * u.deg)
        except FileNotFoundError:
            logger.warning(f"Selavy files not found for {cutout.survey} Stokes {cutout.stokes}, disabling source ellipses.")
            return

        if components.empty:
            logger.warning('No nearby components found.')
            return

        self._find_neighbours(cutout, components)


class SkymapperCutout(CutoutService):
    """Fetch cutout from Skymapper API."""

    def fetch_data(self, cutout):

        self._set_cache_path(cutout)

        # Query Skymapper cutout service for links to cutout in each band
        route = 'http://api.skymapper.nci.org.au/aus/siap/dr2/'
        params = 'query?POS={:.5f},{:.5f}&SIZE={:.3f}&BAND=all&RESPONSEFORMAT=CSV&VERB=3&INTERSECT=covers'
        sm_query_link = route + params

        table = requests.get(sm_query_link.format(cutout.ra, cutout.dec, cutout.size))
        links = pd.read_csv(io.StringIO(table.text))

        # Check cutouts exist at position
        impos = f'{cutout.ra:.2f}, {cutout.dec:.2f}'
        assert len(links) > 0, f'No Skymapper image at {impos}'
        assert 'Time-out' not in links.iloc[0], f'Skymapper Gateway Time-out for image at {impos}'

        # Select image link for requested band
        link = links[links.band == cutout.band].iloc[0].get_image
        img = requests.get(link)

        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                f.write(img.content)


class PanSTARRSCutout(CutoutService):
    """Fetch cutout data via PanSTARRS DR2 API."""

    def fetch_data(self, cutout):

        self._set_cache_path(cutout)

        # Query PanSTARRS cutout service for links to cutout in each band
        pixsize = int(cutout.size * 120 * 120)
        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = f"{service}?ra={cutout.ra}&dec={cutout.dec}&size={pixsize}&format=fits&filters=grizy"
        table = table2df(Table.read(url, format='ascii'))

        # Check cutouts exist at position
        msg = f"No PanSTARRS1 image at {cutout.ra:.2f}, {cutout.dec:.2f}"
        assert not table.empty, msg

        # Select image link for requested band
        urlbase = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
                   f"?ra={cutout.ra}&dec={cutout.dec}&size={pixsize}&format=fits&red=")
        link = urlbase + table[table['filter'] == cutout.band].iloc[0].filename
        img = requests.get(link, allow_redirects=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                f.write(img.content)


class DECamCutout(CutoutService):
    """Fetch cutout data via DECam LS API."""
    
    def fetch_data(self, cutout):

        pixsize = int(cutout.pixsize * 3600 / 0.262)
        if pixsize > 512:
            pixsize = 512
            max_size = pixsize * 0.262 / 3600
            logger.warning(f"Using maximum DECam LS cutout pixsize of {max_size:.3f} deg")

        link = f"http://legacysurvey.org/viewer/fits-cutout?ra={cutout.ra}&dec={cutout.dec}"
        link += f"&pixsize={pixsize}&layer=dr8&pixscale=0.262&bands={cutout.band}"
        img = requests.get(link)

        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                f.write(img.content)


class IPHASCutout(CutoutService):
    """Fetch cutout data via IPHAS API using Vizier query."""

    def fetch_data(self, cutout):

        self._set_cache_path(cutout)
        
        # Query Vizier for observations at position
        v = Vizier(columns=['_r', '_RAJ2000', '_DEJ2000', 'rDetectionID', 'fieldGrade', '*'])
        cat = SURVEYS.loc[cutout.survey]['vizier']

        table = v.query_region(cutout.position, radius=cutout.size * u.deg, catalog=cat)
        table = table2df(table[0]).sort_values('_r')

        # Check that iPHaS image exists at position
        if table.empty:
            raise FITSException(
                f"No IPHAS image at {cutout.position.ra:.2f}, {cutout.position.dec:.2f}")

        # Remove poor quality observations
        table = table[(table.rDetectionID != '') & (table.fieldGrade != 'D')]
        print(table)

        # Build image link from Detection ID
        det_id = table.iloc[1]['rDetectionID']
        run = det_id.split('-')[0]
        ccd = det_id.split('-')[1]
        link = f"http://www.iphas.org/data/images/r{run[:3]}/r{run}-{ccd}.fits.fz"

        img = requests.get(link)

        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                f.write(img.content)


class MWATSCutout(CutoutService):

    def __init__(self, cutout):
        self.plot_source = False
        self.plot_neighbours = False

        mwats_path = SURVEYS.loc[cutout.survey][f'selavy_path_i']
        self.mwats = SelavyCatalogue.from_aegean(mwats_path + 'mwats_raw.parq')

        self.fetch_sources(cutout)
        self.fetch_data(cutout)

        self._make_cutout(cutout)
    
    def fetch_data(self, cutout):

        components = self.mwats.cone_search(cutout.position, 1 * u.degree)

        if components.empty:
            raise FITSException(f"No MWATS images at {cutout.ra:.4f} {cutout.dec:.4f}")

        nearest = components.sort_values('d2d', ascending=True).iloc[0]

        mwats_path = SURVEYS.loc[cutout.survey][f'image_path_{cutout.stokes}']
        self.filepath = mwats_path + nearest.image.replace('_I', f'_{cutout.stokes.upper()}')

    def fetch_sources(self, cutout):

        components = self.mwats.cone_search(cutout.position, 0.5 * cutout.size * u.degree)

        self._find_neighbours(cutout, components)


class SkyviewCutout(CutoutService):
    """Fetch cutout data via SkyView API."""

    def fetch_data(self, cutout):

        self._set_cache_path(cutout)
        sv = SkyView()

        if not os.path.exists(self.filepath):
            skyview_key = SURVEYS.loc[cutout.survey].skyview
            try:
                hdul = sv.get_images(
                    position=cutout.position,
                    survey=[skyview_key],
                    radius=cutout.size * u.deg,
                    show_progress=False
                )[0][0]
            except IndexError:
                raise FITSException('Skyview image list returned empty.')
            except ValueError:
                raise FITSException(f'{cutout.survey} is not a valid SkyView survey.')
            except HTTPError:
                raise FITSException('No response from Skyview server.')

            with open(self.filepath, 'wb') as f:
                hdul.writeto(f)

