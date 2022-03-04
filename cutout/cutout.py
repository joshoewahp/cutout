#!/usr/bin/env python
"""
Cutout module documentation
"""

import io
import os
import re
import sys
import glob
import time
import logging
import requests
import warnings
import matplotlib
import configparser
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astropy.coordinates import SkyCoord, Distance, Angle, SkyOffsetFrame, ICRS
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.time import Time
from astropy.table import Table
from astropy.nddata import Cutout2D
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from astropy.wcs.utils import proj_plane_pixel_scales, celestial_frame_to_wcs, wcs_to_celestial_frame
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.visualization import ZScaleInterval, PowerDistStretch, ImageNormalize
from astropy.nddata import Cutout2D
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
from pathlib import Path
from urllib.error import HTTPError

from astroutils.io import table2df, find_fields, FITSException, get_surveys, get_config

warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

config = get_config()
aux_path = Path(config['DATA']['aux_path'])
vlass_path = Path(config['DATA']['vlass_path'])
cutout_cache = aux_path / 'cutouts'

SURVEYS = get_surveys()
SURVEYS.set_index('survey', inplace=True)

Simbad.add_votable_fields('otype', 'ra(d)', 'dec(d)', 'parallax',
                          'pmdec', 'pmra', 'distance',
                          'sptype', 'distance_result')

SEL2AEG_COLS = {
    'island': 'island_id',
    'source': 'component_id',
    'local_rms': 'rms_image',
    'ra': 'ra_deg_cont',
    'dec': 'dec_deg_cont',
    'err_ra': 'ra_deg_cont_err',
    'err_dec': 'dec_deg_cont_err',
    'peak_flux': 'flux_peak',
    'err_peak_flux': 'flux_peak_err',
    'int_flux': 'flux_int',
    'err_int_flux': 'flux_int_err',
    'a': 'maj_axis',
    'err_a': 'maj_axis_err',
    'b': 'min_axis',
    'err_b': 'min_axis_err',
    'pa': 'pos_ang',
    'err_pa': 'pos_ang_err',
}

logger = logging.getLogger(__name__)


class Cutout:

    def __init__(self, survey, position, radius, **kwargs):
        self.survey = survey
        self.position = position
        self.ra = self.position.ra.to_value(u.deg)
        self.dec = self.position.dec.to_value(u.deg)
        self.radius = radius
        self.basesurvey = kwargs.get('basesurvey', 'racs-low')
        self.stokes = kwargs.get('stokes', 'i')
        self.psf = kwargs.get('psf')
        self.cmap = kwargs.get('cmap', 'gray_r' if self.stokes == 'i' else 'coolwarm')
        self.color = 'k' if self.cmap == 'hot' else 'black'
        self.band = kwargs.get('band', 'g')
        self.rotate_axes = False

        self.kwargs = kwargs

        try:
            self._get_cutout()
        except Exception as e:
            msg = f"{survey} failed: {e}"
            raise FITSException(msg)
        finally:
            if not any(c in self.survey for c in ['racs', 'vast', 'swagx']):
                self.plot_sources = False
                self.plot_neighbours = False

    def __repr__(self):
        return f"Cutout({self.survey}, ra={self.ra:.2f}, dec={self.dec:.2f})"

    def _get_source(self):
        try:
            pattern = re.compile(r'\S*(\d{4}[+-]\d{2}[AB])\S*')
            sbidpattern = re.compile(r'\S*(SB\d{4,5})\S*')
            selpath = SURVEYS.loc[self.survey][f'selavy_path_{self.stokes}']

            # Check if selpath is NaN (happens when using ContourCutout)
            # Should make this less hacky later
            if selpath != selpath:
                selpath = SURVEYS.loc[self.contours][f'selavy_path_{self.stokes}']

            selfiles = glob.glob(f'{selpath}/*components.txt')

            # If no components available, try raw selavy file
            if len(selfiles) == 0:
                selfiles = glob.glob(f'{selpath}/*.txt')
                if len(selfiles) == 0:
                    selfiles = glob.glob(f'{selpath}/*.csv')

            sel = [s for s in selfiles if pattern.sub(r'\1', self.filepath) in s]

            if len(sel) == 0:
                sel = [s for s in selfiles if sbidpattern.sub(r'\1', self.filepath) in s]
                aegean = True
            else:
                aegean = False

            if aegean:
                if len(sel) > 1:
                    df = pd.concat([pd.read_csv(s) for s in sel])
                else:
                    df = pd.read_csv(sel[0])
                df.rename(columns={old: new for old, new in SEL2AEG_COLS.items()}, inplace=True)
            else:
                if len(sel) > 1:
                    df = pd.concat([pd.read_fwf(s, skiprows=[1, ]) for s in sel])
                else:
                    df = pd.read_fwf(sel[0], skiprows=[1, ])

            # If using raw selavy, check that header has been properly removed
            if df.shape[1] < 5:
                logger.warning(f"No components txt, reading raw selavy at {sel[0]} instead.")
                df = pd.read_fwf(sel[0], skiprows=44, comment=r'#')
                df = df.iloc[2:].reset_index(drop=True)
                df.rename(columns={'RA.1': 'ra_deg_cont',
                                   'DEC.1': 'dec_deg_cont',
                                   'F_peak': 'flux_peak',
                                   'MAJ': 'maj_axis',
                                   'MIN': 'min_axis',
                                   'PA': 'pos_ang',
                                   }, inplace=True)
                floatcols = ['ra_deg_cont', 'dec_deg_cont', 'flux_peak', 'maj_axis', 'min_axis', 'pos_ang']
                df[floatcols] = df[floatcols].astype(np.float64)
                df['rms_image'] = 0.25
                df['flux_peak'] *= 1000

            coords = SkyCoord(df.ra_deg_cont, df.dec_deg_cont, unit=u.deg)
            d2d = self.position.separation(coords)
            df['d2d'] = d2d.arcsec
            sources = df.iloc[np.where(d2d.deg < 0.5 * self.radius)[0]]
            sources = sources.sort_values('d2d', ascending=True)

            if any(sources.d2d < self.pos_err / 3600):
                self.source = sources.iloc[0]
                self.neighbours = sources.iloc[1:]
                self.plot_sources = True
            else:
                self.source = None
                self.neighbours = sources
                self.plot_sources = False

            self.plot_neighbours = self.kwargs.get('neighbours', True)

            logger.debug(f'Source: \n {self.source}')
            logger.debug(f'Coords: {coords[np.argmin(d2d)].to_string(style="hmsdms")}')
            if len(self.neighbours) > 0:
                nn = self.neighbours.iloc[0]
                logger.debug(f'Nearest neighbour coords: \n {nn.ra_deg_cont, nn.dec_deg_cont}')
                neighbour_view = self.neighbours[['ra_deg_cont', 'dec_deg_cont', 'maj_axis', 'min_axis',
                                                  'flux_peak', 'rms_image', 'd2d']]
                logger.debug(f'Nearest 5 Neighbours \n {neighbour_view.head()}')

        except IndexError as e:
            logger.debug(e)
            self.plot_sources = False
            self.plot_neighbours = False
            logger.warning('No nearby sources found.')

    def _get_cutout(self):

        if self.kwargs.get('data'):
            c = self.kwargs.get('data')
            self.mjd = c.mjd
            self.data = c.data
            self.wcs = c.wcs
            self.header = c.header
            self.position = c.position

            return

        if not os.path.exists(cutout_cache / self.survey):
            msg = f"{cutout_cache}{self.survey} cutout directory does not exist, creating."
            logger.info(msg)
            os.makedirs(cutout_cache / self.survey)

        if os.path.isfile(self.survey):
            self._get_local_cutout()
        elif any(c in self.survey for c in ['racs', 'vast', 'vlass', 'gw', 'swagx']):
            self._get_local_cutout()
        elif self.survey == 'skymapper':
            self._get_skymapper_cutout()
        elif self.survey == 'panstarrs':
            self._get_panstarrs_cutout()
        elif self.survey == 'decam':
            self._get_decam_cutout()
        elif self.survey == 'iphas':
            self._get_iphas_cutout()
        elif self.survey == 'mwats':
            self._get_mwats_cutout()
        else:
            self._get_skyview_cutout()

        return

    def _get_local_cutout(self):
        """Fetch cutout data via local FITS images (e.g. RACS / VLASS)."""

        fields = self._find_image()
        assert len(fields) > 0, f"No fields located at {self.position.ra:.2f}, {self.position.dec:.2f}"

        closest = fields[fields.dist_field_centre == fields.dist_field_centre.min()].iloc[0]
        image_path = SURVEYS.loc[self.survey][f'image_path_{self.stokes}']

        if self.survey == 'vlass':
            filepath = f'{closest.epoch}/{closest.tile}/{closest.image}/{closest.filename}'
            image_path = vlass_path
        elif self.survey == 'racs-low':
            filepath = f'RACS_test4_1.05_{closest.field}A.fits'
        elif self.survey == 'racs-mid':
            filepath = f'image.{self.stokes.lower()}.VAST_{closest.field}.{closest.sbid}.cont.taylor.0.restored.conv.fits'
        elif 'vast' in self.survey:
            pattern = re.compile(r'vastp(\d+x*)')
            epoch = pattern.sub(r'\1', self.survey)
            zeropad = '0' if (len(epoch) == 1 or len(epoch) == 2 and epoch[-1] == 'x') else ''
            filepath = f'VAST_{closest.field}A.EPOCH{zeropad}{epoch}.{self.stokes.upper()}.fits'
        elif 'swagx' in self.survey:
            filepath = f'image.{self.stokes.lower()}.{closest.field}.cont.taylor.0.restored.fits'
        else:
            filepath = f'*{closest.field}*0.restored.fits'

        try:
            self.filepath = glob.glob(image_path + filepath)[0]
        except IndexError:
            msg = f'Could not match {self.survey} image filepath: \n{image_path + filepath}'
            raise FITSException(msg)

        with fits.open(self.filepath) as hdul:
            logger.debug(f"Making cutout from FITS image located at {self.filepath}")
            header, data = hdul[0].header, hdul[0].data
            wcs = WCS(header, naxis=2)
            try:
                self.mjd = Time(header['DATE-OBS']).mjd
            except:
                self.mjd = Time(header['DATE']).mjd

            try:
                cutout = Cutout2D(data[0, 0, :, :], self.position, self.radius * u.deg, wcs=wcs)
            except IndexError:
                cutout = Cutout2D(data, self.position, self.radius * u.deg, wcs=wcs)

            self.data = cutout.data * 1000
            self.wcs = cutout.wcs
            self.header = self.wcs.to_header()


            cdelt1, cdelt2 = proj_plane_pixel_scales(cutout.wcs)
            self.header.remove("PC1_1", ignore_missing=True)
            self.header.remove("PC2_2", ignore_missing=True)
            self.header.update(
                CDELT1=-cdelt1,
                CDELT2=cdelt2,
                BMAJ=header["BMAJ"],
                BMIN=header["BMIN"],
                BPA=header["BPA"]
            )

        if any(s in self.survey for s in ['racs', 'vast', 'swagx']):
            self.pos_err = SURVEYS.loc[self.basesurvey].pos_err
            self._get_source()
        else:
            # Probably using vlass, yet to include aegean catalogs
            self.plot_sources = False
            self.plot_neighbours = False

    def _get_mwats_cutout(self):
        mwats = pd.read_parquet('/import/ada1/jpri6587/data/mwats_raw.parq')
        mwats = mwats[(mwats.ra > self.ra - 1) & (mwats.ra < self.ra + 1) &
                      (mwats.dec > self.dec - 1) & (mwats.dec < self.dec + 1)]
        assert len(mwats) > 0, "No MWATS sources in simple position filter. Check RA wrapping."

        coords = SkyCoord(ra=mwats.ra, dec=mwats.dec, unit=u.deg)
        mwats['d2d'] = coords.separation(self.position).arcsec

        nearest = mwats.sort_values('d2d', ascending=True).iloc[0]
        assert nearest.d2d < 15, "No MWATS sources within 15 arcsec."

        self.filepath = '/import/extreme1/mebell/dockerized-pipeline/vast-pipeline/DATA/mwats/' + nearest.image

        with fits.open(self.filepath) as hdul:
            self.header, data = hdul[0].header, hdul[0].data
            wcs = WCS(self.header, naxis=2)
            self.mjd = Time(self.header['DATE-OBS']).mjd

            try:
                cutout = Cutout2D(data[0, 0, :, :], self.position, self.radius * u.deg, wcs=wcs)
            except IndexError:
                cutout = Cutout2D(data, self.position, self.radius * u.deg, wcs=wcs)
            self.data = cutout.data * 1000
            self.wcs = cutout.wcs

    def _get_panstarrs_cutout(self):
        """Fetch cutout data via PanSTARRS DR2 API."""
        path = cutout_cache / 'panstarrs/{}_{}arcmin_{}_{}.fits'.format(self.band,
                                                                        '{:.4f}',
                                                                        '{:.4f}',
                                                                        '{:.4f}',)
        imgpath = str(path).format(self.radius * 60, self.ra, self.dec)
        if not os.path.exists(imgpath):
            pixelrad = int(self.radius * 120 * 120)
            service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
            url = (f"{service}?ra={self.ra}&dec={self.dec}&size={pixelrad}&format=fits"
                   f"&filters=grizy")
            table = Table.read(url, format='ascii')

            msg = f"No PS1 image at {self.position.ra:.2f}, {self.position.dec:.2f}"
            assert len(table) > 0, msg

            urlbase = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                       f"ra={self.ra}&dec={self.dec}&size={pixelrad}&format=fits&red=")

            flist = ["yzirg".find(x) for x in table['filter']]
            table = table[np.argsort(flist)]

            for row in table:
                filt = row['filter']
                url = urlbase + row['filename']
                path = cutout_cache / 'panstarrs/{}_{}arcmin_{}_{}.fits'.format(filt,
                                                                                '{:.4f}',
                                                                                '{:.4f}',
                                                                                '{:.4f}',)
                path = str(path).format(self.radius * 60, self.ra, self.dec)

                img = requests.get(url, allow_redirects=True)

                if not os.path.exists(path):
                    with open(path, 'wb') as f:
                        f.write(img.content)

        with fits.open(imgpath) as hdul:
            self.header, self.data = hdul[0].header, hdul[0].data
            self.wcs = WCS(self.header, naxis=2)
            self.mjd = self.header['MJD-OBS']

    def _get_iphas_cutout(self):
        """Fetch cutout data via iPHAS API using Vizier query."""

        path = cutout_cache / self.survey / 'dr2_{:.3f}arcmin_{:.3f}_{:.3f}.fits'

        v = Vizier(columns=['_r', '_RAJ2000', '_DEJ2000', 'rDetectionID', '*'])
        cat = SURVEYS.loc[self.survey]['vizier']

        try:
            table = v.query_region(self.position, radius=self.radius * u.deg, catalog=cat)
            table = table2df(table[0])
            det_id = table.sort_values('_r').iloc[0]['rDetectionID']
        except IndexError:
            raise FITSException(
                f"No iPHAS image at {self.position.ra:.2f}, {self.position.dec:.2f}")
        run = det_id.split('-')[0]
        ccd = det_id.split('-')[1]
        link = f"http://www.iphas.org/data/images/r{run[:3]}/r{run}-{ccd}.fits.fz"

        img = requests.get(link)
        path = str(path).format(self.radius * 60, self.ra, self.dec)

        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(img.content)

        with fits.open(path) as hdul:
            self.header, data = hdul[1].header, hdul[1].data
            wcs = WCS(self.header, naxis=2)
            self.mjd = self.header['MJD-OBS']
            cutout = Cutout2D(data, self.position, self.radius * u.deg, wcs=wcs)
            theta = np.arctan2(self.header['CD1_1'], self.header['CD1_2']) * 180 / np.pi

            if abs(theta) - 90 > 45:
                logger.warning("Image data seems to be rotated by 90 degrees.")
                self.rotate_axes = True
            self.data = cutout.data
            self.wcs = cutout.wcs

    def _get_skymapper_cutout(self):
        """Fetch cutout data via Skymapper API."""

        path = cutout_cache / self.survey / 'dr2{}_jd{:.4f}_{:.4f}arcmin_{:.4f}_{:.4f}.fits'
        linka = 'http://api.skymapper.nci.org.au/aus/siap/dr2/'
        linkb = 'query?POS={:.5f},{:.5f}&SIZE={:.3f}&BAND=all&RESPONSEFORMAT=CSV'
        linkc = '&VERB=3&INTERSECT=covers'
        sm_query = linka + linkb + linkc

        link = linka + 'get_image?IMAGE={}&SIZE={}&POS={},{}&FORMAT=fits'

        table = requests.get(sm_query.format(self.ra, self.dec, self.radius))
        df = pd.read_csv(io.StringIO(table.text))
        impos = f'{self.position.ra:.2f}, {self.position.dec:.2f}'
        assert len(df) > 0, f'No Skymapper image at {impos}'
        assert 'Time-out' not in df.iloc[0], f'Skymapper Gateway Time-out for image at {impos}'

        df = df[df.band == self.band]
        self.mjd = df.iloc[0]['mjd_obs']
        link = df.iloc[0].get_image

        img = requests.get(link)

        path = str(path).format(self.band, self.mjd, self.radius * 60, self.ra, self.dec)

        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(img.content)

        with fits.open(path) as hdul:
            self.header, self.data = hdul[0].header, hdul[0].data
            self.wcs = WCS(self.header, naxis=2)

    def _get_decam_cutout(self):
        """Fetch cutout data via DECam LS API."""
        size = int(self.radius * 3600 / 0.262)
        if size > 512:
            size = 512
            maxradius = size * 0.262 / 3600
            logger.warning(f"Using maximum DECam LS cutout radius of {maxradius:.3f} deg")

        link = f"http://legacysurvey.org/viewer/fits-cutout?ra={self.ra}&dec={self.dec}"
        link += f"&size={size}&layer=dr8&pixscale=0.262&bands={self.band}"
        img = requests.get(link)

        # Fix mjd issues!!
        # This is currently a midpoint between the observing run
        # as no obs-date information is available in the header
        self.mjd = Time(2017.96, format='decimalyear').mjd

        path = cutout_cache / self.survey / 'dr8_jd{:.4f}_{:.4f}arcmin_{:.4f}_{:.4f}_{}band.fits'
        path = str(path).format(self.mjd, self.radius * 60, self.ra, self.dec, self.band)
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(img.content)

        try:
            with fits.open(path) as hdul:
                self.header, self.data = hdul[0].header, hdul[0].data
                self.wcs = WCS(self.header, naxis=2)

        except OSError:
            msg = f"DECam LS image at {self.position.ra:.2f}, {self.position.dec:.2f} is corrupt"
            raise FITSException(msg)

        msg = f"No DECam LS image at {self.position.ra:.2f}, {self.position.dec:.2f}"
        assert self.data is not None, msg

    def _get_skyview_cutout(self):
        """Fetch cutout data via SkyView API."""

        sv = SkyView()
        path = cutout_cache / self.survey / '{:.3f}arcmin_{:.3f}_{:.3f}.fits'
        path = str(path).format(self.radius * 60, self.ra, self.dec)
        progress = self.kwargs.get('progress', False)

        if not os.path.exists(path):
            skyview_key = SURVEYS.loc[self.survey].skyview
            try:
                hdul = sv.get_images(position=self.position, survey=[skyview_key],
                                     radius=self.radius * u.deg, show_progress=progress)[0][0]
            except IndexError:
                raise FITSException('Skyview image list returned empty.')
            except ValueError:
                raise FITSException(f'{self.survey} is not a valid SkyView survey.')
            except HTTPError:
                raise FITSException('No response from Skyview server.')

            with open(path, 'wb') as f:
                hdul.writeto(f)

        with fits.open(path) as hdul:
            self.header, self.data = hdul[0].header, hdul[0].data
            self.wcs = WCS(self.header, naxis=2)

            try:
                self.mjd = Time(self.header['DATE']).mjd
            except KeyError:
                try:
                    self.epoch = self.kwargs.get('epoch')
                    msg = "Could not detect epoch, PM correction disabled."
                    assert self.epoch is not None, msg
                    self.mjd = self.epoch if self.epoch > 3000 else Time(
                        self.epoch, format='decimalyear').mjd
                except AssertionError as e:
                    if self.kwargs.get('pm'):
                        logger.warning(e)
                    self.mjd = None

            self.data *= 1000

    def _find_image(self):
        return find_fields(self.position, self.survey)

    def _plot_setup(self, fig, ax):
        """Create figure and determine normalisation parameters."""

        if ax:
            self.fig = fig
            self.ax = ax
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection=self.wcs)

        if self.kwargs.get('grid', True):
            self.ax.coords.grid(color='white', alpha=0.5)

        if self.kwargs.get('title', True):
            self.ax.set_title(SURVEYS.loc[self.survey]['name'], fontdict={'fontsize': 20,
                                                                          'fontweight': 10})
        if self.kwargs.get('coords') == 'compact':
            tickcolor = 'k' if np.nanmax(np.abs(self.data)) == np.nanmax(self.data) else 'gray'
            self.ax.tick_params(axis='both', direction='in', length=5, color=tickcolor)
            lon = self.ax.coords[0]
            lat = self.ax.coords[1]
            lon.display_minor_ticks(True)
            lat.display_minor_ticks(True)
            lon.set_ticks(number=5)
            lat.set_ticks(number=5)
            self.padlevel = self.kwargs.get('ylabelpad', 5)

        else:
            if self.rotate_axes:
                lon, lat = self.ax.coords
                lon.set_ticks_position('lb')
                lon.set_ticklabel_position('lb')
                lat.set_ticks_position('lb')
                lat.set_ticklabel_position('lb')
                self.set_xlabel("Dec (J2000)")
                self.set_ylabel("RA (J2000)")
            else:
                self.set_xlabel('RA (J2000)')
                self.set_ylabel('Dec (J2000)')

        if self.kwargs.get('obfuscate', False):
            self.hide_coords()

        if self.kwargs.get('annotation'):
            color = 'white' if self.cmap == 'hot' else 'k'
            self.ax.text(0.05, 0.85, self.kwargs.get('annotation'), color=color,
                         weight='bold', transform=self.ax.transAxes)

        if self.kwargs.get('vmax') or self.kwargs.get('vmin'):
            vmin = self.kwargs.get('vmin', -2)
            vmax = self.kwargs.get('vmax', 1)
            self.norm = ImageNormalize(self.data, interval=ZScaleInterval(),
                                       vmin=vmin, vmax=vmax, clip=True)
        elif self.kwargs.get('maxnorm'):
            self.norm = ImageNormalize(self.data, interval=ZScaleInterval(),
                                       vmax=np.nanmax(self.data), clip=True)
        else:
            contrast = self.kwargs.get('contrast', 0.2)
            self.norm = ImageNormalize(self.data, interval=ZScaleInterval(contrast=contrast),
                                       clip=True)

    def _align_ylabel(self, ylabel):
        # Get coordinates of topleft pixel
        topleft = self.wcs.wcs_pix2world(0, self.data.shape[1] + 1, 1)
        dms_tick = SkyCoord(ra=topleft[0], dec=topleft[1], unit=u.deg).dec.dms

        # Round coordinates to nearest 20 arcsec in direction of cutout centre
        # This corresponds to the coordinates of the widest ytick label
        sign = dms_tick[0] // abs(dms_tick[0])
        d_str = f'{int(dms_tick[0])}'
        if len(d_str) == 1 or (len(d_str) == 2 and sign < 0):
            d_str = 's' + d_str
        m_str = f'{int(abs(dms_tick[1])):02d}'

        if sign < 0:
            s_str = f'{int(round(abs(dms_tick[2]) // 20) * 20 + 20):02d}'
        else:
            s_str = f'{int(round(abs(dms_tick[2]) // 20) * 20):02d}'
        if s_str == '60':
            s_str = '00'
            m_str = f'{int(m_str) + 1:02d}'

        # Pad axis label to offset individual ytick label character widths
        dec_str = d_str + m_str + s_str

        charlen = {'-': .65, 's': .075}
        zeropad = 0.8 + sum([charlen.get(c, 0.5) for c in dec_str])

        self.ax.set_ylabel(ylabel, labelpad=self.padlevel - zeropad)

    def _add_cornermarker(self, ra, dec):

        span = self.kwargs.get('corner_span', len(self.data) / 4)
        offset = self.kwargs.get('corner_offset', len(self.data) / 8)
        datapos = self.wcs.wcs_world2pix(ra, dec, 1)

        if any(i < 0 or i > len(self.data) or np.isnan(i) for i in datapos):
            if self.offsets:
                datapos = self.wcs.wcs_world2pix(0, 0, 1)
            else:
                msga = "RA and Dec are outside of data range, and not in offsets mode. "
                msgb = "Using pixel centre for cornermarker."
                logger.warning(msga + msgb)
                datapos = [len(self.data) / 2, len(self.data) / 2]

        x = datapos[0] - 1
        y = datapos[1] - 1

        color = 'k' if self.cmap != 'gray_r' else 'r'
        raline = Line2D(xdata=[x - offset, x - span],
                        ydata=[y, y],
                        color=color, linewidth=2,
                        path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
        decline = Line2D(xdata=[x, x],
                         ydata=[y + offset, y + span],
                         color=color, linewidth=2,
                         path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

        self.ax.add_artist(raline)
        self.ax.add_artist(decline)

    def switch_to_offsets(self):
        """Transform WCS to a frame centred on the image."""

        cdelt1, cdelt2 = proj_plane_pixel_scales(self.wcs)
        ctype = self.wcs.wcs.ctype
        crpix = self.wcs.wcs_world2pix(self.ra, self.dec, 1)

        # Create new WCS as Skymapper does weird things with CDELT
        self.wcs = WCS(naxis=2)

        # Centre pixel is offset by 1 due to array indexing convention
        # self.wcs.wcs.crpix = [(len(self.data)) / 2 + 1,
        #                       (len(self.data)) / 2 + 1]
        self.wcs.wcs.crpix = [crpix[0], crpix[1]]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.cdelt = [-cdelt1, cdelt2]
        self.wcs.wcs.ctype = ctype

        if 'radio' in dir(self):
            r_crpix = self.radio.wcs.wcs_world2pix(self.ra, self.dec, 1)
            # self.radio.wcs.wcs.crpix = [(len(self.radio.data)) / 2 + 1,
            #                             (len(self.radio.data)) / 2 + 1]
            self.radio.wcs.wcs.crpix = [r_crpix[0], r_crpix[1]]
            self.radio.wcs.wcs.crval = [0, 0]

        self.offsets = True

    def hide_coords(self, axis='both', ticks=True, labels=True):
        """Remove all coordinates and identifying information."""
        lon = self.ax.coords[0]
        lat = self.ax.coords[1]
        if axis in ['x', 'both']:
            lon.set_axislabel(' ')
            if labels:
                lon.set_ticklabel_visible(False)
            if ticks:
                lon.set_ticks_visible(False)
        if axis in ['y', 'both']:
            lat.set_axislabel(' ')
            if labels:
                lat.set_ticklabel_visible(False)
            if ticks:
                lat.set_ticks_visible(False)

    def plot(self, fig=None, ax=None):
        """Plot survey data and position overlay."""
        self.sign = self.kwargs.get('sign', 1)
        self._plot_setup(fig, ax)
        self.data *= self.sign
        absmax = max(self.data.max(), self.data.min(), key=abs)
        logger.debug(f"Max flux in cutout: {absmax:.2f} mJy.")
        rms = np.sqrt(np.mean(np.square(self.data)))
        logger.debug(f"RMS flux in cutout: {rms:.2f} mJy.")

        assert (sum((~np.isnan(self.data).flatten())) > 0 and sum(self.data.flatten()) != 0), \
            f"No data in {self.survey}"

        if self.stokes == 'v':
            self.cmap = plt.cm.coolwarm

        self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        if self.kwargs.get('bar', True):
            try:
                self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)
            except UnboundLocalError:
                logger.error("Colorbar failed. Upgrade to recent version of astropy ")

        if self.psf:
            try:
                self.bmaj = self.header['BMAJ'] * 3600
                self.bmin = self.header['BMIN'] * 3600
                self.bpa = self.header['BPA']
            except KeyError:
                logger.warning('Header did not contain PSF information.')
                try:
                    self.bmaj = self.psf[0]
                    self.bmin = self.psf[1]
                    self.bpa = 0
                    logger.warning('Using supplied BMAJ/BMin. Assuming BPA=0')
                except ValueError:
                    logger.error('No PSF information supplied.')

            try:
                cdelt = self.header['CDELT1']
            except KeyError:
                cdelt = self.header['CD1_1']

            if self.kwargs.get('beamsquare'):
                frame = True
                facecolor = 'k'
                edgecolor = 'k'
            else:
                frame = False
                facecolor = 'white'
                edgecolor = 'k'
            x = self.bmin / abs(cdelt) / 3600
            y = self.bmaj / abs(cdelt) / 3600
            self.beam = AnchoredEllipse(self.ax.transData, width=x, height=y,
                                        angle=self.bpa, loc=3, pad=0.5, borderpad=0.4,
                                        frameon=frame)
            self.beam.ellipse.set(facecolor=facecolor, edgecolor=edgecolor)
            self.ax.add_artist(self.beam)

        if self.kwargs.get('source', True):
            if self.plot_sources:  # refactor this to plot_source_ellipse
                if self.kwargs.get('corner'):
                        self._add_cornermarker(self.source.ra_deg_cont, self.source.dec_deg_cont)
                else:
                    self.sourcepos = Ellipse((self.source.ra_deg_cont,
                                            self.source.dec_deg_cont),
                                            self.source.min_axis / 3600,
                                            self.source.maj_axis / 3600,
                                            -self.source.pos_ang,
                                            facecolor='none', edgecolor='r',
                                            ls=':', lw=2,
                                            transform=self.ax.get_transform('world'))
                    self.ax.add_patch(self.sourcepos)

        else:
            if self.kwargs.get('corner'):
                    self._add_cornermarker(self.ra, self.dec)
            else:
                self.bmin = 15
                self.bmaj = 15
                self.bpa = 0
                overlay = SphericalCircle((self.ra * u.deg, self.dec * u.deg),
                                          self.bmaj * u.arcsec,
                                          edgecolor='r',
                                          linewidth=2,
                                          facecolor='none',
                                          transform=self.ax.get_transform('world'))
                self.ax.add_artist(overlay)
        if self.plot_sources:
            self.sourcepos = Ellipse((self.source.ra_deg_cont,
                                      self.source.dec_deg_cont),
                                     self.source.min_axis / 3600,
                                     self.source.maj_axis / 3600,
                                     -self.source.pos_ang,
                                     facecolor='none', edgecolor='r',
                                     ls=':', lw=2,
                                     transform=self.ax.get_transform('world'))
            self.ax.add_patch(self.sourcepos)
        if self.plot_neighbours:
            for idx, neighbour in self.neighbours.iterrows():
                n = Ellipse((neighbour.ra_deg_cont, neighbour.dec_deg_cont),
                            neighbour.min_axis / 3600,
                            neighbour.maj_axis / 3600,
                            -neighbour.pos_ang,
                            facecolor='none', edgecolor='c', ls=':', lw=2,
                            transform=self.ax.get_transform('world'))
                self.ax.add_patch(n)

    def save(self, path, fmt='png'):
        """Save figure with tight bounding box."""
        self.fig.savefig(path, format=fmt, bbox_inches='tight')

    def savefits(self, path):
        """Export FITS cutout to path"""
        header = self.wcs.to_header()
        hdu = fits.PrimaryHDU(data=self.data, header=header)
        hdu.writeto(path)

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel)

    def set_ylabel(self, ylabel, align=False):
        if align:
            self._align_ylabel(ylabel)
        else:
            self.ax.set_ylabel(ylabel)


class ContourCutout(Cutout):

    def __init__(self, survey, position, radius, **kwargs):

        self.contours = kwargs.get('contours', 'racs-low')
        self.clabels = kwargs.get('clabels', False)
        self.bar = kwargs.get('bar', False)
        try:
            data = kwargs.pop('data')
        except KeyError:
            data = None
        self.radio = Cutout(self.contours, position, radius, **kwargs)
        self.cmjd = self.radio.mjd

        self.correct_pm = kwargs.get('pm')
        super().__init__(survey, position, radius, data=data, **kwargs)

        if self.correct_pm:
            self._correct_proper_motion()

        # Ensure contour array shape is not larger than base array shape
        if self.data.shape[0] < self.radio.data.shape[0]:
            logger.warning(
                "Contour data array larger than base, resizing to avoid extra whitespace.")
            logger.error("This is currently broken, contours not displaying.")
            # self.radio.data = reproject_interp((self.radio.data, self.radio.wcs), self.wcs,
            #                                    shape_out=self.data.shape, return_footprint=False)

    def align_image_to_contours(self):
        if not self.correct_pm:
            self._correct_proper_motion()
            self.correct_pm = False

        try:
            pmra = self.oldpos.pm_ra
        except AttributeError as e:
            pmra = self.oldpos.pm_ra_cosdec
            logger.error(e)
            logger.error("Astropy for some reason can't decide on calling this pm_ra or pm_ra_cosdec")


        orig_pos = SkyCoord(
            ra=self.wcs.wcs.crval[0] * u.deg,
            dec=self.wcs.wcs.crval[1] * u.deg,
            frame='icrs',
            distance=self.oldpos.distance,
            pm_ra_cosdec=pmra,
            pm_dec=self.oldpos.pm_dec,
            obstime=Time(self.mjd, format='mjd'))
        newpos = orig_pos.apply_space_motion(Time(self.cmjd, format='mjd'))
        self.wcs.wcs.crval = [newpos.ra.deg, newpos.dec.deg]

    def _correct_proper_motion(self, invert=False):
        """Check SIMBAD for nearby star or pulsar and plot a cross at corrected coordinates"""

        simbad = Simbad.query_region(self.position, radius=180 * u.arcsec)
        if self.kwargs.get('epoch'):
            self.epoch = self.kwargs.get('epoch', self.mjd)
        self.epochtype = 'MJD' if self.epoch > 3e3 else 'decimalyear'
        self.mjd = Time(self.epoch, format=self.epochtype.lower()).mjd

        assert self.mjd is not None, "Date could not be inferred from header, supply with --epoch."

        if simbad is not None:
            simbad = table2df(simbad)
            pm_types = ['*', '**', 'PM*', 'EB*', 'Star', 'PSR', 'Pulsar', 'Flare*']
            simbad = simbad[(simbad['OTYPE'].isin(pm_types)) | (simbad['SP_TYPE'].str.len() > 0)]

            if len(simbad) == 0:
                logger.warning("No high proper-motion objects within 180 arcsec.")
                self.correct_pm = False
                self.pm_coord = SkyCoord(
                    ra=self.position.ra,
                    dec=self.position.dec,
                    frame='icrs',
                    distance=Distance(parallax=1000000 * u.mas),
                    pm_ra_cosdec=0 * u.mas / u.yr,
                    pm_dec=0 * u.mas / u.yr,
                    obstime='J2000')
                self.oldpos = self.pm_coord

                return

            # Treat non-existant proper motion parameters as extremely distance objects
            simbad['PMRA'].fillna(0, inplace=True)
            simbad['PMDEC'].fillna(0, inplace=True)
            simbad['PLX_VALUE'].fillna(1000000, inplace=True)

            obstime = 'J2000'
            proptime = Time(self.cmjd, format='mjd')
            pmra = simbad['PMRA'].values * u.mas / u.yr
            pmdec = simbad['PMDEC'].values * u.mas / u.yr

            dist = Distance(parallax=simbad['PLX_VALUE'].values * u.mas)

            simbad['j2000pos'] = SkyCoord(
                ra=simbad['RA_d'].values * u.deg,
                dec=simbad['DEC_d'].values * u.deg,
                frame='icrs',
                distance=dist,
                pm_ra_cosdec=pmra,
                pm_dec=pmdec,
                obstime=obstime)

            datapos = simbad.j2000pos.apply(
                lambda x: x.apply_space_motion(Time(self.mjd, format='mjd')))
            newpos = simbad.j2000pos.apply(
                lambda x: x.apply_space_motion(proptime))

            simbad['PM Corrected Separation (arcsec)'] = np.round(newpos.apply(
                lambda x: x.separation(self.position).arcsec), 3)
            simbad = simbad[['MAIN_ID', 'OTYPE', 'SP_TYPE', 'DISTANCE_RESULT',
                             'PM Corrected Separation (arcsec)']].copy()
            simbad = simbad.rename(columns={'MAIN_ID': 'Object', 'OTYPE': 'Type',
                                            'DISTANCE_RESULT': 'Separation (arcsec)',
                                            'SP_TYPE': 'Spectral Type'})

            logger.info(f'SIMBAD results:\n {simbad}')
            self.simbad = simbad.sort_values('PM Corrected Separation (arcsec)').head()

            nearest = self.simbad['PM Corrected Separation (arcsec)'].idxmin()
                
            self.oldpos = datapos[nearest]
            self.pm_coord = newpos[nearest]

            near_object = self.simbad.loc[nearest].Object
            msg = f'Proper motion corrected {near_object} to <{self.pm_coord.ra}, {self.pm_coord.dec}>'
            logger.info(msg)

            missing = simbad[simbad['PM Corrected Separation (arcsec)'].isna()]
            if self.simbad['PM Corrected Separation (arcsec)'].min() > 15:
                logger.warning("No PM corrected objects within 15 arcsec")
                self.correct_pm = False
            if len(missing) > 0:
                msg = f"Some objects missing PM data, and may be closer matches \n {missing}"
                logger.warning(msg)

        else:
            self.correct_pm = False
            self.pm_coord = SkyCoord(
                    ra=self.position.ra,
                    dec=self.position.dec,
                    frame='icrs',
                    distance=Distance(parallax=1000000 * u.mas),
                    pm_ra_cosdec=0 * u.mas / u.yr,
                    pm_dec=0 * u.mas / u.yr,
                    obstime='J2000')
            self.oldpos = self.pm_coord

    def plot(self, fig=None, ax=None):
        self.sign = self.kwargs.get('sign', 1)
        self._plot_setup(fig, ax)

        assert (sum((~np.isnan(self.data).flatten())) > 0 and sum(self.data.flatten()) != 0), \
            f"No data in {self.survey}"

        if self.survey == 'swift_xrtcnt':
            self.im = self.ax.imshow(self.data, cmap=self.cmap)
        else:
            self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        if self.kwargs.get('bar', False):
            try:
                self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)
            except UnboundLocalError:
                logger.error("Colorbar failed. Upgrade to recent version of astropy ")

        # Plot radio contours
        self.radio.data *= self.sign
        self.peak = np.nanmax(self.radio.data)
        self.radiorms = np.sqrt(np.mean(np.square(self.radio.data)))

        if self.kwargs.get('rmslevels'):
            self.levels = [self.radiorms * x for x in [3, 6]]
        elif self.kwargs.get('peaklevels'):
            midx = int(self.radio.data.shape[0] / 2)
            midy = int(self.radio.data.shape[1] / 2)
            peak = self.radio.data[midx, midy]
            # self.levels = [peak * x for x in [.35, .6, .85]]
            self.levels = np.logspace(np.log10(0.3 * peak), np.log10(0.9 * peak), 3)
        else:
            self.levels = [self.peak * x for x in [.3, .6, .9]]

        contourwidth = self.kwargs.get('contourwidth', 3)
        color = 'blue' if self.cmap == 'hot' else 'orange'

        self.cs = self.ax.contour(self.radio.data,
                                  transform=self.ax.get_transform(self.radio.wcs),
                                  levels=self.levels, colors=color,
                                  linewidths=contourwidth)
        if self.clabels:
            self.ax.clabel(self.cs, fontsize=10, fmt='%1.1f mJy')
        if self.bar:
            self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)

        if self.survey == 'panstarrs' and self.kwargs.get('title', True):
            self.ax.set_title(f"{SURVEYS.loc[self.survey]['name']} ({self.band}-band)")

        # Plot PM corrected location
        if self.correct_pm:
            name = self.simbad.iloc[0]["Object"]
            oldcoord = SkyCoord(self.oldpos.ra, self.oldpos.dec, unit=u.deg)
            newcoord = SkyCoord(self.pm_coord.ra, self.pm_coord.dec, unit=u.deg)
            oldtime = Time(self.mjd, format='mjd').decimalyear
            newtime = Time(self.cmjd, format='mjd').decimalyear
            handles, labels = [], []

            if oldcoord.separation(newcoord).arcsec < 1:
                self.ax.scatter(self.pm_coord.ra, self.pm_coord.dec, marker='x', s=200, color='r',
                                transform=self.ax.get_transform('world'),
                                    label=f'{name} position at J{newtime:.2f}')
                self.ax.scatter(self.oldpos.ra, self.oldpos.dec, marker='x', s=200, color='b',
                                transform=self.ax.get_transform('world'),
                                label=f'{name} position at J{oldtime:.2f}')
                self.ax.legend()
            else:
                dra, ddec = oldcoord.spherical_offsets_to(newcoord)
                arr_label = f'Proper motion from J{oldtime:.2f}-J{newtime:.2f}'
                self.ax.arrow(self.oldpos.ra.deg, self.oldpos.dec.deg, dra.deg, ddec.deg,
                              width=8e-5, color='r', length_includes_head=True,
                              transform=self.ax.get_transform('world'))

                handles.append(Line2D([], [], ls='none',
                                      marker=r'$\leftarrow$', markersize=10, color='r'))
                labels.append(arr_label)

                self.ax.legend(handles, labels)
