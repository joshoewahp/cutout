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
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table
from tools.logger import Logger
from tools.utils import table2df
from matplotlib.lines import Line2D
from astropy.nddata import Cutout2D
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from matplotlib.patches import Ellipse, Rectangle
from astropy.coordinates import SkyCoord, Distance, Angle
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.visualization import ZScaleInterval, PowerDistStretch, ImageNormalize

from urllib.error import HTTPError
from astropy._erfa.core import ErfaWarning
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
warnings.filterwarnings('ignore', category=ErfaWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

config = configparser.ConfigParser()
config.read('./config/config.ini')

aux_path = config['DATA']['aux_path']
vlass_path = config['DATA']['vlass_path']
cutout_cache = config['DATA']['cutout_cache']

SURVEYS = pd.read_json('./config/surveys.json')
SURVEYS.set_index('survey', inplace=True)
Simbad.add_votable_fields('otype', 'ra(d)', 'dec(d)', 'parallax',
                          'pmdec', 'pmra', 'distance',
                          'sptype', 'distance_result')


class FITSException(Exception):
    pass


class Cutout:

    def __init__(self, survey, position, radius, **kwargs):
        self.survey = survey
        self.position = position
        self.ra = self.position.ra.to_value(u.deg)
        self.dec = self.position.dec.to_value(u.deg)
        self.radius = radius
        self.basesurvey = kwargs.get('basesurvey', 'racsI')
        self.psf = kwargs.get('psf')
        self.cmap = kwargs.get('cmap', 'gray_r')
        self.color = 'k' if self.cmap == 'hot' else 'black'
        self.band = kwargs.get('band', 'g')

        level = 'DEBUG' if kwargs.get('verbose') else 'INFO'
        self.logger = Logger(__name__, kwargs.get('log'), streamlevel=level).logger
        self.logger.propagate = False

        self.kwargs = kwargs

        try:
            self._get_cutout()
        except Exception as e:
            msg = f"{survey} failed: {e}"
            raise FITSException(msg)
        finally:
            if 'racs' not in self.survey and 'vast' not in self.survey:
                self.plot_sources = False
                self.plot_neighbours = False

    def __repr__(self):
        return f"Cutout({self.survey}, ra={self.ra:.2f}, dec={self.dec:.2f})"

    def _get_source(self):
        try:
            pattern = re.compile(r'\S*(\d{4}[+-]\d{2}[AB])\S*')
            selpath = SURVEYS.loc[self.survey]['selavy']
            sel = glob.glob(f'{selpath}/*components.txt')
            sel = [s for s in sel if pattern.sub(r'\1', self.filepath) in s]

            if len(sel) > 1:
                df = pd.concat([pd.read_fwf(s, skiprows=[1, ]) for s in sel])
            else:
                df = pd.read_fwf(sel[0], skiprows=[1, ])
            coords = SkyCoord(df.ra_deg_cont, df.dec_deg_cont, unit=u.deg)
            d2d = self.position.separation(coords)
            df['d2d'] = d2d
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

            self.logger.debug(f'Source: \n {self.source}')
            if len(self.neighbours) > 0:
                nn = self.neighbours.iloc[0]
                self.logger.debug(f'Nearest neighbour coords: \n {nn.ra_deg_cont, nn.dec_deg_cont}')
                self.logger.debug(f'Nearest 5 Neighbours \n {self.neighbours.head()}')

        except IndexError:
            self.plot_sources = False
            self.plot_neighbours = False
            self.logger.warning('No nearby sources found.')

    def _get_cutout(self):

        if not os.path.exists(cutout_cache + self.survey):
            msg = f"{cutout_cache}{self.survey} cutout directory does not exist, creating."
            self.logger.info(msg)
            os.makedirs(cutout_cache + self.survey)

        if os.path.isfile(self.survey):
            self._get_local_cutout()
        elif 'racs' in self.survey or 'vast' in self.survey or 'vlass' in self.survey:
            self._get_local_cutout()
        elif self.survey == 'skymapper':
            self._get_skymapper_cutout()
        elif self.survey == 'panstarrs':
            self._get_panstarrs_cutout()
        elif self.survey == 'decam':
            self._get_decam_cutout()
        else:
            self._get_skyview_cutout()

    def _get_local_cutout(self):
        """Fetch cutout data via local FITS images (e.g. RACS / VLASS)."""

        fields = self._find_image()
        assert len(
            fields) > 0, f"No fields located at {self.position.ra:.2f}, {self.position.dec:.2f}"
        closest = fields[fields.dist_field_centre == fields.dist_field_centre.min()].iloc[0]
        image_path = SURVEYS.loc[self.survey]['images']

        if self.survey == 'vlass':
            filepath = f'{closest.epoch}/{closest.tile}/{closest.image}/{closest.filename}'
            image_path = vlass_path
        elif 'racs' in self.survey:
            filepath = f'RACS_test4_1.05_{closest.field}.fits'
            pol = self.survey[-1]
        elif 'vast' in self.survey:
            pattern = re.compile(r'vastp(\d[x])([IV])')
            epoch = pattern.sub(r'\1', self.survey)
            pol = pattern.sub(r'\2', self.survey)
            filepath = f'VAST_{closest.field}.EPOCH0{epoch}.{pol}.fits'
        else:
            filepath = f'*{closest.field}*0.restored.fits'

        try:
            self.filepath = glob.glob(image_path + filepath)[0]
        except IndexError:
            raise FITSException(
                f'Could not match {self.survey} image filepath: \n{image_path + filepath}')

        with fits.open(self.filepath) as hdul:
            self.header, data = hdul[0].header, hdul[0].data
            wcs = WCS(self.header, naxis=2)
            self.mjd = Time(self.header['DATE']).mjd

            try:
                cutout = Cutout2D(data[0, 0, :, :], self.position, self.radius * u.deg, wcs=wcs)
            except IndexError:
                cutout = Cutout2D(data, self.position, self.radius * u.deg, wcs=wcs)
            self.data = cutout.data * 1000
            self.wcs = cutout.wcs

        if 'racs' in self.survey or 'vast' in self.survey:
            self.pos_err = SURVEYS.loc[self.basesurvey].pos_err
            self._get_source()
        else:
            # Probably using vlass, yet to include aegean catalogs
            self.plot_sources = False
            self.plot_neighbours = False

    def _get_panstarrs_cutout(self):
        """Fetch cutout data via PanSTARRS DR2 API."""
        path = cutout_cache + 'panstarrs/{}_{}arcmin_{}_{}.fits'.format(self.band,
                                                                        '{:.3f}',
                                                                        '{:.3f}',
                                                                        '{:.3f}',)
        imgpath = path.format(self.radius * 60, self.ra, self.dec)
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
                self.mjd = row['mjd']
                filt = row['filter']
                url = urlbase + row['filename']
                path = cutout_cache + 'panstarrs/{}_{}arcmin_{}_{}.fits'.format(filt,
                                                                                '{:.3f}',
                                                                                '{:.3f}',
                                                                                '{:.3f}',)
                path = path.format(self.radius * 60, self.ra, self.dec)

                img = requests.get(url, allow_redirects=True)

                if not os.path.exists(path):
                    with open(path, 'wb') as f:
                        f.write(img.content)

        with fits.open(imgpath) as hdul:
            self.header, self.data = hdul[0].header, hdul[0].data
            self.wcs = WCS(self.header, naxis=2)

    def _get_skymapper_cutout(self):
        """Fetch cutout data via Skymapper API."""

        path = cutout_cache + self.survey + '/dr2_jd{:.3f}_{:.3f}arcmin_{:.3f}_{:.3f}'
        linka = 'http://api.skymapper.nci.org.au/aus/siap/dr2/'
        linkb = 'query?POS={:.5f},{:.5f}&SIZE={:.3f}&BAND=all&RESPONSEFORMAT=CSV'
        linkc = '&VERB=3&INTERSECT=covers'
        sm_query = linka + linkb + linkc

        link = linka + 'get_image?IMAGE={}&SIZE={}&POS={},{}&FORMAT=fits'

        table = requests.get(sm_query.format(self.ra, self.dec, self.radius))
        df = pd.read_csv(io.StringIO(table.text))
        assert len(df) > 0, f'No Skymapper image at {self.position.ra:.2f}, {self.position.dec:.2f}'

        df = df[df.band == 'z']
        self.mjd = df.iloc[0]['mjd_obs']
        link = df.iloc[0].get_image

        img = requests.get(link)

        path = path.format(self.mjd, self.radius * 60, self.ra, self.dec)

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
            self.logger.warning(f"Using maximum DECam LS cutout radius of {maxradius:.3f} deg")

        link = f"http://legacysurvey.org/viewer/fits-cutout?ra={self.ra}&dec={self.dec}"
        link += f"&size={size}&layer=dr8&pixscale=0.262&bands={self.band}"
        img = requests.get(link)

        path = cutout_cache + self.survey + '/dr8_jd{:.3f}_{:.3f}arcmin_{:.3f}_{:.3f}_{}band'
        path = path.format(self.mjd, self.radius * 60, self.ra, self.dec, self.band)
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(img.content)

        with fits.open(path) as hdul:
            self.header, self.data = hdul[0].header, hdul[0].data
            self.wcs = WCS(self.header, naxis=2)

        msg = f"No DECam LS image at {self.position.ra:.2f}, {self.position.dec:.2f}"
        assert self.data is not None, msg

    def _get_skyview_cutout(self):
        """Fetch cutout data via SkyView API."""

        sv = SkyView()
        path = cutout_cache + self.survey + '/{:.3f}arcmin_{:.3f}_{:.3f}.fits'
        path = path.format(self.radius * 60, self.ra, self.dec)
        progress = self.kwargs.get('progress', False)

        if not os.path.exists(path):
            skyview_key = SURVEYS.loc[self.survey].sv
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
                        self.logger.warning(e)
                    self.mjd = None

            self.data *= 1000

    def _find_image(self):
        """Return DataFrame of survey fields containing coord."""

        survey = self.survey.replace('I', '').replace('V', '')
        try:
            image_df = pd.read_csv(aux_path + f'{survey}_fields.csv')
        except FileNotFoundError:
            raise FITSException(f"Missing field metadata csv for {survey}.")

        beam_centre = SkyCoord(ra=image_df['cr_ra_pix'], dec=image_df['cr_dec_pix'],
                               unit=u.deg)
        image_df['dist_field_centre'] = beam_centre.separation(self.position).deg

        pbeamsize = 1 * u.degree if self.survey == 'vlass' else 5 * u.degree
        return image_df[image_df.dist_field_centre < pbeamsize].reset_index(drop=True)

    def _obfuscate(self):
        """Remove all coordinates and identifying information."""
        lon = self.ax.coords[0]
        lat = self.ax.coords[1]
        lon.set_ticks_visible(False)
        lon.set_ticklabel_visible(False)
        lat.set_ticks_visible(False)
        lat.set_ticklabel_visible(False)
        lon.set_axislabel('')
        lat.set_axislabel('')

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
        self.ax.set_xlabel('RA (J2000)')
        self.ax.set_ylabel('Dec (J2000)')

        if self.kwargs.get('title', True):
            self.ax.set_title(SURVEYS.loc[self.survey]['name'], fontdict={'fontsize': 20,
                                                                          'fontweight': 10})
        if self.kwargs.get('obfuscate', False):
            self._obfuscate()

        if self.kwargs.get('annotation'):
            color = 'white' if self.cmap == 'hot' else 'k'
            self.ax.text(0.05, 0.85, self.kwargs.get('annotation'), color=color,
                         weight='bold', transform=self.ax.transAxes)

    def _add_cornermarker(self, ra, dec, span, offset):
        color = 'white' if self.cmap != 'gray_r' else 'r'
        cosdec = np.cos(np.radians(dec))
        raline = Line2D(xdata=[ra + offset / cosdec, ra + span / cosdec],
                        ydata=[dec, dec],
                        color=color, linewidth=2,
                        path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()],
                        transform=self.ax.get_transform('world'))
        decline = Line2D(xdata=[ra, ra],
                         ydata=[dec + offset, dec + span],
                         color=color, linewidth=2,
                         path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()],
                         transform=self.ax.get_transform('world'))
        self.ax.add_artist(raline)
        self.ax.add_artist(decline)

    def plot(self, fig=None, ax=None):
        """Plot survey data and position overlay."""
        self.sign = self.kwargs.get('sign', 1)
        self._plot_setup(fig, ax)
        self.data *= self.sign
        absmax = max(self.data.max(), self.data.min(), key=abs)
        self.logger.debug(f"Max flux in cutout: {absmax:.2f} mJy.")
        rms = np.sqrt(np.mean(np.square(self.data)))
        self.logger.debug(f"RMS flux in cutout: {rms:.2f} mJy.")

        assert (sum((~np.isnan(self.data).flatten())) > 0 and sum(self.data.flatten()) != 0), \
            f"No data in {self.survey}"

        if self.kwargs.get('maxnorm'):
            self.norm = ImageNormalize(self.data, interval=ZScaleInterval(),
                                       vmax=self.data.max(), clip=True)
        else:
            self.norm = ImageNormalize(self.data, interval=ZScaleInterval(contrast=0.2),
                                       clip=True)

        self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        if self.kwargs.get('bar', True):
            try:
                self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)
            except UnboundLocalError:
                self.logger.error("Colorbar failed. Upgrade to recent version of astropy ")

        if self.psf:
            try:
                self.bmaj = self.header['BMAJ'] * 3600
                self.bmin = self.header['BMIN'] * 3600
                self.bpa = self.header['BPA']
            except KeyError:
                self.logger.warning('Header did not contain PSF information.')
                try:
                    self.bmaj = self.psf[0]
                    self.bmin = self.psf[1]
                    self.bpa = 0
                    self.logger.warning('Using supplied BMAJ/BMin. Assuming BPA=0')
                except ValueError:
                    self.logger.error('No PSF information supplied.')

            rhs = self.wcs.wcs_pix2world(self.data.shape[0], 0, 1)
            lhs = self.wcs.wcs_pix2world(0, 0, 1)

            # Offset PSF marker by the major axis in pixel coordinates
            try:
                cdelt = self.header['CDELT1']
            except KeyError:
                cdelt = self.header['CD1_1']
            beamsize_pix = self.bmaj / abs(cdelt) / 3600
            ax_len_pix = abs(lhs[0] - rhs[0]) / abs(cdelt) / 3600
            beam = self.wcs.wcs_pix2world(beamsize_pix, beamsize_pix, 1)
            self.beamx = beam[0]
            self.beamy = beam[1]

            self.beam = Ellipse((self.beamx, self.beamy),
                                self.bmin / 3600, self.bmaj / 3600,
                                -self.bpa, facecolor='white', edgecolor='k',
                                transform=self.ax.get_transform('world'),
                                zorder=10)
            self.ax.add_patch(self.beam)

            # Optionally plot square around the PSF
            # Set size to greater of 110% PSF size or 10% ax length
            if self.kwargs.get('beamsquare', False):
                boxsize = max(beamsize_pix * 1.15, ax_len_pix * .1)
                offset = beamsize_pix - boxsize / 2
                self.square = Rectangle((offset, offset),
                                        boxsize, boxsize,
                                        facecolor='white', edgecolor='k',
                                        # transform=self.ax.get_transform('world'),
                                        zorder=5)
                self.ax.add_patch(self.square)

        if self.plot_sources:
            if self.kwargs.get('corner'):
                self._add_cornermarker(self.source.ra_deg_cont,
                                       self.source.dec_deg_cont,
                                       self.kwargs.get('corner_span', 20 / 3600),
                                       self.kwargs.get('corner_offset', 10 / 3600))
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
                self._add_cornermarker(self.ra,
                                       self.dec,
                                       self.kwargs.get('corner_span', 20 / 3600),
                                       self.kwargs.get('corner_offset', 10 / 3600))
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


class ContourCutout(Cutout):

    def __init__(self, survey, position, radius, **kwargs):
        self.contours = kwargs.get('contours', 'racsI')
        self.clabels = kwargs.get('clabels', False)
        self.bar = kwargs.get('bar', False)
        self.radio = Cutout(self.contours, position, radius, **kwargs)
        try:
            self.mjd = Time(self.radio.header['DATE']).mjd
        except KeyError:
            self.mjd = None

        self.correct_pm = kwargs.get('pm')

        super().__init__(survey, position, radius, **kwargs)
        if self.correct_pm:
            self._correct_proper_motion()

    def _correct_proper_motion(self):
        """Check SIMBAD for nearby star or pulsar and plot a cross at corrected coordinates"""

        simbad = Simbad.query_region(self.position, radius=180 * u.arcsec)
        self.epoch = self.kwargs.get('epoch', 2019.609728489631)
        self.epochtype = 'MJD' if self.epoch > 3e3 else 'decimalyear'
        self.mjd = Time(self.epoch, format=self.epochtype.lower()).mjd

        if simbad is not None:
            simbad = table2df(simbad)
            pm_types = ['*', '**', 'PM*', 'Star', 'PSR', 'Pulsar', 'Flare*']
            simbad = simbad[(simbad['OTYPE'].isin(pm_types)) | (simbad['SP_TYPE'].str.len() > 0)]

            if len(simbad) == 0:
                self.logger.warning("No high proper-motion objects within 180 arcsec.")
                self.correct_pm = False
                return

            dist = Distance(parallax=simbad['PLX_VALUE'].values * u.mas)
            simbad['oldpos'] = SkyCoord(
                ra=simbad['RA_d'].values * u.deg,
                dec=simbad['DEC_d'].values * u.deg,
                frame='icrs',
                distance=dist,
                pm_ra_cosdec=simbad['PMRA'].values * u.mas / u.yr,
                pm_dec=simbad['PMDEC'].values * u.mas / u.yr,
                obstime='J2000')

            newpos = simbad.oldpos.apply(
                lambda x: x.apply_space_motion(Time(self.mjd, format='mjd')))
            simbad['PM Corrected Separation (arcsec)'] = np.round(newpos.apply(
                lambda x: x.separation(self.position).arcsec), 3)

            simbad = simbad[['MAIN_ID', 'OTYPE', 'SP_TYPE', 'DISTANCE_RESULT',
                             'PM Corrected Separation (arcsec)']].copy()
            simbad = simbad.rename(columns={'MAIN_ID': 'Object', 'OTYPE': 'Type',
                                            'DISTANCE_RESULT': 'Separation (arcsec)',
                                            'SP_TYPE': 'Spectral Type'})
            self.logger.info(f'SIMBAD results:\n {simbad}')
            self.simbad = simbad.sort_values('PM Corrected Separation (arcsec)').head()
            nearest = self.simbad['PM Corrected Separation (arcsec)'].idxmin()
            self.pm_coord = newpos[nearest]
            object = self.simbad.loc[nearest].Object
            msg = f'Proper motion corrected {object} to <{self.pm_coord.ra}, {self.pm_coord.dec}>'
            self.logger.info(msg)
            missing = simbad[simbad['PM Corrected Separation (arcsec)'].isna()]
            if self.simbad['PM Corrected Separation (arcsec)'].min() > 15:
                self.logger.warning("No PM corrected objects within 15 arcsec")
                self.correct_pm = False
            if len(missing) > 0:
                msg = f"Some objects missing PM data, and may be closer matches \n {missing}"
                self.logger.warning(msg)

        else:
            self.correct_pm = False

    def plot(self, fig=None, ax=None):
        self.sign = self.kwargs.get('sign', 1)
        self._plot_setup(fig, ax)

        assert (sum((~np.isnan(self.data).flatten())) > 0 and sum(self.data.flatten()) != 0), \
            f"No data in {self.survey}"

        self.norm = ImageNormalize(self.data, interval=ZScaleInterval(contrast=0.2),
                                   clip=True)

        if self.survey == 'swift_xrtcnt':
            self.im = self.ax.imshow(self.data, cmap=self.cmap)
        else:
            self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        # Plot radio contours
        self.radio.data *= self.sign
        self.peak = self.radio.data.max()
        color = 'blue' if self.cmap == 'hot' else 'orange'
        self.cs = self.ax.contour(self.radio.data, transform=self.ax.get_transform(self.radio.wcs),
                                  levels=[self.peak * x for x in [.4, .6, .8]], colors=color,
                                  linewidths=3)
        if self.clabels:
            self.ax.clabel(self.cs, fontsize=10, fmt='%1.1f mJy')
        if self.bar:
            self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)

        if self.survey == 'panstarrs' and self.kwargs.get('title', True):
            self.ax.set_title(f"{SURVEYS.loc[self.survey]['name']} ({self.band}-band)")

        # Plot PM corrected location
        if self.correct_pm:
            name = self.simbad.iloc[0]["Object"]
            self.ax.scatter(self.pm_coord.ra, self.pm_coord.dec, marker='x', s=200, color='r',
                            transform=self.ax.get_transform('world'),
                            label=f'{name} position at {self.epochtype} {self.epoch} ')
            self.ax.legend()
