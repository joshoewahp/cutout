#!/usr/bin/env python
"""
Cutout module documentation
"""

import io
import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Distance
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.simbad import Simbad
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse

from astroutils.io import FITSException, get_surveys
from cutout.services import (
    RawCutout,
    LocalCutout,
    MWATSCutout,
    SkymapperCutout,
    PanSTARRSCutout,
    DECamCutout,
    IPHASCutout,
    SkyviewCutout,
)

warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

SURVEYS = get_surveys()
SURVEYS.set_index('survey', inplace=True)

Simbad.add_votable_fields(
    'otype',
    'ra(d)',
    'dec(d)',
    'parallax',
    'pmdec',
    'pmra',
    'distance',
    'sptype',
    'distance_result',
)

logger = logging.getLogger(__name__)


@dataclass
class CornerMarker:
    position: SkyCoord
    wcs: WCS
    colour: str
    span: float
    offset: float

    def __post_init__(self):
        self.datapos = self.wcs.wcs_world2pix(self.position.ra, self.position.dec, 1)

    def _get_xy_lims(self):
        """Get x/y pixel coordinates of marker position."""
        
        x = self.datapos[0] - 1
        y = self.datapos[1] - 1

        return x, y

    @property
    def raline(self):
        """Construct right ascension marker line."""

        x, y = self._get_xy_lims()
        raline = Line2D(
            xdata=[x - self.offset, x - self.span],
            ydata=[y, y],
            color=self.colour,
            linewidth=2,
            zorder=10,
            path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
        )

        return raline
        
    @property
    def decline(self):
        """Construct declination marker line."""

        x, y = self._get_xy_lims()
        decline =  Line2D(
            xdata=[x, x],
            ydata=[y + self.offset, y + self.span],
            color=self.colour,
            linewidth=2,
            zorder=10,
            path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
        )

        return decline

    def in_pixel_range(self, pixmin: int, pixmax: int) -> bool:
        """Check whether the pixel coordinate of marker is in a valid range."""
        
        if any(i < pixmin or i > pixmax or np.isnan(i) for i in self.datapos):
            return False

        return True


class Cutout:

    def __init__(self, survey, position, size, stokes='i', **kwargs):
        self.survey = survey
        self.position = position
        self.ra = self.position.ra.to_value(u.deg)
        self.dec = self.position.dec.to_value(u.deg)
        self.size = size
        self.stokes = stokes
        self.sign = kwargs.pop('sign', 1)
        self.cmap = kwargs.pop('cmap', 'coolwarm' if self.stokes == 'v' else 'gray_r')
        self.correct_pm = kwargs.pop('pm', False)
        self.rotate_axes = False

        self.options = kwargs

        try:
            self._get_cutout()
            self._determine_epoch()
        except Exception as e:
            msg = f"{survey} failed: {e}"
            raise FITSException(msg)

    def __repr__(self):
        template = "Cutout('{}', SkyCoord(ra={:.4f}, dec=dec{:.4f}, unit='deg'), size={:.4f})"
        return template.format(self.survey, self.ra, self.dec, self.size)

    def __getattr__(self, name):
        """Overload __getattr__ to make CutoutService attributes accessible directly from Cutout."""

        try:
            return getattr(self._cutout, name)
        except (RecursionError, AttributeError):
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

    @property
    def normalisable(self):
        """Property that is True if data can be colourmap normalised."""

        return np.abs(np.nansum(self.data)) > 0

    def _check_data_valid(self):
        """Run checks for invalid or missing data (e.g. all NaN or 0 pixels) from valid FITS file."""

        is_valid = (sum(~np.isnan(self.data).flatten()) > 0 and self.data.flatten().sum() != 0)
        if not is_valid:
            raise FITSException(f"No data in {self.survey}")

    def _get_cutout(self):

        # # This was used to pass in a data array from another cutout, which is useful when
        # # shifting the image data to a proper motion corrected location. Refactor this to
        # # de-clutter this function

        # if self.options.get('data'):
        #     c = self.options.get('data')
        #     self.mjd = c.mjd
        #     self.data = c.data
        #     self.wcs = c.wcs
        #     self.header = c.header
        #     self.position = c.position

        #     return

        if os.path.isfile(self.survey):
            self._cutout = RawCutout(self)
            self.surveyname = ''
        else:
            self.surveyname = SURVEYS.loc[self.survey]['name']

            if SURVEYS.loc[self.survey].local:
                self._cutout = LocalCutout(self)
            elif self.survey == 'skymapper':
                self._cutout = SkymapperCutout(self)
            elif self.survey == 'panstarrs':
                self._cutout = PanSTARRSCutout(self)
            elif self.survey == 'decam':
                self._cutout = DECamCutout(self)
            elif self.survey == 'iphas':
                self._cutout = IPHASCutout(self)
            elif self.survey == 'mwats':
                self._cutout = MWATSCutout(self)
            else:
                self._cutout = SkyviewCutout(self)

        self._cutout.fetch_sources(self)

        return                           
    
    def _determine_epoch(self):

        epoch = self.options.pop('epoch', False)

        fits_date_keys = [
            'MJD-OBS',
            'MJD',
            'DATE-OBS',
            'DATE',
        ]

        # Try each FITS header keyword in sequence if epoch not provided directly
        if not epoch:
            for key in fits_date_keys:
                try:
                    epoch = self.header[key]
                    epochtype = 'mjd' if 'MJD' in key else None
                    break
                except KeyError:
                    continue

        # If epoch still not resolved, disable PM correction
        if not epoch:
            msg = f"Could not detect {self.survey} epoch, PM correction disabled."
            logger.warning(msg)
            self.correct_pm = False

            return

        self.mjd = Time(epoch, format=epochtype).mjd

    def _plot_setup(self, fig, ax):
        """Create figure and determine normalisation parameters."""

        self._check_data_valid()

        if ax:
            self.fig = fig
            self.ax = ax
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection=self.wcs)

        # Set basic figure display options
        if self.options.get('grid', True):
            self.ax.coords.grid(color='white', alpha=0.5)

        if self.options.get('title', True):
            title = self.options.get('title', self.surveyname)
            self.ax.set_title(title, fontdict={'fontsize': 20, 'fontweight': 10})

        self.set_xlabel('RA (J2000)')
        self.set_ylabel('Dec (J2000)')

        # Set compact or extended label / tick configuration
        if self.options.get('compact', False):
            tickcolor = 'k' if np.nanmax(np.abs(self.data)) == np.nanmax(self.data) else 'gray'

            lon = self.ax.coords[0]
            lat = self.ax.coords[1]

            lon.display_minor_ticks(True)
            lat.display_minor_ticks(True)

            lon.set_ticks(number=5)
            lat.set_ticks(number=5)

            self.ax.tick_params(axis='both', direction='in', length=5, color=tickcolor)
            self.padlevel = self.options.get('ylabelpad', 5)

        # Set colourmap normalisation
        self.norm = self._get_cmap_normalisation()
        
       
    def _get_cmap_normalisation(self):
        """Create colourmap normalisation for cutout map.
        
        User supplied parameters take precedence in order of:
        --maxnorm
        --vmin and/or --vmax
        
        or by default ZScaleInterval computes low-contrast limits which
        are made symmetric for Stokes V cutouts.
        """

        # Non-normalisable data should return cmap = None
        if not self.normalisable:
            return

        # Get min/max based upon ZScale with contrast parameter
        contrast = self.options.get('contrast', 0.2)
        vmin, vmax = ZScaleInterval(contrast=contrast).get_limits(self.data)

        # Make this symmetric if using Stokes V
        if self.stokes == 'v':
            v = max(abs(vmin), abs(vmax))
            vmin = -v
            vmax = v

        # Override with user-supplied values if present
        if self.options.get('vmin') or self.options.get('vmax'):
            vmin = self.options.get('vmin', -2)
            vmax = self.options.get('vmax', 1)

        # Normalise with maximum value in data
        if self.options.get('maxnorm'):
            vmax = np.nanmax(self.data)
            vmin = None

        norm = ImageNormalize(
            self.data,
            interval=ZScaleInterval(),
            vmin=vmin,
            vmax=vmax,
            clip=True
        )

        return norm


    def _align_ylabel(self, ylabel):
        """Consistently offset y-axis label with varying coordinate label size."""
        
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

    def add_annotation(self, annotation, location='upper left', **kwargs):

        props = {
            'size': kwargs.get('size', 12),
            'color': kwargs.get('color', 'firebrick'),
            'alpha': kwargs.get('alpha', 0.7),
            'weight': kwargs.get('weight', 'heavy'),
            'family': 'sans-serif',
            'usetex': False,
        }
        text = AnchoredText(annotation, loc=location, frameon=False, prop=props) 

        self.ax.add_artist(text)

    def add_cornermarker(self, marker):

        if not marker.in_pixel_range(0, len(self.data)):
            msga = "Cornermarker will be disabled as RA and Dec are outside of data range."
            logger.warning(msga + msgb)

            return

        self.ax.add_artist(marker.raline)
        self.ax.add_artist(marker.decline)

    def add_source_ellipse(self):
        """Overplot dashed line ellipses for the nearest source within positional uncertainty."""

        # Add ellipse for source within positional uncertainty
        if self.plot_source:
            source_colour = 'springgreen' if self.stokes == 'v' else 'springgreen'
            self.sourcepos = Ellipse(
                (self.source.ra_deg_cont, self.source.dec_deg_cont),
                self.source.min_axis / 3600,
                self.source.maj_axis / 3600,
                -self.source.pos_ang,
                facecolor='none',
                edgecolor=source_colour,
                ls=':',
                lw=2,
                zorder=10,
                transform=self.ax.get_transform('world')
            )
            self.ax.add_patch(self.sourcepos)
            
        # Add ellipse for other components in the FoV
        if self.plot_neighbours:
            neighbour_colour = 'k' if self.stokes == 'v' else 'rebeccapurple'
            for idx, neighbour in self.neighbours.iterrows():
                n = Ellipse(
                    (neighbour.ra_deg_cont, neighbour.dec_deg_cont),
                    neighbour.min_axis / 3600,
                    neighbour.maj_axis / 3600,
                    -neighbour.pos_ang,
                    facecolor='none',
                    edgecolor=neighbour_colour,
                    ls=':',
                    lw=2,
                    zorder=1,
                    transform=self.ax.get_transform('world')
                )
                self.ax.add_patch(n)
        
    def add_psf(self):

        try:
            self.bmaj = self.header['BMAJ'] * 3600
            self.bmin = self.header['BMIN'] * 3600
            self.bpa = self.header['BPA']
        except KeyError:
            logger.warning('Header did not contain PSF information, disabling PSF marker.')
            return

        try:
            cdelt = self.header['CDELT1']
        except KeyError:
            cdelt = self.header['CD1_1']

        if self.options.get('beamsquare'):
            frame = True
            facecolor = 'k'
            edgecolor = 'k'
        else:
            frame = False
            facecolor = 'white'
            edgecolor = 'k'

        x = self.bmin / abs(cdelt) / 3600
        y = self.bmaj / abs(cdelt) / 3600

        self.beam = AnchoredEllipse(
            self.ax.transData,
            width=x,
            height=y,
            angle=self.bpa,
            loc=3,
            pad=0.5,
            borderpad=0.4,
            frameon=frame
        )
        self.beam.ellipse.set(facecolor=facecolor, edgecolor=edgecolor)

        self.ax.add_artist(self.beam)

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

        self._plot_setup(fig, ax)

        if self.stokes == 'v':
            self.cmap = plt.cm.coolwarm

        absmax = max(self.data.max(), self.data.min(), key=abs)
        rms = np.sqrt(np.mean(np.square(self.data)))

        logger.debug(f"Max flux in cutout: {absmax:.2f} mJy.")
        logger.debug(f"RMS flux in cutout: {rms:.2f} mJy.")

        self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        if self.options.get('bar', True):
            try:
                self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)
            except UnboundLocalError:
                logger.error("Colorbar failed. Upgrade to recent version of astropy ")

        if self.options.get('psf'):
            self.add_psf()

        self.add_source_ellipse()

    def save(self, path, fmt='png'):
        """Save figure with tight bounding box."""
        self.fig.savefig(path, format=fmt, bbox_inches='tight')

    def savefits(self, path):
        """Save FITS cutout."""
        
        header = self.wcs.to_header()
        header['BUNIT'] = 'Jy/beam'
        hdu = fits.PrimaryHDU(data=self.data / 1e3, header=header)
        hdu.writeto(path)

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel)

    def set_ylabel(self, ylabel, align=False):
        if align:
            self._align_ylabel(ylabel)
        else:
            self.ax.set_ylabel(ylabel)


class ContourCutout(Cutout):

    def __init__(self, survey, position, size, **kwargs):

        # If custom data provided for ContourCutout, pop from kwargs
        # to avoid being read as radio data by Cutout sub-call.
        data = kwargs.pop('data', None)
        stokes = kwargs.pop('stokes', 'i')

        # Other ContourCutout specific keywords are also popped
        self.contours = kwargs.pop('contours', 'racs-low')
        self.clabels = kwargs.pop('clabels', False)
        self.bar = kwargs.pop('bar', False)
        self.band = kwargs.pop('band', 'g')

        self.radio = Cutout(self.contours, position, size, **kwargs)
        self.radio.mjd = self.radio.mjd

        super().__init__(survey, position, size, data=data, stokes=stokes, **kwargs)

        if self.correct_pm:
            self.correct_proper_motion()

    def _add_pm_location(self):
        """Overplot proper motion correction as an arrow."""

        name = self.simbad.iloc[0]["Object"]
        oldcoord = SkyCoord(self.oldpos.ra, self.oldpos.dec, unit=u.deg)
        newcoord = SkyCoord(self.pm_coord.ra, self.pm_coord.dec, unit=u.deg)
        oldtime = Time(self.mjd, format='mjd').decimalyear
        newtime = Time(self.radio.mjd, format='mjd').decimalyear
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
            self.ax.arrow(
                self.oldpos.ra.deg,
                self.oldpos.dec.deg,
                dra.deg,
                ddec.deg,
                width=8e-5,
                color='r',
                length_includes_head=True,
                zorder=10,
                transform=self.ax.get_transform('world')
            )

            arrow_handle = Line2D([], [], ls='none', marker=r'$\leftarrow$', markersize=10, color='r')
            arrow_label = f'Proper motion from J{oldtime:.2f}-J{newtime:.2f}'
            handles.append(arrow_handle)
            labels.append(arrow_label)

            self.ax.legend(handles, labels)

    def shift_coordinate_grid(self, pm_coord, shift_epoch):
        """Shift WCS of pixel data to epoch based upon the proper motion encoded in pm_coord."""

        # Replace pixel data / WCS with copy centred on source
        contour_background = ContourCutout(
            self.survey,
            pm_coord,
            self.size,
            band=self.band,
        )
        self.data = contour_background.data
        self.wcs = contour_background.wcs
        
        # Astropy for some reason can't decide on calling this pm_ra or pm_ra_cosdec
        try:
            pm_ra = pm_coord.pm_ra
        except AttributeError as e:
            pm_ra = pm_coord.pm_ra_cosdec

        # Update CRVAL coordinates based on propagated proper motion
        orig_pos = SkyCoord(
            ra=self.wcs.wcs.crval[0] * u.deg,
            dec=self.wcs.wcs.crval[1] * u.deg,
            frame='icrs',
            distance=pm_coord.distance,
            pm_ra_cosdec=pm_ra,
            pm_dec=pm_coord.pm_dec,
            obstime=pm_coord.obstime,
        )
        newpos = orig_pos.apply_space_motion(shift_epoch)

        self.wcs.wcs.crval = [newpos.ra.deg, newpos.dec.deg]

    def correct_proper_motion(self, invert=False):
        """Check SIMBAD for nearby star or pulsar and plot a cross at corrected coordinates."""

        # If mjd not set directly, check that it was set from FITS headers in get_cutout method
        if self.mjd is None:
            raise FITSException("Date could not be inferred from header, supply with epoch keyword.")

        obstime = Time(self.mjd, format='mjd')

        simbad = Simbad.query_region(self.position, radius=180 * u.arcsec)

        # Catch SIMBAD failure either from None return of query or no stellar type matches in region
        try:
            simbad = simbad.to_pandas()
            pm_types = ['*', '**', 'PM*', 'EB*', 'Star', 'PSR', 'Pulsar', 'Flare*']
            simbad = simbad[(simbad['OTYPE'].isin(pm_types)) | (simbad['SP_TYPE'].str.len() > 0)]

            assert len(simbad) > 0

        except (ValueError, AssertionError):
            logger.warning("No high proper-motion objects within 180 arcsec.")
            self.correct_pm = False

            return

        # Treat non-existent proper motion parameters as extremely distant objects
        simbad['PMRA'].fillna(0, inplace=True)
        simbad['PMDEC'].fillna(0, inplace=True)
        simbad['PLX_VALUE'].fillna(0.01, inplace=True)

        newtime = Time(self.radio.mjd, format='mjd')
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
            obstime='J2000',
        )

        datapos = simbad.j2000pos.apply(lambda x: x.apply_space_motion(obstime))
        newpos = simbad.j2000pos.apply(lambda x: x.apply_space_motion(newtime))

        simbad_cols = {
            'MAIN_ID': 'Object',
            'OTYPE': 'Type',
            'SP_TYPE': 'Spectral Type',
            'DISTANCE_RESULT': 'Separation (arcsec)',
        }
        simbad = simbad.rename(columns=simbad_cols)
        simbad = simbad[simbad_cols.values()].copy()
        simbad['PM Corrected Separation (arcsec)'] = np.round(newpos.apply(
            lambda x: x.separation(self.position).arcsec), 3)

        # Only display PM results if object within 15 arcsec
        if simbad['PM Corrected Separation (arcsec)'].min() > 15:
            logger.warning("No PM corrected objects within 15 arcsec")
            self.correct_pm = False

            return

        self.simbad = simbad.sort_values('PM Corrected Separation (arcsec)')
        logger.info(f'SIMBAD results:\n {self.simbad.head()}')

        nearest = self.simbad['PM Corrected Separation (arcsec)'].idxmin()

        self.oldpos = datapos[nearest]
        self.pm_coord = newpos[nearest]

        near_object = self.simbad.loc[nearest].Object
        msg = f'{near_object} proper motion corrected to <{self.pm_coord.ra:.4f}, {self.pm_coord.dec:.4f}>'
        logger.info(msg)

        missing = simbad[simbad['PM Corrected Separation (arcsec)'].isna()]
        if len(missing) > 0:
            msg = f"Some objects missing PM data, and may be a closer match than presented:\n {missing}"
            logger.warning(msg)

        return

    def plot(self, fig=None, ax=None):
        self._plot_setup(fig, ax)

        self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        # Plot radio contours
        self.radio.data *= self.sign
        self.peak = np.nanmax(self.radio.data)
        self.radiorms = np.sqrt(np.mean(np.square(self.radio.data)))

        if self.options.get('rmslevels'):
            self.levels = [self.radiorms * x for x in [3, 6]]
        elif self.options.get('peaklevels'):
            midx = int(self.radio.data.shape[0] / 2)
            midy = int(self.radio.data.shape[1] / 2)
            peak = self.radio.data[midx, midy]
            self.levels = np.logspace(np.log10(0.3 * peak), np.log10(0.9 * peak), 3)
        else:
            self.levels = [self.peak * x for x in [.3, .6, .9]]

        contour_width = self.options.get('contourwidth', 3)
        contour_color = 'k' if self.cmap == 'coolwarm' else 'orange'

        self.cs = self.ax.contour(
            self.radio.data,
            transform=self.ax.get_transform(self.radio.wcs),
            levels=self.levels,
            colors=contour_color,
            linewidths=contour_width,
        )

        if self.clabels:
            self.ax.clabel(self.cs, fontsize=10, fmt='%1.1f mJy')

        if self.bar:
            self.fig.colorbar(self.im, label=r'Flux Density (mJy beam$^{-1}$)', ax=self.ax)

        if self.survey == 'panstarrs' and self.options.get('title', True):
            self.ax.set_title(f"{SURVEYS.loc[self.survey]['name']} ({self.band}-band)")

        if self.correct_pm:
            self._add_pm_location()
            
