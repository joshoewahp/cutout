#!/usr/bin/env python
"""
Cutout module documentation
"""

import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass

import astropy.units as u
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astroutils.io import FITSException, get_surveys
from astroutils.query import query_simbad
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, AnchoredText, AuxTransformBox
from matplotlib.patches import Ellipse

# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
from regions import EllipseSkyRegion

from cutout.services import (
    DECamCutout,
    IPHASCutout,
    LocalCutout,
    MWATSCutout,
    PanSTARRSCutout,
    RawCutout,
    SkymapperCutout,
    SkyviewCutout,
)

warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)

SURVEYS = get_surveys()
SURVEYS.set_index("survey", inplace=True)


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
            path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
        )

        return raline

    @property
    def decline(self):
        """Construct declination marker line."""

        x, y = self._get_xy_lims()
        decline = Line2D(
            xdata=[x, x],
            ydata=[y + self.offset, y + self.span],
            color=self.colour,
            linewidth=2,
            zorder=10,
            path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
        )

        return decline

    def in_pixel_range(self, pixmin: int, pixmax: int) -> bool:
        """Check whether the pixel coordinate of marker is in a valid range."""

        return self.wcs.footprint_contains(self.position)


class Cutout:
    def __init__(self, survey, position, size, stokes="i", tiletype="TILES", **kwargs):
        self.survey = survey
        self.position = position
        self.ra = self.position.ra.to_value(u.deg)
        self.dec = self.position.dec.to_value(u.deg)
        self.size = size
        self.stokes = stokes
        self.tiletype = tiletype
        self.sign = kwargs.pop("sign", 1)
        self.bar = kwargs.pop("bar", True)
        self.cmap = kwargs.pop("cmap", "coolwarm" if self.stokes == "v" else "gray_r")
        self.band = kwargs.pop("band", "g")
        self.selavy = kwargs.pop("selavy", None)
        self.correct_pm = False
        self.offsets = False

        # Create contour attribute dictionary
        self.cs_dict = defaultdict(dict)

        self.options = kwargs

        try:
            self._get_cutout()
            self._determine_epoch()
        except (AssertionError, FITSException, FileNotFoundError) as e:
            msg = f"{survey} failed: {e}"
            raise FITSException(msg)

    def __repr__(self):
        temp = "{}('{}', SkyCoord(ra={:.4f}, dec={:.4f}, unit='deg'), size={:.4f})"
        return temp.format(
            self.__class__.__name__,
            self.survey,
            self.ra,
            self.dec,
            self.size,
        )

    def __getattr__(self, name):
        """Overload __getattr__ to make CutoutService attributes accessible directly from Cutout."""

        try:
            return getattr(self._cutout, name)
        except (RecursionError, AttributeError):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )

    def _check_data_valid(self):
        """Run checks for invalid or missing data (e.g. all NaN or 0 pixels) from valid FITS file."""

        non_nan_count = sum(~np.isnan(self.data).flatten())
        non_zero_count = np.abs(np.nansum(self.data))

        is_valid = non_nan_count > 0 and non_zero_count > 0

        if not is_valid:
            raise FITSException(f"No data in {self.survey}")

    def _get_cutout(self):
        # First check if survey parameter is a FITS image path
        if os.path.isfile(self.survey) or self.survey.endswith("fits"):
            self._cutout = RawCutout(self)
            self.surveyname = ""
            self.is_radio = True

            return

        if self.survey not in SURVEYS.index:
            raise FITSException(f"Survey {self.survey} not in defined in surveys.json.")

        self.surveyname = SURVEYS.loc[self.survey]["name"]
        self.is_radio = SURVEYS.loc[self.survey].radio

        # Any locally accessible surveys are created with the LocalCutout service
        local = SURVEYS.loc[self.survey].local
        if local:
            self._cutout = LocalCutout(self)
            return

        # Otherwise use the appropriate API cutout service, use SkyView as default
        api_service = {
            "skymapper": SkymapperCutout,
            "panstarrs": PanSTARRSCutout,
            "decam": DECamCutout,
            "iphas": IPHASCutout,
            "mwats": MWATSCutout,  # Local?
        }.get(self.survey, SkyviewCutout)

        self._cutout = api_service(self)

        return

    def _determine_epoch(self):
        epoch = self.options.pop("epoch", False)

        fits_date_keys = [
            "MJD-OBS",
            "MJD",
            "DATE-OBS",
            "DATE",
        ]

        if not epoch:
            # Try each obs date FITS header keyword in sequence if epoch not provided directly
            for key in fits_date_keys:
                try:
                    epoch = self.header[key]
                    epochtype = "mjd" if "MJD" in key else None
                    break
                except KeyError:
                    continue

            # If epoch still not resolved, disable PM correction
            else:
                msg = f"Could not detect {self.survey} epoch, PM correction disabled."
                logger.warning(msg)
                self.obstime = None

                return

        else:
            epochtype = "mjd" if epoch > 3000 else "decimalyear"

        self.obstime = Time(epoch, format=epochtype)

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
        if self.options.get("grid", True):
            self.ax.coords.grid(color="white", alpha=0.5)

        if self.options.get("title", True):
            title = self.options.get("title", self.surveyname)
            self.ax.set_title(title, fontdict={"fontsize": 20, "fontweight": 10})

        self.set_xlabel("RA (J2000)")
        self.set_ylabel("Dec (J2000)")

        # Set compact or extended label / tick configuration
        if self.options.get("compact", True):
            tickcolor = (
                "k" if np.nanmax(np.abs(self.data)) == np.nanmax(self.data) else "gray"
            )

            lon = self.ax.coords[0]
            lat = self.ax.coords[1]

            lon.display_minor_ticks(True)
            lat.display_minor_ticks(True)

            lon.set_ticks(number=5)
            lat.set_ticks(number=5)

            self.ax.tick_params(axis="both", direction="in", length=5, color=tickcolor)

        # Change ticklabels to arcsecond offsets in offset frame
        if self.offsets:
            self.ax.coords[0].set_coord_type("longitude", coord_wrap=180 * u.deg)
            self.ax.coords[0].set_major_formatter("s")
            self.ax.coords[1].set_major_formatter("s")

        # Set colourmap normalisation
        self.norm = self._get_cmap_normalisation()
        self.cmap_label = r"Flux Density (mJy beam$^{-1}$)" if self.is_radio else ""

    def _get_cmap_normalisation(self):
        """Create colourmap normalisation for cutout map.

        User supplied parameters take precedence in order of:
        --maxnorm
        --vmin and/or --vmax

        or by default ZScaleInterval computes low-contrast limits which
        are made symmetric for Stokes V cutouts.
        """

        # Get min/max based upon ZScale with contrast parameter
        contrast = self.options.get("contrast", 0.2)
        vmin, vmax = ZScaleInterval(contrast=contrast).get_limits(self.data)

        # Make this symmetric if using Stokes V
        if self.stokes == "v":
            v = max(abs(vmin), abs(vmax))
            vmin = -v
            vmax = v

        # Override with user-supplied values if present
        if self.options.get("vmin") or self.options.get("vmax"):
            vmin = self.options.get("vmin", -2)
            vmax = self.options.get("vmax", 1)

        # Normalise with maximum value in data
        if self.options.get("maxnorm"):
            vmax = np.nanmax(self.data)
            vmin = None

        norm = ImageNormalize(
            self.data,
            interval=ZScaleInterval(),
            vmin=vmin,
            vmax=vmax,
            clip=True,
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
        d_str = f"{int(dms_tick[0])}"
        if len(d_str) == 1 or (len(d_str) == 2 and sign < 0):
            d_str = "s" + d_str
        m_str = f"{int(abs(dms_tick[1])):02d}"

        if sign < 0:
            s_str = f"{int(round(abs(dms_tick[2]) // 20) * 20 + 20):02d}"
        else:
            s_str = f"{int(round(abs(dms_tick[2]) // 20) * 20):02d}"
        if s_str == "60":
            s_str = "00"
            m_str = f"{int(m_str) + 1:02d}"

        # Pad axis label to offset individual ytick label character widths
        dec_str = d_str + m_str + s_str

        charlen = {"-": 0.65, "s": 0.075}
        zeropad = 0.8 + sum([charlen.get(c, 0.5) for c in dec_str])

        padlevel = self.options.get("ylabelpad", 5)
        self.ax.set_ylabel(ylabel, labelpad=padlevel - zeropad)

    def add_annotation(self, annotation, location="upper left", **kwargs):
        props = {
            "size": kwargs.get("size", 12),
            "color": kwargs.get("color", "firebrick"),
            "alpha": kwargs.get("alpha", 0.7),
            "weight": kwargs.get("weight", "heavy"),
            "family": "sans-serif",
            "usetex": False,
            "path_effects": kwargs.get("path_effects"),
        }

        text = AnchoredText(annotation, loc=location, frameon=False, prop=props)

        self.ax.add_artist(text)

    def add_contours(self, survey, position, shift_epoch=None, **kwargs):
        stokes = kwargs.get("stokes", "i")
        colors = kwargs.get("colors", "darkorange")
        label = kwargs.get("contourlabel", survey)

        contour_cutout = Cutout(
            survey,
            self.position,
            size=self.size,
            stokes=stokes,
        )

        # If using offset frame, align contours to data
        if self.offsets:
            r_crpix = contour_cutout.wcs.wcs_world2pix(self.ra, self.dec, 1)
            contour_cutout.wcs.wcs.crpix = [r_crpix[0], r_crpix[1]]
            contour_cutout.wcs.wcs.crval = [0, 0]

        datamax = np.nanmax(contour_cutout.data)
        perc_levels = np.array([0.3, 0.6, 0.9]) * datamax
        levels = kwargs.get("levels", perc_levels)

        if min(levels) > datamax:
            raise ValueError(
                "All contour levels exceed maximum data value of {datamax:.2f}."
            )

        if shift_epoch:
            contour_cutout.shift_coordinate_grid(position, shift_epoch)

        cs = self.ax.contour(
            contour_cutout.data,
            colors=colors,
            linewidths=self.options.get("contourwidth", 3),
            transform=self.ax.get_transform(contour_cutout.wcs),
            levels=levels,
        )

        # Add to contour artist collection for legend label access
        self.cs_dict[label] = Line2D([], [], color=colors)

    def add_cornermarker(self, marker):
        in_pixel_range = self.wcs.footprint_contains(marker.position)
        if not in_pixel_range:
            msga = (
                "Cornermarker will be disabled as RA and Dec are outside of data range."
            )
            logger.warning(msga)

            return

        self.ax.add_artist(marker.raline)
        self.ax.add_artist(marker.decline)

    def add_markers(self, coords, **kwargs):
        in_pixel_range = self.wcs.footprint_contains(coords)
        coords = coords[in_pixel_range]

        self.ax.scatter(
            coords.ra,
            coords.dec,
            transform=self.ax.get_transform("world"),
            **kwargs,
        )

    def add_source_ellipse(self):
        """Overplot dashed line ellipses for the nearest source within positional uncertainty."""

        # Add ellipse for source within positional uncertainty
        if self.plot_source:
            source_colour = "k" if self.stokes == "v" else "springgreen"
            position = SkyCoord(
                ra=self.source.ra_deg_cont, dec=self.source.dec_deg_cont, unit="deg"
            )
            self.sourcepos = EllipseSkyRegion(
                position,
                width=self.source.maj_axis * u.arcsec,
                height=self.source.min_axis * u.arcsec,
                angle=(self.source.pos_ang + 90) * u.deg,
            ).to_pixel(self.wcs)
            self.sourcepos.plot(
                ax=self.ax,
                facecolor="none",
                edgecolor=source_colour,
                ls=":",
                lw=1.5,
                zorder=10,
            )

        # Add ellipse for other components in the FoV
        if self.plot_neighbours:
            neighbour_colour = "k" if self.stokes == "v" else "rebeccapurple"
            for _, neighbour in self.neighbours.iterrows():
                position = SkyCoord(
                    ra=neighbour.ra_deg_cont,
                    dec=neighbour.dec_deg_cont,
                    unit="deg",
                )
                n = EllipseSkyRegion(
                    position,
                    width=neighbour.maj_axis * u.arcsec,
                    height=neighbour.min_axis * u.arcsec,
                    angle=(neighbour.pos_ang + 90) * u.deg,
                ).to_pixel(self.wcs)
                n.plot(
                    ax=self.ax,
                    facecolor="none",
                    edgecolor=neighbour_colour,
                    ls=":",
                    lw=2,
                    zorder=1,
                )

    def add_psf(self, frameon=True):
        try:
            self.bmaj = self.header["BMAJ"] * 3600
            self.bmin = self.header["BMIN"] * 3600
            self.bpa = self.header["BPA"]
        except KeyError:
            logger.warning(
                "Header does not contain PSF information, disabling PSF marker."
            )
            return

        try:
            cdelt = self.header["CDELT1"]
        except KeyError:
            cdelt = self.header["CD1_1"]

        facecolor = "k" if frameon else "white"
        edgecolor = "k"

        x = self.bmin / abs(cdelt) / 3600
        y = self.bmaj / abs(cdelt) / 3600

        psf = Ellipse(
            (0, 0),
            width=x,
            height=y,
            angle=self.bpa,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

        psf_transform_box = AuxTransformBox(self.ax.transData)
        psf_transform_box.add_artist(psf)

        box = AnchoredOffsetbox(
            child=psf_transform_box,
            loc="lower left",
            pad=0.5,
            borderpad=0.4,
            frameon=frameon,
        )
        self.ax.add_artist(box)

    def switch_to_offsets(self):
        """Transform WCS to a frame centred on the image."""

        cdelt1, cdelt2 = proj_plane_pixel_scales(self.wcs)
        ctype = self.wcs.wcs.ctype
        crpix = self.wcs.wcs_world2pix(self.ra, self.dec, 1)

        # Create new WCS as Skymapper does weird things with CDELT
        self.wcs = WCS(naxis=2)

        self.wcs.wcs.crpix = [crpix[0], crpix[1]]
        self.wcs.wcs.crval = [0, 0]
        self.wcs.wcs.cdelt = [-cdelt1, cdelt2]
        self.wcs.wcs.ctype = ctype

        if "radio" in dir(self):
            r_crpix = self.radio.wcs.wcs_world2pix(self.ra, self.dec, 1)
            self.radio.wcs.wcs.crpix = [r_crpix[0], r_crpix[1]]
            self.radio.wcs.wcs.crval = [0, 0]

        if self.plot_source:
            # Shift source ellipses
            source = SkyCoord(
                ra=self.source.ra_deg_cont,
                dec=self.source.dec_deg_cont,
                unit="deg",
            )

            source_ra, source_dec = self.position.spherical_offsets_to(source)

            self.source.ra_deg_cont = source_ra.deg
            self.source.dec_deg_cont = source_dec.deg

        if self.plot_neighbours:
            # Neighbours not quite transforming properly
            c = SkyCoord(
                ra=self.neighbours.ra_deg_cont,
                dec=self.neighbours.dec_deg_cont,
                unit="deg",
            )
            ras = []
            decs = []
            for neighbour in c:
                n_ra, n_dec = self.position.spherical_offsets_to(neighbour)
                ras.append(n_ra.deg)
                decs.append(n_dec.deg)

            self.neighbours.ra_deg_cont = ras
            self.neighbours.dec_deg_cont = decs

        self.offsets = True

    def shift_coordinate_grid(self, pm_coord, shift_epoch):
        """Shift WCS of pixel data to epoch based upon the proper motion encoded in pm_coord."""

        # Replace pixel data / WCS with copy centred on source
        contour_background = Cutout(
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
        except AttributeError:
            pm_ra = pm_coord.pm_ra_cosdec

        # Update CRVAL coordinates based on propagated proper motion
        orig_pos = SkyCoord(
            ra=self.wcs.wcs.crval[0] * u.deg,
            dec=self.wcs.wcs.crval[1] * u.deg,
            frame="icrs",
            distance=pm_coord.distance,
            pm_ra_cosdec=pm_ra,
            pm_dec=pm_coord.pm_dec,
            obstime=pm_coord.obstime,
        )
        newpos = orig_pos.apply_space_motion(shift_epoch)

        self.wcs.wcs.crval = [newpos.ra.deg, newpos.dec.deg]

        return

    def hide_coords(self, axis="both", ticks=True, labels=True):
        """Remove all coordinates and identifying information."""

        lon = self.ax.coords[0]
        lat = self.ax.coords[1]

        if axis in ["x", "both"]:
            lon.set_axislabel(" ")
            if labels:
                lon.set_ticklabel_visible(False)
            if ticks:
                lon.set_ticks_visible(False)

        if axis in ["y", "both"]:
            lat.set_axislabel(" ")
            if labels:
                lat.set_ticklabel_visible(False)
            if ticks:
                lat.set_ticks_visible(False)

    def query_simbad(self, epoch: Time = None):
        """Query SIMBAD and apply proper motion corrections."""

        if epoch is None and self.obstime is None:
            raise FITSException(
                "Date could not be inferred from data header, supply with epoch keyword."
            )

        # If obstime not set directly, check that it was set from FITS headers in get_cutout method
        obstime = self.obstime if epoch is None else epoch
        radius = max(180 * u.arcsec, self.size)

        # Make simbad query
        simbad, newpos = query_simbad(self.position, radius=radius, obstime=obstime)

        if simbad is None:
            return

        return simbad, newpos

    def plot(self, fig=None, ax=None, **kwargs):
        """Plot survey data and position overlay."""

        self._plot_setup(fig, ax)

        absmax = max(self.data.max(), self.data.min(), key=abs)
        rms = np.sqrt(np.mean(np.square(self.data)))

        logger.debug(f"Max flux in cutout: {absmax:.2f} mJy.")
        logger.debug(f"RMS flux in cutout: {rms:.2f} mJy.")

        self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm)

        if self.bar:
            self.fig.colorbar(
                self.im,
                label=self.cmap_label,
                ax=self.ax,
            )

        self.add_source_ellipse()

    def save(self, path, fmt="png"):
        """Save figure with tight bounding box."""
        self.fig.savefig(path, format=fmt, bbox_inches="tight")

    def savefits(self, path):
        """Save FITS cutout."""

        header = self.wcs.to_header()
        header["BUNIT"] = "Jy/beam"
        hdu = fits.PrimaryHDU(data=self.data / 1e3, header=header)
        hdu.writeto(path)

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel)

    def set_ylabel(self, ylabel, align=False):
        if align:
            self._align_ylabel(ylabel)
        else:
            self.ax.set_ylabel(ylabel)
