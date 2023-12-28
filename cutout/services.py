import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.error import HTTPError

import astropy.units as u
import requests
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from astroutils.io import FITSException, find_fields, get_config, get_surveys
from astroutils.source import SelavyCatalogue

warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)

config = get_config()
aux_path = Path(config["DATA"]["aux_path"])
vlass_path = Path(config["DATA"]["vlass_path"])
cutout_cache = aux_path / "cutouts"

SURVEYS = get_surveys()
SURVEYS.set_index("survey", inplace=True)

logger = logging.getLogger(__name__)


class CutoutService(ABC):
    def __init__(self, cutout):
        self.plot_source = False
        self.plot_neighbours = False
        self.hdulindex = 0

        self.local = False
        self.source = None

        self.fetch_data(cutout)
        self.fetch_sources(cutout)

        self._make_cutout(cutout)

    def _set_cache_path(self, cutout):
        """Create a cache directory and filename pattern for externally retrieved image data."""

        if not os.path.exists(cutout_cache / cutout.survey):
            msg = f"{cutout_cache}/{cutout.survey} cutout directory does not exist, creating."
            logger.info(msg)
            os.makedirs(cutout_cache / cutout.survey)

        band = f"_{cutout.band}" if "band" in cutout.__dict__ else ""
        path = f"{cutout.size.value*60:.4f}arcmin_{cutout.ra:.4f}_{cutout.dec:.4f}{band}.fits"
        self.filepath = cutout_cache / cutout.survey / path

    @abstractmethod
    def fetch_data(self, cutout, *args):
        """Fetch cutout data from source and set filepath attribute accordingly."""

    @abstractmethod
    def fetch_sources(self, cutout, *args):
        """Fetch sources within field of view of cutout and set to components attribute."""

    def _find_neighbours(self, cutout):
        """Set source and neighbour attributes from component list."""

        if self.components.empty:
            logger.debug("No nearby components found.")
            return

        self.components.sort_values("d2d", inplace=True)

        # Set nearest component within positional uncertainty
        # as the target source and all others as neighbours
        pos_err = (
            cutout.size if self.local else SURVEYS.loc[cutout.survey].pos_err * u.arcsec
        )

        if self.components.iloc[0].d2d * u.arcsec < pos_err * 5:
            self.source = self.components.iloc[0].copy()
            self.neighbours = self.components.iloc[1:].copy()
            self.plot_source = True
        else:
            self.source = None
            self.neighbours = self.components

        self.plot_neighbours = cutout.options.get("neighbours", True)

        if self.source is not None:
            logger.debug(f"Source: \n{self.source}")

        if not self.neighbours.empty:
            nn = self.neighbours.iloc[0]
            neighbour_view = self.neighbours[
                [
                    "ra_deg_cont",
                    "dec_deg_cont",
                    "maj_axis",
                    "min_axis",
                    "flux_peak",
                    "flux_int",
                    "rms_image",
                    "d2d",
                ]
            ]
            logger.debug(
                f"Nearest neighbour coordinates: \n{nn.ra_deg_cont} {nn.dec_deg_cont}"
            )
            logger.debug(f"Nearest 5 Neighbours \n{neighbour_view.head()}")

    def _make_cutout(self, cutout):
        with fits.open(self.filepath) as hdul:
            logger.debug(f"Making cutout from FITS image located at:\n{self.filepath}")

            header, data = hdul[self.hdulindex].header, hdul[self.hdulindex].data
            wcs = WCS(header, naxis=2)

        try:
            cutout2d = Cutout2D(data[0, 0, :, :], cutout.position, cutout.size, wcs=wcs)
        except IndexError:
            cutout2d = Cutout2D(data, cutout.position, cutout.size, wcs=wcs)
        except NoOverlapError:
            field = cutout.options.get("fieldname")
            raise FITSException(
                f"Field {field} does not contain position {cutout.position.ra:.2f}, {cutout.position.dec:.2f}"
            )

        self.data = cutout2d.data * 1000
        self.wcs = cutout2d.wcs
        self.header = self.wcs.to_header()
        for key in ["BMAJ", "BMIN", "BPA"]:
            if header.get(key):
                self.header[key] = header[key]


class RawCutout(CutoutService):
    """Fetch cutout data from specific local FITS image path."""

    def fetch_data(self, cutout):
        self.filepath = cutout.survey
        self.hdulindex = 0

    def fetch_sources(self, cutout):
        try:
            selavy = SelavyCatalogue(cutout.selavy)
            self.components = selavy.cone_search(cutout.position, 0.5 * cutout.size)
        except FileNotFoundError:
            logger.error(
                f"Selavy files not found at {cutout.selavy}, disabling source ellipses."
            )
            return

        self.local = True
        self._find_neighbours(cutout)

        return


class LocalCutout(CutoutService):
    """Fetch cutout data via local FITS images (e.g. RACS / VLASS)."""

    def fetch_data(self, cutout):
        is_vlass = "vlass" in cutout.survey
        tiletype = None if is_vlass else cutout.tiletype
        fields = find_fields(cutout.position, cutout.survey, tiletype=tiletype)

        if fields.empty:
            raise FITSException(
                f"No fields located at {cutout.position.ra:.2f}, {cutout.position.dec:.2f}"
            )

        # Choose field if specified by user, or default to nearest field centre
        fieldname = cutout.options.get("fieldname")
        sbid = cutout.options.get("sbid")
        fields.sort_values("dist_field_centre", inplace=True)

        logger.debug(f"Fields:\n{fields}")

        if not is_vlass:
            if fieldname:
                fields = fields[fields.field == fieldname]
            if sbid:
                fields = fields[fields.sbid == sbid]

        try:
            self.field = fields.iloc[0]
        except IndexError:
            raise FITSException(f"Field {fieldname} not located")

        self.filepath = self.field[f"{cutout.stokes}_path"]

    def fetch_sources(self, cutout):
        """Locate target and neighbouring selavy components within positional uncertainty and FoV."""

        # VLASS source catalogues not yet available
        if "vlass" in cutout.survey:
            return

        try:
            selavy = SelavyCatalogue.from_params(
                epoch=cutout.survey,
                fields=self.field.field,
                sbids=self.field.sbid,
                stokes=cutout.stokes,
                tiletype=cutout.tiletype,
            )
            self.components = selavy.cone_search(cutout.position, 0.5 * cutout.size)
        except FileNotFoundError:
            logger.debug(
                f"Selavy files not found for {cutout.survey} Stokes {cutout.stokes}, disabling source ellipses."
            )
            return

        self._find_neighbours(cutout)


class SkymapperCutout(CutoutService):
    """Fetch cutout from Skymapper API."""

    def fetch_data(self, cutout):
        self._set_cache_path(cutout)

        # Query Skymapper cutout service for links to cutout in each band
        route = "https://api.skymapper.nci.org.au/aus/siap/dr3/"
        params = "query?POS={:.5f},{:.5f}&SIZE={:.3f}&BAND=all&RESPONSEFORMAT=VOTABLE&VERB=3&INTERSECT=covers"

        sm_query_link = route + params.format(cutout.ra, cutout.dec, cutout.size.value)

        table = requests.get(sm_query_link)
        with open(f"{self.filepath}.xml", "wb") as f:
            f.write(table.content)
        links = Table.read(f"{self.filepath}.xml", format="votable")[
            [
                "band",
                "get_image",
            ]
        ].to_pandas()
        os.system(f"rm {self.filepath}.xml")

        # Check cutouts exist at position
        impos = f"{cutout.ra:.2f}, {cutout.dec:.2f}"
        assert len(links) > 0, f"No Skymapper image at {impos}"
        assert (
            "Time-out" not in links.iloc[0]
        ), f"Skymapper Gateway Time-out for image at {impos}"

        # Select image link for requested band
        link = links[links.band == cutout.band].iloc[0].get_image
        img = requests.get(link)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(img.content)

    def fetch_sources(self, cutout):
        return


class PanSTARRSCutout(CutoutService):
    """Fetch cutout data via PanSTARRS DR2 API."""

    def fetch_data(self, cutout):
        self._set_cache_path(cutout)

        # Query PanSTARRS cutout service for links to cutout in each band
        # Pixel size is 0.5 arcsec, and we want 2x cutout size passed
        # to the API for the full image width in pixels
        pixsize = int(cutout.size.value * 3600 / 0.5 / 2)
        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = f"{service}?ra={cutout.ra}&dec={cutout.dec}&size={pixsize}&format=fits&filters=grizy"
        table = Table.read(url, format="ascii").to_pandas()

        # Check cutouts exist at position
        msg = f"No PanSTARRS1 image at {cutout.ra:.2f}, {cutout.dec:.2f}"
        assert not table.empty, msg

        # Select image link for requested band
        urlbase = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
            f"?ra={cutout.ra}&dec={cutout.dec}&size={pixsize}&format=fits&red="
        )
        link = urlbase + table[table["filter"] == cutout.band].iloc[0].filename
        img = requests.get(link, allow_redirects=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(img.content)

    def fetch_sources(self, cutout):
        return


class DECamCutout(CutoutService):
    """Fetch cutout data via DECam LS API."""

    def fetch_data(self, cutout):
        self._set_cache_path(cutout)

        pixsize = int(cutout.size.value * 3600 / 0.262)
        if pixsize > 512:
            pixsize = 512
            max_size = pixsize * 0.262 / 3600
            logger.debug(f"Using maximum DECam LS cutout pixsize of {max_size:.3f} deg")

        link = f"https://www.legacysurvey.org/viewer/fits-cutout?ra={cutout.ra}&dec={cutout.dec}"
        link += f"&size={pixsize}&layer=ls-dr9&pixscale=0.262&bands={cutout.band}"
        img = requests.get(link)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(img.content)

    def fetch_sources(self, cutout):
        return


class IPHASCutout(CutoutService):
    """Fetch cutout data via IPHAS API using Vizier query."""

    def fetch_data(self, cutout):
        self._set_cache_path(cutout)
        self.hdulindex = 1

        # Query Vizier for observations at position
        v = Vizier(
            columns=["_r", "_RAJ2000", "_DEJ2000", "rDetectionID", "fieldGrade", "*"]
        )
        cat = SURVEYS.loc[cutout.survey]["vizier"]

        table = v.query_region(cutout.position, radius=cutout.size, catalog=cat)

        # Check that iPHaS image exists at position
        exc = FITSException(
            f"No IPHAS image at {cutout.position.ra:.2f}, {cutout.position.dec:.2f}"
        )
        if len(table) == 0:
            raise exc
        table = table[0].to_pandas().sort_values("_r")
        if table.empty:
            raise exc

        # Remove poor quality observations
        table = table[(table.rDetectionID != "") & (table.fieldGrade != "D")]

        # Build image link from Detection ID
        det_id = table.iloc[1]["rDetectionID"]
        run = det_id.split("-")[0]
        ccd = det_id.split("-")[1]
        link = f"http://www.iphas.org/data/images/r{run[:3]}/r{run}-{ccd}.fits.fz"

        img = requests.get(link)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                f.write(img.content)

    def fetch_sources(self, cutout):
        return


class MWATSCutout(CutoutService):
    def __init__(self, cutout):
        self.plot_source = False
        self.plot_neighbours = False

        mwats_selavy_path = SURVEYS.loc[cutout.survey][f"selavy_path_i"]
        self.mwats = SelavyCatalogue.from_aegean(mwats_selavy_path + "mwats_raw.parq")

        self.fetch_sources(cutout)
        self.fetch_data(cutout)

        self._make_cutout(cutout)

    def fetch_data(self, cutout):
        components = self.mwats.cone_search(cutout.position, 1 * u.degree)

        if components.empty:
            raise FITSException(f"No MWATS images at {cutout.ra:.4f} {cutout.dec:.4f}")

        nearest = components.sort_values("d2d", ascending=True).iloc[0]
        mwats_image_path = SURVEYS.loc[cutout.survey][f"image_path_{cutout.stokes}"]

        self.filepath = mwats_image_path + nearest.image.replace(
            "_I", f"_{cutout.stokes.upper()}"
        )

    def fetch_sources(self, cutout):
        self.components = self.mwats.cone_search(cutout.position, 0.5 * cutout.size)

        self._find_neighbours(cutout)


class SkyviewCutout(CutoutService):
    """Fetch cutout data via SkyView API."""

    def fetch_data(self, cutout):
        self._set_cache_path(cutout)

        if not os.path.exists(self.filepath):
            skyview_key = SURVEYS.loc[cutout.survey].skyview

            try:
                sv = SkyView()
                hdul = sv.get_images(
                    position=cutout.position,
                    survey=[skyview_key],
                    radius=cutout.size,
                    show_progress=False,
                )[0][0]
            except IndexError:
                raise FITSException("Skyview image list returned empty.")
            except ValueError:
                raise FITSException(f"{cutout.survey} is not a valid SkyView survey.")
            except HTTPError:
                raise FITSException("No response from Skyview server.")

            with open(self.filepath, "wb") as f:
                hdul.writeto(f)

    def fetch_sources(self, cutout):
        return
