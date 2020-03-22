import configparser
import glob
import os
import re
import time
import numpy as np
import pandas as pd
from astropy import wcs
from astropy.io import fits
from pathlib import Path

config = configparser.ConfigParser()
config.read('./config/config.ini')
aux_path = config['DATA']['aux_path']


def make_field_csv(epoch):
    """Generate metadata csv for dataset fields."""
    image_path = config['DATA'][f'stokesI_path_{epoch}']

    vals = []
    for field in glob.glob(image_path + '*.fits'):
        pattern = re.compile(r'\S*(\d{4}[-+]\d{2}[AB])\S*')
        sbidpattern = re.compile(r'\S*(SB\d{4,5})\S*')
        name = pattern.sub(r'\1', field)
        if '-mm' in epoch:
            sbid = 'SBXXX'
        else:
            sbid = sbidpattern.sub(r'\1', field)

        try:
            with fits.open(field) as hdul:
                header = hdul[0].header
                w = wcs.WCS(header, naxis=2)
                size_x = header["NAXIS1"]
                size_y = header["NAXIS2"]

                central_coords = [[size_x / 2., size_y / 2.]]
                centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)

                vals.append({'field': name,
                             'sbid': sbid,
                             'cr_ra': header['CRVAL1'],
                             'cr_dec': header['CRVAL2'],
                             'cr_ra_pix': centre[0][0],
                             'cr_dec_pix': centre[0][1]})

        except Exception as e:
            print(e)
            raise
            vals.append({'field': name, 'sbid': sbid, 'cr_ra': np.nan, 'cr_dec': np.nan,
                         'cr_ra_pix': np.nan, 'cr_dec_pix': np.nan})

    df = pd.DataFrame(vals)
    df = df.dropna()

    print(df)
    df.to_csv(aux_path + f'{epoch}_fields.csv', index=False)

    return


def make_vlass_fields(base_dir):

    pattern = re.compile(r'\S+(J\d{6}[-+]\d{6})\S+')
    fields = list(Path(base_dir).rglob("*subim.fits"))
    names = [f.parts[-1] for f in fields]
    df = pd.DataFrame({'filename': names,
                       'coord': [pattern.sub(r'\1', name) for name in names],
                       'epoch': [f.parts[4] for f in fields],
                       'tile': [f.parts[5] for f in fields],
                       'image': [f.parts[6] for f in fields]})

    vals = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(idx)

        with fits.open(f'{base_dir}{row.epoch}/{row.tile}/{row.image}/{row.filename}') as hdul:
            header = hdul[0].header
            w = wcs.WCS(header, naxis=2)
            size_x = header["NAXIS1"]
            size_y = header["NAXIS2"]

            central_coords = [[size_x / 2., size_y / 2.]]
            centre = w.wcs_pix2world(np.array(central_coords, np.float_), 1)

            vals.append({'image': row.image,
                         'cr_ra': header['CRVAL1'],
                         'cr_dec': header['CRVAL2'],
                         'cr_ra_pix': centre[0][0],
                         'cr_dec_pix': centre[0][1],
                         'date': header['DATE-OBS']})

    df = df.merge(pd.DataFrame(vals), on='image')

    print(df)
    df.to_csv(aux_path + 'vlass_fields.csv', index=False)


def make_raw_cat(epoch, pol):
    selavy_path = config['DATA'][f'selavy{pol}_path_{epoch}']

    components = [selavy_path + c for c in os.listdir(selavy_path) if
                  'components.txt' in c]

    pattern = re.compile(r'\S*(\d{4}[+-]\d{2}[AB])\S*')
    sbidpattern = re.compile(r'\S*(SB\d{4,5})\S*')

    csvs = []
    for csv in components:
        sign = -1 if csv.split('/')[-1][0] == 'n' else 1
        sbid = 'SBXXXX' if '-mm' in epoch else sbidpattern.sub(r'\1', csv)

        df = pd.read_fwf(csv, skiprows=[1, ])
        df.insert(0, 'sbid', sbid)
        df.insert(0, 'field', pattern.sub(r'\1', csv))
        df['sign'] = sign
        csvs.append(df)
    df = pd.concat(csvs, ignore_index=True, sort=False)
    df.to_csv(aux_path + f'{epoch}_raw_selavy_cat.csv', index=False)

    return


def table2df(table):
    """Clean conversion of Astropy table to DataFrame. """
    df = table.to_pandas()
    return df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)


def timeit(func):
    """Decorator to time func"""

    def _timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print('{} took {:2.2f} ms'.format(func.__name__, (te - ts) * 1000))
        return result

    return _timed
