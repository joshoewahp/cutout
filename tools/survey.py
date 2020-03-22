import re

class Survey:

    def __init__(self, survey, name, radius, radio, cutout, **kwargs):

        self.survey = survey
        self.name = name
        self.radius = radius
        self.radio = radio
        self.cutout = cutout
        self.local = any([s in self.survey for s in ['racs', 'vast', 'vlass', 'gw']])
        self.cat = kwargs.get("cat", None)
        self.sv = kwargs.get("sv", None)
        self.nu = kwargs.get("nu", None)
        self.year = kwargs.get("year", None)
        self.rms = kwargs.get("rms", None)
        self.color = kwargs.get("color", None)
        self.marker = kwargs.get("marker", None)
        self.flux_col = kwargs.get("flux_col", None)
        self.eflux_col = kwargs.get("eflux_col", None)
        self.pos_err = kwargs.get("pos_err", None)
        self.dec_min = kwargs.get("dec_min", None)
        self.dec_max = kwargs.get("dec_max", None)
        self.gal_lim = kwargs.get("gal_lim", None)
        self.patchy = kwargs.get("patchy", None)

        if any([s in self.survey for s in ['racs', 'vast', 'gw']]):
            self._set_local_paths()
        else:
            self.images = None
            self.selavy = None
            
    def _set_local_paths(self):

        if 'gw' in self.survey:
            direc = "/import/ada1/aste7152/askap/GW/S190814bv/"
            epoch = {'1': 'askap_obs_20190816/',
                     '2': 'askap_obs_20190823/',
                     '3': 'askap_obs_20190916/',
                     '4': 'askap_obs_20191107/'}.get(self.survey[-2])
            self.images = direc + epoch
            self.selavy = direc + epoch
            
        else:
            base = "/import/ada1/askap/"
            pol = self.survey[-1]

            if 'racs' in self.survey:
                direc = base + "RACS/aug2019_reprocessing/COMBINED_MOSAICS/"
                self.images = direc + 'I_mosaic_1.0/'
                self.selavy = direc + 'racs_cat/'
                
            elif 'vast' in self.survey:
                pattern = re.compile(r'vastp(\d{1,2})(x*)[IV]')
                epoch = pattern.sub(r'\1', self.survey)
                epoch = epoch if len(epoch) == 2 else f'0{epoch}'
                partial = pattern.sub(r'\2', self.survey)
                direc = base + f"PILOT/EPOCH{epoch}{partial}/COMBINED/"
                self.images = direc + f"STOKES{pol}_IMAGES/"
                self.selavy = direc + f"STOKES{pol}_SELAVY/"


if __name__ == '__main__':

    
    racs = Survey('vastp9I', 'RACS (I)', 0.025, nu=887.49, rms=0.25, year=2019.5,
          	  color='firebrick', marker='full', flux_col='flux_int',
          	  eflux_col='flux_int_err', flux_mult=1, pos_err=2, dec_min=-90, dec_max=40,
          	  gal_lim=0, patch=False, radio=True, cutout=True)

    print(racs.images)
    print(racs.selavy)
