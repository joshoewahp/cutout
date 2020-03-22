import logging
try:
    import colorlog
    color = True
except ImportError as e:
    print(e)
    color = False

levels = {'DEBUG': logging.DEBUG,
          'INFO': logging.INFO,
          'WARNING': logging.WARNING,
          'ERROR': logging.ERROR,
          'CRITICAL': logging.CRITICAL}


class Logger:

    def __init__(self, name, filename=None, **kwargs):

        self.name = name
        self.filename = filename
        self.level = levels[kwargs.get('level', 'DEBUG')]
        self.stream = kwargs.get('stream')
        self.streamlevel = levels[kwargs.get('streamlevel', 'INFO')]
        self.logger = logging.getLogger(self.name)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self._setup()

    def _setup(self):
        self.logger.setLevel(self.level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        if self.filename:
            self.file_handler = logging.FileHandler(self.filename)
            self.file_handler.setFormatter(self.formatter)
            self.file_handler.setLevel(self.level)
            self.logger.addHandler(self.file_handler)

        if color:
            self.colorformatter = colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s%(reset)s %(asctime)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                reset=True,
                log_colors={'DEBUG': 'cyan',
                            'INFO': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'red,bg_white', })
            self.stream_handler = colorlog.StreamHandler()
            self.stream_handler.setFormatter(self.colorformatter)
        else:
            self.stream_handler = logging.StreamHandler()

        self.stream_handler.setLevel(self.streamlevel)
        self.logger.addHandler(self.stream_handler)
