from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )
        self.parser.add_argument(
            '--which_timestamp',
            type=str,
            default='',
            help='which timestamp to load? set to latest to use latest cached model'
        )
        self.parser.add_argument(
            '--store_timestamp',
            type=str,
            default=None,
            help='Which timestamp folder to load from'
        )
        self.parser.add_argument(
            '--name',
            type=str,
            default=None,
            help='Name of save folder'
        )
        self.parser.add_argument(
            '--use_object_model',
            action='store_true'
        )
        self.is_train = False