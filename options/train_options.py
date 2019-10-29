from .base_options import BaseOptions 

class TrainOptions(BaseOptions):
    # this class include training options

    def initialize(self):
        parser = BaseOptions.initialize(self)
        # add some training options here
        self.parser.add_argument("--isTrain", type=bool, default=True)
        self.parser.add_argument("--dataroot", type=str, default='datasets/DJI_hd/train')
        self.parser.add_argument("--noagument", action='store_true')
        self.parser.add_argument("--nomultiscale", action='store_true')
        self.parser.add_argument("--load", action='store_true')
        self.parser.add_argument("--iter_name", type=str, default='latest.pth')
        self.parser.add_argument("--lr", type=float, default=1e-4)

        return parser