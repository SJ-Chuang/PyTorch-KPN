from yacs.config import CfgNode as CN
from datetime import datetime

_C = CN()

_C.OUTPUT_DIR = datetime.now().strftime("kpn-%Y%m%d-%H%M%S")

_C.MODEL = CN()
_C.MODEL.NAME = "LRASPP"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.INPUT_SHAPE = (416, 416)
_C.MODEL.WEIGHTS = ""

_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.VAL = ()
_C.DATASETS.TEST = ()
_C.DATASETS.SIGMA = 3

_C.SOLVER = CN()
_C.SOLVER.EPOCH = 300
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.BASE_LR = 0.0008
_C.SOLVER.BETAS = (0.9, 0.999)
_C.SOLVER.EPS = 1e-8
_C.SOLVER.WEIGHT_DECAY = 0

def get_cfg():
    return _C.clone()