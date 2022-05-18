from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "LRASPP"

def get_cfg():
    return _C.clone()