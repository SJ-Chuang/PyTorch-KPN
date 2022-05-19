from pytorchkpn.engine import DefaultTrainer
from pytorchkpn.data import DatasetCatalog
from pytorchkpn.config import get_cfg
import os, json

class DataList:
    def __init__(self, image_path):
        self.image_path = image_path
        
    def get_datalist(self):
        data_list = []
        for p in self.image_path:
            pre, ext = os.path.splitext(p)
            label = json.load(open(pre+".json"))
            data_list.append({
                "image": p,
                "annotations": [np.mean(shape["points"], 0) for shape in label["shapes"]]
            })
        return data_list

DatasetCatalog.register("train", DataList(open("../train.txt").read().splitlines()).get_datalist)
DatasetCatalog.register("val", DataList(open("../val.txt").read().splitlines()).get_datalist)

cfg = get_cfg()
cfg.DATASETS.TRAIN = ("train", )
cfg.DATASETS.VAL = ("val")
trainer = DefaultTrainer(cfg)