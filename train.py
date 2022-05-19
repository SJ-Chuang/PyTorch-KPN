from pytorchkpn.engine import DefaultTrainer
from pytorchkpn.data import DatasetCatalog
from pytorchkpn.config import get_cfg
from pytorchkpn.evaluation import do_evaluation
import numpy as np
import cv2, os, json
import torch

class DataList:
    def __init__(self, image_path):
        self.image_path = image_path
        
    def get_datalist(self):
        data_list = []
        for p in self.image_path:
            h, w = cv2.imread(p).shape[:2]
            pre, ext = os.path.splitext(p)
            label = json.load(open(pre+".json"))
            data_list.append({
                "image": p,
                "height": h,
                "width": w,
                "annotations": [
                    {
                        "keypoint": np.mean(shape["points"], 0),
                        "category_id": 0
                    } for shape in label["shapes"]
                ]
                
                
            })
        return data_list

DatasetCatalog.register("train", DataList(open("../train.txt").read().splitlines()).get_datalist)
DatasetCatalog.register("val", DataList(open("../val.txt").read().splitlines()).get_datalist)

cfg = get_cfg()
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.VAL = ("val",)
cfg.SOLVER.EPOCH = 300

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.train()

model = trainer.model
model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best_kpn.pth")))

do_evaluation(cfg, model, dataset_name="val")