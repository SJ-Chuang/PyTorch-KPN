from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from typing import List, Tuple, Dict
from collections import UserDict
from .utils import Point2Heatmap
import numpy as np
import cv2, torch

__all__ = [
    "DatasetCatalog", "KPDataset",
    "build_train_loader", "build_val_loader"
]

class _DatasetCatalog(UserDict):
    """
    A dictionary that stores information about the datasets and how to obtain them.
    """
    def register(self, name: str, func):
        """
        Register a dataset.
        Args:
            name (str): name of the dataset.
            func (callable): a callable which takes no arguments and returns a list of dicts.
                dict format:
                    {"image": "path/to/an/image", "annotations": [[x0, y0], [x1, y1]]}
        """
        assert callable(func), "You must register a function with `DatasetCatalog.register`."
        assert name not in self, f"Dataset '{name}' is already registered."
        self[name] = func
    
    def get(self, name):
        """
        Get a registered function.
        Args:
            name (str): the name a registered dataset.

        Returns:
            List[Dict]: list of datasets.
        """
        assert name in self, f"Dataset '{name}' is not registered."
        return self[name]()
    
    def remove(self, name):
        """
        Remove a dataset from DatasetCatalog.
        Args:
            name (str): the name a registered dataset.
        """
        assert name in self, f"Dataset '{name}' is not registered. It cannot be removed."
        self.pop(name)
        
class KPDataset(Dataset):
    def __init__(self, data_list: List, input_shape: Tuple[int]=(416, 416), sigma: int=3, device="cuda"):
        """
        data_list (List): list of keypoint dataset
        input_shape (Tuple[int]): input shape of the model.
        sigma (int): sigma of gaussian kernel.
        device (str): device to use
        """
        self.data_list = data_list
        self.input_shape = input_shape
        self.sigma = sigma
        self.device = device
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        img = cv2.imread(data["image"])
        assert img is not None, f"{data['image']} is not found"
        hm = Point2Heatmap(data["annotations"], img.shape[:2], sigma=self.sigma)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float().to(self.device)
        hm_tensor = torch.tensor(hm).unsqueeze(0).float().to(self.device)
        return T.Resize(self.input_shape)(img_tensor), T.Resize(self.input_shape)(hm_tensor)

def build_train_loader(cfg):
    """
    Build a data loader for training
    Args:
        cfg (CfgNode): the full config to be used.
    
    Returns:
        Pytorch data loader
    """
    data_list = []
    assert len(cfg.DATASETS.TRAIN) > 0, "Must define at least one dataset in cfg.DATASETS.TRAIN"
    for name in cfg.DATASETS.TRAIN:
        data_list.extend(DatasetCatalog.get(name))
    dataset = KPDataset(data_list, cfg.MODEL.INPUT_SHAPE, cfg.DATASETS.SIGMA, device=cfg.MODEL.DEVICE)
    return DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)

def build_val_loader(cfg):
    """
    Build a data loader for validation
    Args:
        cfg (CfgNode): the full config to be used.
    
    Returns:
        Pytorch data loader
    """
    data_list = []
    assert len(cfg.DATASETS.TRAIN) > 0, "Must define at least one dataset in cfg.DATASETS.VAL"
    for name in cfg.DATASETS.VAL:
        data_list.extend(DatasetCatalog.get(name))
    dataset = KPDataset(data_list, cfg.MODEL.INPUT_SHAPE, cfg.DATASETS.SIGMA, device=cfg.MODEL.DEVICE)
    return DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)

DatasetCatalog = _DatasetCatalog()