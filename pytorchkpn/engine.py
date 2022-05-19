from pytorchkpn.registry import MODEL_REGISTRY
from .data import build_train_loader, build_val_loader
from .utils import logger
import torch

class DefaultTrainer:
    """
    Default trainer
    Args:
        cfg (CfgNode): the full config to be used.
    """
    def __init__(self, cfg):
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        train_loader = self.build_train_loader(cfg)
    
    def train(self):
        ### TODO
        pass
        
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module
        """
        model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg.MODEL.INPUT_SHAPE)
        logger.info(f"Model: {model}")
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        Overwrite it if you'd like a different optimizer.
        """
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=cfg.SOLVER.EPS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        Overwrite it if you'd like a different data loader.
        """
        return build_train_loader(cfg)

    @classmethod
    def build_val_loader(cls, cfg):
        """
        Returns:
            iterable

        Overwrite it if you'd like a different data loader.
        """
        return build_val_loader(cfg)
