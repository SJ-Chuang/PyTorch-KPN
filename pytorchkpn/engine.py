from pytorchkpn.registry import MODEL_REGISTRY
from torchvision import transforms as T
from .data import build_train_loader, build_val_loader
from .utils import logger

import torchvision.transforms.functional as TF
import os, torch, json, random

def RandomAug(image, target):
    """
    Apply Augmentation
    Args:
        image (torch.tensor): input image
        target (torch.tensor): target heatmap
    Returns:
        augmented image and target heatmap
    """
    if random.random() > 0.5:
        image = TF.hflip(image)
        target = TF.hflip(target)
    
    if random.random() > 0.5:
        image = TF.vflip(image)
        target = TF.vflip(target)
    
    return image, target

class DefaultPredictor:
    """
    Default predictor
    Args:
        cfg (CfgNode): the full config to be used. 
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = MODEL_REGISTRY.get(self.cfg.MODEL.NAME)(
            self.cfg.MODEL.INPUT_SHAPE, self.cfg.MODEL.NUM_CLASSES).to(self.cfg.MODEL.DEVICE)
        
        self.model.eval()
        assert self.cfg.MODEL.WEIGHTS, "weights path must be given."
        self.model.load_state_dict(torch.load(self.cfg.MODEL.WEIGHTS))
        print(f"Load weights from {self.cfg.MODEL.WEIGHTS}")

    def __call__(self, img_bgr):
        """
        Args:
            img_bgr (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            detections (List): detected keypoints.
        """
        with torch.no_grad():
            image = T.Resize(self.cfg.MODEL.INPUT_SHAPE)(
                torch.tensor(img_bgr).permute(2, 0, 1).float().to(self.cfg.MODEL.DEVICE))
            image = image.unsqueeze(0)
            hm = torch.clamp(self.model(image).sigmoid_(), min=1e-4, max=1-1e-4)
            hmax = torch.nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
            keypoints = hm * (hmax == hm).float()
            _, n_classes, hm_h, hm_w = keypoints.shape
            C, Y, X = torch.where(keypoints[0] > self.cfg.TEST.THRESH)
            probs = keypoints[0, C, Y, X].cpu().numpy()
            detections = [[int(c), prob, int(x), int(y)] for c, prob, x, y in zip(C, probs, X, Y)]
        
        return detections
            


class DefaultTrainer:
    """
    Default trainer
    Args:
        cfg (CfgNode): the full config to be used.
    """
    def __init__(self, cfg):
        self.model = self.build_model(cfg).to(cfg.MODEL.DEVICE)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.criterion = self.build_loss_func()
        self.aug = self.build_aug_func()
        self.train_loader = self.build_train_loader(cfg)
        self.val_loader = self.build_val_loader(cfg)
        self.epoch = cfg.SOLVER.EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg.clone()
        if self.cfg.MODEL.WEIGHTS:
            self.model.load_state_dict(torch.load(self.cfg.MODEL.WEIGHTS))
            print(f"Load weights from {self.cfg.MODEL.WEIGHTS}")
    
    def train(self):
        open(os.path.join(self.output_dir, "kpn_cfg.yaml"), "w").write(self.cfg.dump())
        lowest_val_loss, history = float("inf"), [[], []]
        for e in range(self.epoch):
            self.model.train()
            train_loss = 0
            for b, (x, target_hm) in enumerate(self.train_loader):
                x, target_hm = self.aug(x, target_hm)
                self.optimizer.zero_grad()
                y = self.model(x)
                pred_hm = torch.clamp(y.sigmoid_(), min=1e-4, max=1-1e-4)
                loss = self.criterion(pred_hm, target_hm)
                loss.backward()
                self.optimizer.step()
                train_loss += loss
                print(f"batch {b+1}/{len(self.train_loader)}: Train Loss = {loss:<.5f}", end="\r")
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, target_hm in self.val_loader:
                    y = self.model(x)
                    pred_hm = torch.clamp(y.sigmoid_(), min=1e-4, max=1-1e-4)
                    val_loss += self.criterion(pred_hm, target_hm)
            console = f"Epoch {e+1:>3}/{self.epoch} Train Loss: {train_loss/len(self.train_loader):<.5f}, Val Loss: {val_loss/len(self.val_loader):<.5f}"
            
            if lowest_val_loss > val_loss:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_kpn.pth"))
                console += " (best saved)"
                lowest_val_loss = val_loss
            logger.info(console)
            history[0].append(float(train_loss/len(self.train_loader)))
            history[1].append(float(val_loss/len(self.val_loader)))
            json.dump(history, open(os.path.join(self.output_dir, "history.json"), "w"), indent=2)
        
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "last_kpn.pth"))
        
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module
        """
        model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg.MODEL.INPUT_SHAPE, cfg.MODEL.NUM_CLASSES)
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
    def build_loss_func(cls):
        """
        Returns:
            callable loss function
            
        Overwrite it if you'd like a different loss function.
        """
        return torch.nn.MSELoss()
    
    @classmethod
    def build_aug_func(cls):
        """
        Returns:
            callable augmentation function
            
        Overwrite it if you'd like different augmentation styles.
        """
        return RandomAug
        
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
