from .data import build_test_loader
import numpy as np
import torch

__all__ = [
    "APEvaluator", "KPEvaluator",
    "do_evaluation"
]

class APEvaluator:
    """
    Average precition (AP) evaluator
    Args:
        n_classes (int): number of classes
    """
    def __init__(self, n_classes, iou_thresh=0.5):
        self.n_classes = n_classes
        self.iou_thresh = iou_thresh
        
    def ismatch(self, dect, gts):
        maxIOU = 0.0
        matched = None
        for i, gt in enumerate(gts):
            box1, box2 = dect[2:], gt[1:]
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])
            if xB < xA or yB < yA:
                continue
            
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
            boxBArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > maxIOU:
                maxIOU = iou
                matched = i
        
        if maxIOU >= self.iou_thresh:
            return matched
            
        return None
    
    def evaluate(self, preds, targets):
        AP = []
        for c in range(self.n_classes):
            dects = sorted([pred for pred in preds if pred[0] == c], key=lambda conf: conf[1], reverse=True)
            gts = [target for target in targets if target[0] == c]
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            seen = np.zeros(len(gts))
            for idx, dect in enumerate(dects):
                matched = self.ismatch(dect, gts)
                if matched is not None:
                    if seen[matched] == 0:
                        TP[idx] = 1
                        seen[matched] = 1
                    else:
                        FP[idx] = 1
                else:
                    FP[idx] = 1
            
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / len(gts)
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            
            mrec = [0] + rec.tolist() + [1]
            mpre = [0] + prec.tolist() + [0]
            
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            ii = []
            for i in range(len(mrec) - 1):
                if mrec[1+i] != mrec[i]:
                    ii.append(i + 1)
                    
            ap = 0
            for i in ii:
                ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
            
            AP.append(ap)
            
        return AP

class KPEvaluator(APEvaluator):
    """
    Average precition (AP) evaluator for keypoint detection
    Args:
        n_classes (int): number of classes
    """
    def __init__(self, n_classes, min_dist=7):
        super().__init__(n_classes)
        self.min_dist = min_dist
    
    def ismatch(self, dect, gts):
        minDist = float("inf")
        matched = None
        for i, gt in enumerate(gts):
            (x1, y1), (x2, y2) = dect[2:], gt[1:]
            dist = np.linalg.norm([x1-x2, y1-y2])
            if dist < minDist:
                minDist = dist
                matched = i
        
        if minDist <= self.min_dist:
            return matched
            
        return None

def do_evaluation(cfg, model, dataset_name: str):
    """
    Do evaluation on `dataset_name`.
    Args:
        model (torch.nn.Module): a pretrained model.
        dataset_name (str): name of the dataset to be evaluated.
    """
    evaluator = KPEvaluator(1, min_dist=7)
    model.eval()
    pred_points, ground_truth_points = [], []
    test_loader = build_test_loader(cfg, dataset_name)
    data_list = test_loader.dataset.data_list
    iW, iH = cfg.MODEL.INPUT_SHAPE
    with torch.no_grad():
        for idx, (x, target_hm) in enumerate(test_loader):
            y = model(x)
            hm = torch.clamp(y.sigmoid_(), min=1e-4, max=1-1e-4)
            hmax = torch.nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
            keypoints = hm * (hmax == hm).float()
            _, n_classes, hm_h, hm_w = keypoints.shape
            C, Y, X = torch.where(keypoints[0] > cfg.TEST.THRESH)
            probs = keypoints[0, C, Y, X].cpu().numpy()
            pred_points.extend([[int(c), prob, int(x), int(y)] for c, prob, x, y in zip(C, probs, X, Y)])
            ground_truth_points.extend([
                [anno["category_id"], anno["keypoint"][0]*iW/data_list[idx]["width"], anno["keypoint"][1]*iH/data_list[idx]["height"]] \
                for anno in data_list[idx]["annotations"]
            ])
    
    AP = evaluator.evaluate(pred_points, ground_truth_points)
    return 