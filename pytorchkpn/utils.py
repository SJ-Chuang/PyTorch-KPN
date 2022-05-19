from termcolor import colored
import numpy as np
import sys, math, logging

__all__ = [
    "Point2Heatmap",
    "logger"
]

def Point2Heatmap(points, shape, sigma=3):
    '''
    Convert points to keypoint heatmaps.
    Args:
        points (list): points to be converted. (e.g. [[x1, y1], [x2, y2], ...])
        shape (tuple): shape of (H, W), where H and W are the height and width of the heatmap.
        sigma (int): sigma of gaussian kernel.
        
    Returns:
        a keypoint heatmap
    '''
    H, W = shape
    hm = []
    for point in points:
        x, y = point
        channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        hm.append(np.reshape(channel, newshape=(H, W)))
    return np.max(hm, 0)

logging.basicConfig(
    filename="log.txt",
    format=colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
    filemode='w'
)
logger = logging.getLogger("pytorchkpn")
logger.setLevel(logging.DEBUG) 
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)