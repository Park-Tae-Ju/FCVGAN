import torch
from torch.nn import functional as F
import numpy as np
import cv2


def load_img(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        with open(img_path.encode('utf-8'), 'rb') as f:
            bytes = bytearray(f.read())
            nparr = np.asarray(bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return img_bgr
    if img_bgr is None:
        raise FileNotFoundError(f"{img_path} is not found.")
    else:
        return img_bgr


def load_rgb_img(path):
    img_bgr = load_img(path)
    if not isinstance(img_bgr, np.ndarray):
        raise IOError("Fail to read %s" % path)
    img_rgb = img_bgr[:, :, ::-1]
    return img_rgb

