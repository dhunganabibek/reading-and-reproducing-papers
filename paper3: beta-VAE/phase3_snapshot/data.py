"""dSprites ('2D shapes') dataset of Matthey et al. (2017): 737,280 binary 64x64 images from the
Cartesian product of shape (3), scale (6), orientation (40), position X (32), position Y (32)
(and a constant colour factor).  latents_classes columns: [color, shape, scale, orientation, posX, posY].
"""
import os
import urllib.request
import numpy as np

URL = ("https://github.com/google-deepmind/dsprites-dataset/raw/master/"
       "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
FACTORS = ["color", "shape", "scale", "orientation", "posX", "posY"]


def load_dsprites(path=None):
    path = path or os.environ.get("DSPRITES_NPZ", "data/dsprites.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print("downloading dSprites ...", flush=True)
        urllib.request.urlretrieve(URL, path)
    d = np.load(path, allow_pickle=True, encoding="latin1")
    imgs = d["imgs"].astype(np.uint8)                   # (N, 64, 64) in {0,1}
    classes = d["latents_classes"].astype(np.int64)     # (N, 6)
    return imgs, classes
