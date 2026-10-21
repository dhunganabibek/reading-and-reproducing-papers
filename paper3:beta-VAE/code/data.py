"""dSprites (2D shapes): 737,280 binary 64x64 images, one for every combination of the generative factors."""
import os
import urllib.request

import numpy as np

URL = ("https://github.com/google-deepmind/dsprites-dataset/raw/master/"
       "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
LOCAL_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "dsprites.npz")


def load_dsprites(path=None):
    """Images (N, 64, 64) in {0, 1}, and factor classes (N, 6): color, shape, scale, orientation, posX, posY."""
    path = path or os.environ.get("DSPRITES_NPZ") or (LOCAL_NPZ if os.path.exists(LOCAL_NPZ) else "data/dsprites.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print("downloading dSprites ...", flush=True)
        urllib.request.urlretrieve(URL, path)
    d = np.load(path, allow_pickle=True, encoding="latin1")
    return d["imgs"].astype(np.uint8), d["latents_classes"].astype(np.int64)
