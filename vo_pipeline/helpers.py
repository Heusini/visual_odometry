from typing import Optional, Callable
import numpy as np
import cv2
import os

def load_images(
    path: str,
    filter : Optional[Callable[[str], bool]] = None,
    start : Optional[int] = None,
    end : Optional[int] = None
) -> np.ndarray:
    
    filenames = []

    # open directory and read all filenames
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file() and (filter is None or filter(entry.name)):
                filenames.append(entry.name)

    # sort filenames
    filenames.sort()

    if start is None: start = 0
    if end is None: end = len(filenames)

    # load images
    imgs = []
    for filename in filenames[start:end]:
        img = cv2.imread(path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)

    return np.array(imgs)