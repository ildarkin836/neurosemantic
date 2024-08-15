import numpy as np

def crop_face(img, bbox):
    x, y, w, h = bbox.astype(int)
    return img[y: y+h, x:x+w]