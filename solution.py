import argparse
import os

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='First test task on images similarity.', add_help=False)
parser.add_argument('-h','--help', action='help', default=argparse.SUPPRESS,
                    help='show this help message and exit')
parser.add_argument('--path', help='folder with images', required=True)

args = parser.parse_args()

def get_histogram(img):
    im = Image.open(img)
    r = np.asarray(im.convert("RGB", (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)))
    g = np.asarray(im.convert("RGB", (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)))
    b = np.asarray(im.convert("RGB", (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)))
    hr, h_bins = np.histogram(r, bins=256, normed=True, density=True)
    hg, h_bins = np.histogram(g, bins=256, normed=True, density=True)
    hb, h_bins = np.histogram(b, bins=256, normed=True, density=True)
    hist = np.array([hr, hg, hb]).ravel()
    return (hist)

def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])
    return d

def distance(im1, im2):
    distance = chi2_distance(get_histogram(im1), get_histogram(im2))

    if distance < 0.15:
        return ([os.path.basename(im1), os.path.basename(im2)])

    else:
        return ('')

myList = [os.path.join(r, file) for r, d, f in os.walk(args.path) for file in f]
unorderedPair = ((x, y) for x in myList for y in myList if y > x)
total = []
for pair in unorderedPair:
    total = distance(pair[0], pair[1])
    if total != '':
        print(total)