import argparse
import os
import pickle as pkl
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm

import sklearn

emb_path = '/Users/xujinghua/NeMo/embeddings/embeddings_manifest_embeddings.pkl'

emb = pkl.load(open(emb_path, 'rb'))

X = emb['an4_clstk@mrab@an74-mrab-b.wav']
Y = emb['an4_clstk@mjgk@cen8-mjgk-b.wav']

# print(emb1)
# print(sklearn.metrics.pairwise.cosine_similarity(emb1, emb2))

score = (X @ Y.T) / (((X @ X.T) * (Y @ Y.T)) ** 0.5)
score = (score + 1) / 2

print(score)

# I got 0.9999061881044735, which is not right for two different speakers
