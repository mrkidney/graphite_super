from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

import node2vec
from gensim.models import Word2Vec

from optimizer import OptimizerVAE
from input_data import *
from model import *
from preprocessing import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('test_count', 10, 'Set num tests')
flags.DEFINE_integer('emb_size', 128, 'Number of eigenvectors for embedding')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if FLAGS.seeded:
    np.random.seed(1)

dataset_str = FLAGS.dataset

# Load data
adj, features = load_data(dataset_str)

adj_def = adj

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

rocs = np.zeros(FLAGS.test_count)
aps = np.zeros(FLAGS.test_count)


def deepwalk(adj):
    nx_G = nx.from_scipy_sparse_matrix(adj)
    G = node2vec.Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(10, 80)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=FLAGS.emb_size, window=10, min_count=0, sg=1, workers=8, iter=1)

    z = np.zeros((adj.shape[0], FLAGS.emb_size))
    for i in range(adj.shape[0]):
        z[i] = model.wv[str(i)]
    return z


for k in range(FLAGS.test_count):

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = get_test_edges(adj_def)
    val_edges = tuple(zip(*val_edges))
    val_edges_false = tuple(zip(*val_edges_false))
    test_edges = tuple(zip(*test_edges))
    test_edges_false = tuple(zip(*test_edges_false))
    adj = adj_train

    z = deepwalk(adj)
    adj_rec = np.dot(z, z.T)

    preds = sigmoid(adj_rec[test_edges])
    preds_neg = sigmoid(adj_rec[test_edges_false])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    rocs[k] = roc_score
    aps[k] = ap_score

print((np.mean(rocs), stats.sem(rocs)))
print((np.mean(aps), stats.sem(aps)))  
