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

from optimizer import OptimizerVAE
from input_data import *
from model import *
from preprocessing import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 4.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0.2, 'Scale down contribution of autoregressive to final link prediction')
flags.DEFINE_integer('vae', 1, '1 for doing VGAE embeddings first')


flags.DEFINE_integer('verbose', 1, 'verboseness')
flags.DEFINE_integer('test_count', 1, 'batch of tests')
flags.DEFINE_integer('save', 0, '1 to save final embeddings for future visualization')

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'vgae', 'Model string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('gpu', -1, 'Which gpu to use')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('scale', 0, 'use scaled inner prod')


if FLAGS.seeded:
    np.random.seed(1)
    #tf.set_random_seed(1)

dataset_str = FLAGS.dataset
model_str = FLAGS.model

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)
adj_norm = preprocess_graph(adj)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

for test in range(FLAGS.test_count):

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32)
    }

    num_nodes = adj.shape[0]

    # Create model
    model = GCNModelFeedback(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    if FLAGS.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sess = tf.Session()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu) # Or whichever device you would like to use
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    # Train model
    for epoch in range(FLAGS.epochs):

        if FLAGS.edge_dropout > 0:
            adj_train_mini = edge_dropout(adj, FLAGS.edge_dropout)
            adj_norm_mini = preprocess_graph(adj_train_mini)
        else:
            adj_norm_mini = adj_norm

        feed_dict = construct_feed_dict(adj_norm_mini, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        avg_cost = outs[1]
        avg_accuracy = outs[2]

        if FLAGS.verbose:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(roc_curr),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "test_roc=", "{:.5f}".format(roc_score),
                  "test_ap=", "{:.5f}".format(ap_score))
