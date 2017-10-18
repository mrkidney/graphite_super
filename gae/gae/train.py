from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, edge_dropout

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 14, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 14, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 14, 'Number of units in hidden layer 4.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0.5, 'Scale down contribution of autoregressive to final link prediction')
flags.DEFINE_integer('sphere_prior', 0, '1 for normalizing the embeddings to be near sphere surface')
flags.DEFINE_integer('relnet', 0, '1 for relational network between embeddings to predict edges')
flags.DEFINE_integer('auto_node', 0, '1 for autoregressive by node')
flags.DEFINE_integer('auto_edge', 0, '1 for autoregressive by edge')
flags.DEFINE_integer('vae', 1, '1 for doing VGAE embeddings first')
flags.DEFINE_integer('test', 1, 'Number of tests for mean and std')
flags.DEFINE_float('auto_dropout', 0., 'Dropout for specifically autoregressive neurons')

flags.DEFINE_integer('parallel', 1, 'Internal use, dont mess with')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('gpu', -1, 'Which gpu to use')
flags.DEFINE_integer('verbose', 0, 'Print all epochs')

model_str = FLAGS.model
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

roc = np.zeros(FLAGS.test)
ap = np.zeros(FLAGS.test)
for i in range(FLAGS.test):
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_def)
    adj = adj_train

    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'parallel': tf.placeholder_with_default(1., shape=())
    }

    num_nodes = adj.shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    if FLAGS.gpu == -1:
        sess = tf.Session()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu) # Or whichever device you would like to use
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    cost_val = []
    acc_val = []


    def get_roc_score(edges_pos, edges_neg):
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: 0.})
        feed_dict.update({placeholders['parallel']: 0.})
        FLAGS.parallel = 0
        emb, recon = sess.run([model.z_mean, model.reconstructions_noiseless], feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adj_rec = np.reshape(recon, (num_nodes, num_nodes))

        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    cost_val = []
    acc_val = []
    val_roc_score = []

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        if FLAGS.edge_dropout > 0:
            adj_train_mini = edge_dropout(adj, FLAGS.edge_dropout)
            adj_norm_mini = preprocess_graph(adj_train_mini)
        else:
            adj_norm_mini = adj_norm

        t = time.time()
        feed_dict = construct_feed_dict(adj_norm_mini, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['parallel']: 1.})
        FLAGS.parallel = 1
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl], feed_dict=feed_dict)

        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr = -1
        ap_curr = -1
        if not FLAGS.auto_node or (epoch + 1) % 50 == 0:
            roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
            val_roc_score.append(roc_curr)

        if FLAGS.verbose:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                  "val_ap=", "{:.5f}".format(ap_curr))

    feed_dict.update({placeholders['parallel']: 0.})
    FLAGS.parallel = 0
    FLAGS.auto_dropout = 0.
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
    print(str(roc_score) + ", " + str(ap_score))
    sess.close()
    roc[i] = roc_score
    ap[i] = ap_score

print((np.mean(roc), np.std(roc)))
print((np.mean(ap), np.std(ap)))
