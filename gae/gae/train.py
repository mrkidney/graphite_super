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
from sklearn.preprocessing import normalize

from optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from model import GCNModelRelnet, GCNModelVAE, GCNModelAuto
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, edge_dropout, preprocess_graph_coo
from preprocessing import preprocess_partials

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 14, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 10, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 7, 'Number of units in hidden layer 4.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0.15, 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0.2, 'Scale down contribution of autoregressive to final link prediction')
flags.DEFINE_integer('sphere_prior', 0, '1 for normalizing the embeddings to be near sphere surface')
flags.DEFINE_integer('relnet', 0, '1 for relational network between embeddings to predict edges')
flags.DEFINE_integer('auto_node', 0, '1 for autoregressive by node')
flags.DEFINE_integer('vae', 1, '1 for doing VGAE embeddings first')
flags.DEFINE_float('auto_dropout', 0.1, 'Dropout for specifically autoregressive neurons')
flags.DEFINE_float('threshold', 0.75, 'Threshold for autoregressive graph prediction')

flags.DEFINE_integer('weird', 0, 'you know')
flags.DEFINE_integer('verbose', 1, 'verboseness')
flags.DEFINE_integer('mini_batch', 10, 'mini batches of partial graphs')

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('gpu', -1, 'Which gpu to use')
flags.DEFINE_integer('seeded', 0, 'Set numpy random seed')

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

rocs = np.zeros(10)
aps = np.zeros(10)
for test in range(10):
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_def)
    adj = adj_train

    if FLAGS.auto_node:
        partials = preprocess_partials(adj)
    else:
        partials = sparse_to_tuple(adj)

    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'adj_label_mini': tf.sparse_placeholder(tf.float32),
        'partials': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'auto_dropout': tf.placeholder_with_default(0., shape=()),
        'row': tf.placeholder_with_default(0, shape=())
    }

    num_nodes = adj.shape[0]

    # Create model
    model = None
    if FLAGS.relnet:
        model = GCNModelRelnet(placeholders, num_features, num_nodes, features_nonzero)
    elif FLAGS.auto_node:
        model = GCNModelAuto(placeholders, num_features, num_nodes, features_nonzero)
    else:
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
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
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def relu(x):
        return np.maximum(x, 0)
    def cast(x):
        y = np.zeros_like(x)
        y[x < FLAGS.threshold] = 0
        y[x >= FLAGS.threshold] = 1
        return y

    def auto_build(emb, w1, w2):
        z = normalize(emb)

        for row in range(FLAGS.mini_batch):
            partial_adj = cast(sigmoid(np.dot(z, z.T)))
            partial_norm = preprocess_graph_coo(partial_adj)

            hidden = np.dot(z, w1)
            hidden = relu(partial_norm.dot(hidden))
            hidden = np.dot(hidden, w2)
            hidden = partial_norm.dot(hidden)
            hidden = normalize(hidden)

            z = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * hidden
            z = normalize(z)
        return sigmoid(np.dot(z, z.T))


    def reconstruct():
        feed_dict = construct_feed_dict(adj_norm, adj_label, adj_label, features, partials, placeholders)
        feed_dict.update({placeholders['dropout']: 0.})
        feed_dict.update({placeholders['auto_dropout']: 0.})

        if not FLAGS.auto_node:
            emb, recon = sess.run([model.z_mean, model.reconstructions_noiseless], feed_dict=feed_dict)
            return np.reshape(recon, (num_nodes, num_nodes))

        emb, w1, w2 = sess.run([model.z_mean, model.w1, model.w2], feed_dict=feed_dict)
        return auto_build(emb, w1, w2)




    def get_roc_score(edges_pos, edges_neg, adj_rec = None):

        if adj_rec is None:
            adj_rec = reconstruct()

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



    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    val_rocs = np.zeros(FLAGS.epochs)
    test_rocs = np.zeros(FLAGS.epochs)
    test_aps = np.zeros(FLAGS.epochs)

    # Train model
    for epoch in range(FLAGS.epochs):

        if FLAGS.edge_dropout > 0:
            adj_train_mini = edge_dropout(adj, FLAGS.edge_dropout)
            adj_label_mini = adj_train_mini + sp.eye(adj_train_mini.shape[0])
            adj_label_mini = sparse_to_tuple(adj_label_mini)
            adj_norm_mini = preprocess_graph(adj_train_mini)
        else:
            adj_label_mini = adj_label
            adj_norm_mini = adj_norm

        row = 0
        if FLAGS.auto_node:
            row = np.random.choice(len(partials))
            #partials = sparse_to_tuple(partials[row])

        feed_dict = construct_feed_dict(adj_norm_mini, adj_label_mini, adj_label, features, partials, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['auto_dropout']: FLAGS.auto_dropout})
        feed_dict.update({placeholders['row']: row})
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl], feed_dict=feed_dict)

        avg_cost = outs[1]
        avg_accuracy = outs[2]

        if FLAGS.auto_node and (epoch + 1) % 50 != 0:
            continue

        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
        val_rocs[epoch] = roc_curr
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
        test_rocs[epoch] = roc_score
        test_aps[epoch] = ap_score

        if FLAGS.verbose:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(roc_curr),
                  "val_ap=", "{:.5f}".format(ap_curr))

    arg = np.argmax(val_rocs)
    rocs[test] = test_rocs[arg]
    aps[test] = test_aps[arg]

    if FLAGS.verbose:
        print(arg)
        print(test_rocs[arg])
        print(test_aps[arg])
        break
if not FLAGS.verbose:
    print((np.mean(rocs), np.std(rocs)))
    print((np.mean(aps), np.std(aps)))