import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def visualize_data():
    #Currently using stuff from Citeseer with no features
    emb = np.load("emb.npy")
    labels = np.load("labels.npy")
    labels = np.dot(labels, np.arange(6))

    color_labels = 1.0 * np.arange(6)
    rgb_values = sns.color_palette("hls", 6)
    labels = [rgb_values[int(label)] for label in labels]

    emb_2D = TSNE(perplexity = 50, early_exaggeration = 3.0, verbose = 2, learning_rate = 20.0, n_iter = 5000, n_iter_without_progress = 1000).fit_transform(emb)
    plt.scatter(emb_2D[:,0], emb_2D[:,1], s = 5, color = labels)
    plt.show()
    plt.close()




