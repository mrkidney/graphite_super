from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from layers import InnerProductConfigurer, Dense, GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, AutoregressiveConfigurer
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.partials = tf.sparse_reshape(placeholders['partials'], (-1, num_nodes))
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)


        # self.z = Dense(input_dim=FLAGS.hidden2,
        #                                   output_dim=FLAGS.hidden3,
        #                                   dropout=self.dropout,
        #                                   act=tf.nn.relu,
        #                                   logging=self.logging)(self.z)
        
        # self.z = Dense(input_dim=FLAGS.hidden3,
        #                                   output_dim=FLAGS.hidden4,
        #                                   dropout=self.dropout,
        #                                   act=lambda x: x,
        #                                   logging=self.logging)(self.z)

        self.z_adj = tf.sigmoid(tf.matmul(self.z, tf.transpose(self.z)))

        self.z = GraphConvolution(input_dim=FLAGS.hidden2,
                                          output_dim=FLAGS.hidden3,
                                          adj = self.z_adj
                                          dropout=self.dropout,
                                          act=tf.nn.relu,
                                          logging=self.logging)(self.z)
        
        self.z = GraphConvolution(input_dim=FLAGS.hidden3,
                                          output_dim=FLAGS.hidden4,
                                          adj = self.z_adj,
                                          dropout=self.dropout,
                                          act=lambda x: x,
                                          logging=self.logging)(self.z)

        self.reconstructions = InnerProductConfigurer(input_dim=FLAGS.hidden4,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)

        self.reconstructions = Dense(input_dim=FLAGS.hidden4,
                                          output_dim=FLAGS.hidden5,
                                          dropout=self.dropout,
                                          act=tf.nn.relu,
                                          logging=self.logging)(self.reconstructions)

        self.reconstructions = Dense(input_dim=FLAGS.hidden5,
                                          output_dim=1,
                                          dropout=self.dropout,
                                          act=lambda x: x,
                                          logging=self.logging)(self.reconstructions)

        self.reconstructions = tf.reshape(self.reconstructions, [-1])
