from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from layers import *
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

class GCNModelFeedback(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_label = placeholders['adj_orig']
        self.labels = placeholders['labels']
        self.labels_mask = placeholders['labels_mask']
        self.weight_norm = 0
        self.build()

    def encoder(self, inputs):

        hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)(inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=0.,
                                       logging=self.logging)(hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=0.,
                                          logging=self.logging)(hidden1)

    def get_z(self, random):

        z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
        if not random or not FLAGS.vae:
          z = self.z_mean

        return z

    def decoder(self, z):

        l0 = GraphConvolutionDense(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden3,
                                      sparse_inputs = True,
                                      act=tf.nn.relu,
                                      dropout=0.,
                                      logging=self.logging)

        l1 = GraphConvolutionDense(input_dim=FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)

        l2 = GraphConvolutionDense(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden2,
                                              act=lambda x: x,
                                              dropout=0.,
                                              logging=self.logging)

        l3 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                        act=lambda x: x,
                                        logging=self.logging)

        recon = l3(z)
        recon = tf.nn.sigmoid(recon)
        d = tf.reduce_sum(recon, 1)
        d = tf.pow(d, -0.5)
        recon = tf.expand_dims(d, 0) * recon * tf.expand_dims(d, 1)

        update = l1((z, recon, z)) + l0((self.inputs, recon, z))
        update = l2((update, recon, z))

        update = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * update
        reconstructions = l3(update)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

    def _build(self):
  
        self.encoder(self.inputs)
        z = self.get_z(random = True)
        z_noiseless = self.get_z(random = False)
        if not FLAGS.vae:
          z = z_noiseless

        self.reconstructions = self.decoder(z)

        inputs = self.inputs

        hidden = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)

        output = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)

        self.output = output(hidden(inputs))

        self.weight_norm = tf.nn.l2_loss(hidden.vars['weights']) + tf.nn.l2_loss(output.vars['weights'])


