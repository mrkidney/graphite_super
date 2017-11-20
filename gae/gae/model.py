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

class GCNModel(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModel, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adj_label = placeholders['adj_orig']
        self.labels = placeholders['labels']
        self.labels_mask = placeholders['labels_mask']
        self.weight_norm = 0
        self.build()

    def _build(self):

        self.reconstructions = 0
        inputs = self.inputs

        hidden = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden4,
                                              adj=self.adj,
                                              act=tf.nn.relu,
                                              features_nonzero=self.features_nonzero,
                                              dropout=self.dropout,
                                              logging=self.logging)

        output = GraphConvolution(input_dim=FLAGS.hidden4,
                                       output_dim=self.output_dim,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)

        self.outputs = output(hidden(inputs))

        self.weight_norm = FLAGS.weight_decay * tf.nn.l2_loss(hidden.vars['weights'])


class GCNModelFeedback(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelFeedback, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
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

        hidden = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)
        hidden1 = hidden(inputs)

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
                                      features_nonzero=self.features_nonzero,
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

        update = l1((z, recon)) + l0((self.inputs, recon))
        update = l2((update, recon))

        update = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * update
        reconstructions = l3(update)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions, update

    def _build(self):
  
        self.encoder(self.inputs)
        z = self.get_z(random = True)
        z_noiseless = self.get_z(random = False)
        if not FLAGS.vae:
          z = z_noiseless

        self.reconstructions, _ = self.decoder(z)
        _, z_f = self.decoder(z_noiseless)

        hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden4,
                                      act=tf.nn.relu,
                                      features_nonzero=self.features_nonzero,
                                      adj = self.adj,
                                      dropout=self.dropout,
                                      logging=self.logging)

        hidden2 = GraphConvolution(input_dim=FLAGS.hidden2,
                                      output_dim=FLAGS.hidden4,
                                      act=tf.nn.relu,
                                      adj = self.adj,
                                      dropout=0.,
                                      logging=self.logging)        

        output = GraphConvolution(input_dim=FLAGS.hidden4,
                                       output_dim=self.output_dim,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)


        # self.outputs = hidden1(self.inputs) + hidden2(z)
        self.outputs = hidden1(self.inputs) + hidden2(z_f)
        self.outputs = output(self.outputs)

        self.weight_norm = FLAGS.weight_decay * tf.nn.l2_loss(hidden1.vars['weights']) + FLAGS.emb_decay * tf.nn.l2_loss(hidden2.vars['weights'])


