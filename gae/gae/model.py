from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from layers import Dense, GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, AutoregressiveDecoder, GraphConvolutionDense
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

class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.auto_dropout = placeholders['auto_dropout']
        self.adj_label = placeholders['adj_orig']
        self.noise = placeholders['noise']
        self.temp = placeholders['temp']
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
                                       dropout=self.dropout,
                                       logging=self.logging)(hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(hidden1)

    def get_z(self):

        z = self.z_mean + self.noise * tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        if FLAGS.auto_node or FLAGS.sphere_prior:
          z = tf.nn.l2_normalize(z, dim = 1)
        return z

    def decoder(self, z):

        reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      dropout=0.,
                                      logging=self.logging)(z)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

    def _build(self):
  
        self.encoder(self.inputs)
        z = self.get_z()
        # z_noiseless = self.get_z(random = False)
        # if not FLAGS.vae:
        #   z = z_noiseless

        self.reconstructions = self.decoder(z)
        #self.reconstructions_noiseless = self.decoder(z_noiseless)
        self.reconstructions_noiseless = self.reconstructions

class GCNModelFeedbackInput(GCNModelVAE):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelFeedbackInput, self).__init__(placeholders, num_features, num_nodes, features_nonzero, **kwargs)

    def decoder(self, z):
        recon = tf.nn.sigmoid(tf.matmul(z, tf.transpose(z)))
        d = tf.reduce_sum(recon, 1)
        recon = tf.expand_dims(d, 0) * recon * tf.expand_dims(d, 1)

        hidden1 = GraphConvolutionDense(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)((tf.sparse_tensor_to_dense(self.inputs), recon)) 

        hidden2 = GraphConvolutionDense(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden4,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)((hidden1, recon)) 

        hidden2 = tf.nn.l2_normalize(hidden2, 1)

        hidden2 = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * hidden2
        hidden2 = tf.nn.l2_normalize(hidden2, 1)

        reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden4,
                                      act=lambda x: x,
                                      logging=self.logging)(hidden2)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

class GCNModelFeedback(GCNModelVAE):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelFeedback, self).__init__(placeholders, num_features, num_nodes, features_nonzero, **kwargs)

    def decoder(self, z):
        recon = tf.nn.sigmoid(tf.matmul(z, tf.transpose(z)))
        recon += tf.eye(self.n_samples)
        d = tf.reduce_sum(recon, 1)
        recon = tf.expand_dims(d, 0) * recon * tf.expand_dims(d, 1)

        hidden1 = GraphConvolutionDense(input_dim=FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)((z, recon)) 

        hidden2 = GraphConvolutionDense(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden4,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)((hidden1, recon)) 

        hidden2 = tf.nn.l2_normalize(hidden2, 1)

        #hidden2 = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * hidden2
        hidden2 = (1 - self.temp) * z + self.temp * hidden2
        hidden2 = tf.nn.l2_normalize(hidden2, 1)

        reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden4,
                                      act=lambda x: x,
                                      logging=self.logging)(hidden2)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

class GCNModelRelnet(GCNModelVAE):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelRelnet, self).__init__(placeholders, num_features, num_nodes, features_nonzero, **kwargs)

    def decoder(self, z):

        hidden1 = Dense(input_dim=FLAGS.hidden2,
                                              output_dim=FLAGS.hidden3,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(z) 

        hidden2 = Dense(input_dim=FLAGS.hidden3,
                                              output_dim=FLAGS.hidden4,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)(hidden1) 

        reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden4,
                                      act=lambda x: x,
                                      logging=self.logging)(hidden2)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

class GCNModelAuto(GCNModelVAE):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelAuto, self).__init__(placeholders, num_features, num_nodes, features_nonzero, **kwargs)

    def decoder(self, z):
        l1 = GraphConvolution(input_dim=FLAGS.hidden2,
                                       output_dim=FLAGS.hidden3,
                                       adj=self.adj,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)

        update = l1(z)
        self.w1 = l1.vars['weights']
        l2 = GraphConvolution(input_dim=FLAGS.hidden3,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.auto_dropout,
                                       logging=self.logging)
        self.w2 = l2.vars['weights']
        update = l2(update)
        update = tf.nn.l2_normalize(update, 1)

        z = (1 - FLAGS.autoregressive_scalar) * z + FLAGS.autoregressive_scalar * update
        z = tf.nn.l2_normalize(z, 1)

        reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      dropout=0.,
                                      logging=self.logging)(z)

        reconstructions = tf.reshape(reconstructions, [-1])
        return reconstructions

    def _build(self):
  
        self.encoder(self.inputs)
        z = self.get_z(random = True)
        z_noiseless = self.get_z(random = False)
        if not FLAGS.vae:
          z = z_noiseless

        self.reconstructions = self.decoder(z)
