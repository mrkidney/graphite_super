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

        # hidden = GraphConvolutionSparse(input_dim=self.input_dim,
        #                                       output_dim=16,
        #                                       adj=self.adj,
        #                                       act=tf.nn.relu,
        #                                       features_nonzero=self.features_nonzero,
        #                                       dropout=self.dropout,
        #                                       logging=self.logging)

        # output = GraphConvolution(input_dim=FLAGS.hidden_y * 5,
        #                                output_dim=self.output_dim,
        #                                adj=self.adj,
        #                                act=lambda x: x,
        #                                dropout=self.dropout,
        #                                logging=self.logging)

        hidden = MultiGraphAttention(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden_y,
                                              adj=self.adj,
                                              act=tf.nn.elu,
                                              features_nonzero=self.features_nonzero,
                                              num_head = 8,
                                              dropout=self.dropout,
                                              logging=self.logging)

        output = MultiGraphAttention(input_dim=FLAGS.hidden_y * 8,
                                       output_dim=self.output_dim,
                                       adj=self.adj,
                                       sparse=False,
                                       features_nonzero=self.features_nonzero,
                                       num_head = 1,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)

        self.outputs = output(hidden(inputs))

        # self.weight_norm = FLAGS.weight_decay * tf.nn.l2_loss(hidden.vars['weights'])
        self.weight_norm = FLAGS.weight_decay * hidden.vars['weight_l2']

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

    def sample(self, mean, log_std, dim):
        return mean + tf.random_normal([self.n_samples, dim]) * tf.exp(log_std)

    def reconstruct_graph(self, emb, normalize = True):
        embT = tf.transpose(emb)
        graph = tf.matmul(emb, embT)
        if normalize:
          graph = tf.nn.sigmoid(graph)
          d = tf.reduce_sum(graph, 1)
          d = tf.pow(d, -0.5)
          d = tf.stop_gradient(d)
          graph = tf.expand_dims(d, 0) * graph * tf.expand_dims(d, 1)
        return graph

    def define_layers(self):

        self.hidden_z1q_layer = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden_z1q,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)

        self.z1q_mean_layer = GraphConvolution(input_dim=FLAGS.hidden_z1q,
                                       output_dim=FLAGS.dim_z1,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=0.,
                                       logging=self.logging)

        self.z1q_log_std_layer = GraphConvolution(input_dim=FLAGS.hidden_z1q,
                                          output_dim=FLAGS.dim_z1,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=0.,
                                          logging=self.logging)

        # self.hidden_y_layer_x = GraphConvolutionSparse(input_dim=self.input_dim,
        #                                       output_dim=FLAGS.hidden_y,
        #                                       adj=self.adj,
        #                                       features_nonzero=self.features_nonzero,
        #                                       act=lambda x: x,
        #                                       dropout=self.dropout,
        #                                       logging=self.logging)

        # self.hidden_y_layer_z1 = GraphConvolution(input_dim=FLAGS.dim_z1,
        #                                output_dim=FLAGS.hidden_y,
        #                                act=lambda x: x,
        #                                adj=self.adj,
        #                                dropout=self.dropout,
        #                                logging=self.logging)

        # self.y_layer = GraphConvolution(input_dim=FLAGS.hidden_y,
        #                                output_dim=self.output_dim,
        #                                act=lambda x: x,
        #                                adj=self.adj,
        #                                dropout=self.dropout,
        #                                logging=self.logging)

        self.hidden_y_layer_x = MultiGraphAttention(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden_y,
                                              adj=self.adj,
                                              act=tf.nn.elu,
                                              features_nonzero=self.features_nonzero,
                                              num_head = FLAGS.num_head,
                                              dropout=self.dropout,
                                              logging=self.logging)

        self.hidden_y_layer_z1 = MultiGraphAttention(input_dim=FLAGS.dim_z1,
                                              output_dim=FLAGS.hidden_y,
                                              adj=self.adj,
                                              act=tf.nn.elu,
                                              sparse=False,
                                              features_nonzero=self.features_nonzero,
                                              num_head = 1,
                                              dropout=self.dropout,
                                              logging=self.logging)

        self.y_layer = MultiGraphAttention(input_dim=FLAGS.hidden_y * (FLAGS.num_head+1),
                                       output_dim=self.output_dim,
                                       adj=self.adj,
                                       sparse=False,
                                       features_nonzero=self.features_nonzero,
                                       num_head = 1,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)

        #self.weight_norm += FLAGS.weight_decay * tf.nn.l2_loss(self.hidden_y_layer_x.vars['weights'])
        #self.weight_norm += FLAGS.z1_decay * tf.nn.l2_loss(self.hidden_y_layer_z1.vars['weights'])
        self.weight_norm += FLAGS.weight_decay * self.hidden_y_layer_x.vars['weight_l2']
        self.weight_norm += FLAGS.z1_decay * self.hidden_y_layer_z1.vars['weight_l2']
        # self.weight_norm += FLAGS.graphite_decay * tf.nn.l2_loss(self.hidden_y_layer_graphite.vars['weights'])

        self.hidden_z2_layer = Dense(input_dim=FLAGS.dim_z1 + self.output_dim,
                                       output_dim=FLAGS.hidden_z2,
                                       act=tf.nn.relu,
                                       dropout=0.,
                                       logging=self.logging)

        self.z2_mean_layer = Dense(input_dim=FLAGS.hidden_z2,
                                       output_dim=FLAGS.dim_z2,
                                       act=lambda x: x,
                                       dropout=0.,
                                       logging=self.logging)

        self.z2_log_std_layer = Dense(input_dim=FLAGS.hidden_z2,
                                       output_dim=FLAGS.dim_z2,
                                       act=lambda x: x,
                                       dropout=0.,
                                       logging=self.logging)

        self.hidden_z1p_layer = Dense(input_dim=FLAGS.dim_z2 + self.output_dim,
                                              output_dim=FLAGS.hidden_z1p,
                                              act=tf.nn.relu,
                                              dropout=0.,
                                              logging=self.logging)

        self.z1p_mean_layer = Dense(input_dim=FLAGS.hidden_z1p,
                                       output_dim=FLAGS.dim_z1,
                                       act=lambda x: x,
                                       dropout=0.,
                                       logging=self.logging)

        self.z1p_log_std_layer = Dense(input_dim=FLAGS.hidden_z1p,
                                          output_dim=FLAGS.dim_z1,
                                          act=lambda x: x,
                                          dropout=0.,
                                          logging=self.logging)
        
        self.hidden_x_input_layer = GraphConvolutionDense(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden_x,
                                      sparse_inputs = True,
                                      features_nonzero=self.features_nonzero,
                                      act=lambda x: x,
                                      dropout=0.,
                                      logging=self.logging)

        self.hidden_x_z1_layer = GraphConvolutionDense(input_dim=FLAGS.dim_z1,
                                              output_dim=FLAGS.hidden_x,
                                              act=lambda x: x,
                                              dropout=0.,
                                              logging=self.logging)

        self.x_layer = GraphConvolutionDense(input_dim=FLAGS.hidden_x,
                                              output_dim=FLAGS.dim_z1,
                                              act=lambda x: x,
                                              dropout=0.,
                                              logging=self.logging)
    
    def encoder_z1(self, inputs):
        hidden = self.hidden_z1q_layer(inputs)
        return self.z1q_mean_layer(hidden), self.z1q_log_std_layer(hidden)

    def encoder_y(self, z1, inputs):
        hidden = tf.concat((self.hidden_y_layer_x(inputs), self.hidden_y_layer_z1(z1)), 1)
        return self.y_layer(hidden)

    def encoder_z2(self, z1, y):
        prior_full = tf.concat((z1, y), axis = 1)
        hidden = self.hidden_z2_layer(prior_full)
        return self.z2_mean_layer(hidden), self.z2_log_std_layer(hidden)

    def decoder_z1(self, z2, y):
        prior_full = tf.concat((z2, y), axis = 1)
        hidden = self.hidden_z1p_layer(prior_full)
        return self.z1p_mean_layer(hidden), self.z1p_log_std_layer(hidden)

    def decoder_x(self, z1):
        graph = self.reconstruct_graph(z1)

        hidden = self.hidden_x_z1_layer((z1, graph)) + self.hidden_x_input_layer((self.inputs, graph))
        hidden = tf.nn.relu(hidden)
        emb = self.x_layer((hidden, graph))

        emb = (1 - FLAGS.autoregressive_scalar) * z1 + FLAGS.autoregressive_scalar * emb

        reconstructions = self.reconstruct_graph(emb, normalize = False)

        return tf.reshape(reconstructions, [-1]), emb

    def _build(self):
        self.define_layers()
  
        self.z1q_mean, self.z1q_log_std = self.encoder_z1(self.inputs)
        self.z1q = self.sample(self.z1q_mean, self.z1q_log_std, FLAGS.dim_z1)

        # self.reconstructions, self.zf = self.decoder_x(self.z1q)
        # _, self.zf_noiseless = self.decoder_x(self.z1q_mean)

        # self.y = self.encoder_y(self.zf, self.inputs)
        # self.outputs = self.encoder_y(self.zf_noiseless, self.inputs)

        self.reconstructions, _ = self.decoder_x(self.z1q)

        self.y = self.encoder_y(self.z1q, self.inputs)
        self.outputs = self.encoder_y(self.z1q_mean, self.inputs)


