from gae.initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dense_tensor_to_sparse(x, num_nodes):
    idx = tf.where(tf.not_equal(x, 0))
    return tf.SparseTensor(idx, tf.gather_nd(x, idx), [num_nodes, num_nodes])

def sparse_convolution(adj, deg, inputs):
    output = tf.sparse_tensor_dense_matmul(deg, inputs)
    output = tf.sparse_tensor_dense_matmul(adj, output)
    output = tf.sparse_tensor_dense_matmul(deg, output)
    return output

def sparse_diag(num_nodes):
    rows = tf.range(num_nodes, dtype = tf.int64)
    rows = tf.stack([rows, rows], axis = 1)
    rows = tf.reshape(rows, [-1, 2])
    eye = tf.SparseTensor(rows, tf.ones(num_nodes), [num_nodes, num_nodes])
    return rows, eye

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., pos=False, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            if pos:
                self.vars['weights'] = tf.square(self.vars['weights'])
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.nn.dropout(x, 1-self.dropout)
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class AutoregressiveDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, hidden_dim, hidden_dim2, adj, num_nodes, dropout=0., auto_dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(AutoregressiveDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.adj = adj
        self.num_nodes = num_nodes
        self.auto_dropout = auto_dropout
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights1'] = weight_variable_glorot(input_dim + 1, hidden_dim, name="weights1")
            self.vars['weights2'] = weight_variable_glorot(hidden_dim, hidden_dim2, name="weights2")


    def _call(self, inputs):
        adj = self.adj
        z = tf.nn.dropout(inputs, 1-self.dropout)

        x = tf.transpose(z)
        x = tf.matmul(z, x)
        x *= (1 - FLAGS.autoregressive_scalar) * FLAGS.sigmoid_scalar

        num_nodes = self.num_nodes

        rows, eye = sparse_diag(num_nodes)

        def z_update(row):
            partial_adj = tf.sparse_slice(adj, [0,0], row)
            partial_adj = tf.sparse_reset_shape(partial_adj, [num_nodes, num_nodes])
            partial_adj = tf.sparse_maximum(partial_adj, eye)
            deg = tf.sparse_reduce_sum(partial_adj, 0)
            deg = tf.pow(tf.maximum(deg, 1), -0.5)
            deg = tf.SparseTensor(rows, deg, [num_nodes, num_nodes])

            helper_feature = tf.one_hot([row[0]], num_nodes)
            helper_feature = tf.reshape(helper_feature, [num_nodes, 1])
            z_prime = tf.concat((z, helper_feature), 1)

            hidden = tf.matmul(z_prime, self.vars['weights1'])
            hidden = tf.nn.relu(sparse_convolution(partial_adj, deg, hidden))
            hidden = tf.nn.dropout(hidden, 1-self.auto_dropout)
            hidden = tf.matmul(hidden, self.vars['weights2'])
            hidden = sparse_convolution(partial_adj, deg, hidden)

            if FLAGS.sphere_prior:
                hidden = tf.nn.l2_normalize(hidden, dim = 1)
            index = tf.cast(row[0], tf.int32)
            vec = tf.expand_dims(hidden[index], 1)
            hidden = tf.squeeze(tf.matmul(hidden, vec))
            if not FLAGS.sphere_prior:
                hidden = tf.nn.tanh(hidden)
            hidden *= FLAGS.autoregressive_scalar * FLAGS.sigmoid_scalar
            hidden = tf.concat([hidden[:index + 1], tf.zeros([num_nodes - index - 1])], 0)
            return hidden


        supplement = tf.map_fn(z_update, rows, dtype = tf.float32, parallel_iterations = 100)
        supplement = (supplement + tf.transpose(supplement))
        outputs = x + supplement
        return outputs

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
