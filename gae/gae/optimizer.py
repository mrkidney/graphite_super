import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def y_semi_supervised(preds, labels, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, 1)
    return labels * mask + preds * (1 - mask)

def y_prior_distribution(labels, mask, dim):
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, 1)
    prior = tf.ones_like(labels) / dim
    return labels * mask + prior * (1 - mask)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

# def log_normal_pdf_tf(mean, log_std, obs):
#     dist = tf.contrib.distributions.MultivariateNormalDiag(mean, tf.exp(log_std))
#     return dist.log_prob(obs)

def log_normal_pdf_tf(mean, log_std, obs, dim = 7):
    pdf = -0.5 * (tf.square(obs - mean) * tf.exp(-2.0 * log_std)) - 0.5 * (dim * tf.log(2 * np.pi) + 2 * log_std)
    return tf.reduce_sum(pdf, 1)

# def kl_categorical(probs, prior):
#     probs_dist = tf.contrib.distributions.Categorical(probs)
#     prior_dist = tf.contrib.distributions.Categorical(prior)
#     return tf.contrib.distributions.kl_divergence(probs_dist, prior_dist)

def kl_categorical(probs, dim, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    full_mask = tf.expand_dims(mask, 1)

    dummy = tf.ones_like(probs) / (1.0 * dim)
    probs = dummy * full_mask + probs * (1 - full_mask)

    kl = tf.reduce_sum(probs * tf.maximum(tf.log(probs * dim), -10000), 1)
    mask = 1 - mask
    mask /= tf.reduce_mean(mask)
    kl *= mask
    return kl

def kl(mean, log_std):
    return 0.5 * tf.reduce_sum(1 + 2 * log_std - tf.square(mean) - tf.square(tf.exp(log_std)), 1)

class OptimizerSuper(object):
    def __init__(self, model):

        self.cost = model.weight_norm

        self.cost += masked_softmax_cross_entropy(model.outputs, model.labels, model.labels_mask)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.accuracy = masked_accuracy(model.outputs, model.labels, model.labels_mask)

class OptimizerSemiGen(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        if FLAGS.subsample:
            indices = labels.indices
            no_edge_indices = tf.random_uniform(tf.shape(indices), maxval = num_nodes, dtype=tf.int64)
            no_edge_tensor = tf.SparseTensor(no_edge_indices, tf.zeros_like(labels.values), labels.dense_shape)
            all_tensor = tf.sparse_maximum(no_edge_tensor, labels_sub)
            
            all_indices = all_tensor.indices

            preds_vals = tf.reduce_sum(tf.gather(preds, all_indices[:,0]) * tf.gather(preds, all_indices[:,1]), axis = 1)
            labels_vals = all_tensor.values

            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_vals, targets=labels_vals, pos_weight=1))
        else:
            preds_sub = tf.matmul(preds_sub, tf.transpose(preds_sub))
            labels_sub = tf.sparse_tensor_to_dense(labels_sub, validate_indices = False)
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        y_semi = y_semi_supervised(tf.nn.softmax(model.y), model.labels, model.labels_mask)
        y_prior = y_prior_distribution(model.labels, model.labels_mask, model.output_dim)

        self.cost += (1.0 / num_nodes) * tf.reduce_mean(kl_categorical(y_semi, model.output_dim, model.labels_mask))

        self.cost += (1.0 / num_nodes) * tf.reduce_mean(tf.maximum(log_normal_pdf_tf(model.z1q_mean, model.z1q_log_std, model.z1q, model.output_dim), -10000))

        for label in range(model.output_dim):
            y_pos = tf.one_hot(indices = label, depth = model.output_dim)
            y_pos = tf.ones_like(model.y) * y_pos

            z2_mean, z2_log_std = model.encoder_z2(model.z1q, y_pos)
            z2 = model.sample(z2_mean, z2_log_std, FLAGS.dim_z2)
            z1p_mean, z1p_log_std = model.decoder_z1(z2, y_pos)

            self.cost -= (1.0 / num_nodes) * tf.reduce_mean(y_semi[:,label] * kl(z2_mean, z2_log_std))
            self.cost -= (1.0 / num_nodes) * tf.reduce_mean(y_semi[:,label] * tf.maximum(log_normal_pdf_tf(z1p_mean, z1p_log_std, model.z1q, model.output_dim), -10000))
        
        self.cost *= FLAGS.tau

        self.cost += FLAGS.alpha * masked_softmax_cross_entropy(model.outputs, model.labels, model.labels_mask)
        self.cost += model.weight_norm

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.accuracy = masked_accuracy(model.outputs, model.labels, model.labels_mask)

class OptimizerSemi(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        if FLAGS.subsample:
            indices = labels.indices
            no_edge_indices = tf.random_uniform(tf.shape(indices), maxval = num_nodes, dtype=tf.int64)
            no_edge_tensor = tf.SparseTensor(no_edge_indices, tf.zeros_like(labels.values), labels.dense_shape)
            all_tensor = tf.sparse_maximum(no_edge_tensor, labels_sub)
            
            all_indices = all_tensor.indices

            preds_vals = tf.reduce_sum(tf.gather(preds, all_indices[:,0]) * tf.gather(preds, all_indices[:,1]), axis = 1)
            labels_vals = all_tensor.values

            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_vals, targets=labels_vals, pos_weight=1))
        else:
            preds_sub = tf.matmul(preds_sub, tf.transpose(preds_sub))
            labels_sub = tf.sparse_tensor_to_dense(labels_sub, validate_indices = False)
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        self.cost -= (1.0 / num_nodes) * tf.reduce_mean(kl(model.z1q_mean, model.z1q_log_std))

        self.cost *= FLAGS.tau

        self.cost += FLAGS.alpha * masked_softmax_cross_entropy(model.outputs, model.labels, model.labels_mask)
        self.cost += model.weight_norm

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.accuracy = masked_accuracy(model.outputs, model.labels, model.labels_mask)

