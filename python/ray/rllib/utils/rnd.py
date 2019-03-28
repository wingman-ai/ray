""" Implementation of Random Network Distillation (RND).
Original Paper: https://arxiv.org/abs/1810.12894 """

import logging

import numpy as np
import tensorflow as tf

from mpi4py import MPI

logger = logging.getLogger(__name__)

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)

def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n + 1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]

def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis + 1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems  # shape 1,


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), comm=None, use_mpi=True):
        self.mean = np.zeros(shape, 'float64')
        self.use_mpi = use_mpi
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

    def update(self, x):
        if self.use_mpi:
            batch_mean, batch_std, batch_count = mpi_moments(x, axis=0, comm=self.comm)
        else:
            batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RND(object):
    def __init__(self, obs_ph, is_training_ph, obs_space, action_space, logit_dim, model_cfg, sess, rnd_predictor_update_proportion=1.0):
        # set proportion of bonuses to actually use
        self._rnd_predictor_update_proportion = rnd_predictor_update_proportion

        # get handle to TF session for bonus inference
        self._sess = sess

        # get handles to RND net placeholders for bonus inference
        self._obs_ph = obs_ph
        self._is_training_ph = is_training_ph

        convfeat = 32
        enlargement = 2
        rep_size = 512
                                            # 42 42 1
        self.ob_rms = RunningMeanStd(shape=list(obs_space.shape[:2]) + [1], use_mpi=False)

        # self.ph_mean = tf.placeholder(dtype=tf.float32, shape=(obs_ph.shape[1], obs_ph.shape[2], 1), name="obmean")
        # self.ph_std = tf.placeholder(dtype=tf.float32, shape=(obs_ph.shape[1], obs_ph.shape[2], 1), name="obstd")

        # build target and predictor networks
        logger.info("Building RND networks...")
        with tf.variable_scope("rnd"):
            with tf.variable_scope("target"):

                xr = obs_ph
                xr = tf.cast(xr, tf.float32)
                xr = xr[:, :, :, -1:]
                # xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]

                self._targets = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))


            self._targets = tf.stop_gradient(self._targets) # freeze target network

            with tf.variable_scope("predictor"):
                xr = obs_ph
                xr = tf.cast(xr, tf.float32)
                xr = xr[:, :, :, -1:]
                # xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xr, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)

                X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                self._preds  = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))


        # build intr reward (bonus)
        self._intr_rew = self._build_intr_reward()

        # build loss for random network
        self._loss = self._build_loss()

        comm = MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None
        self.rff_int = RewardForwardFilter(0.99)
        self.rff_rms_int = RunningMeanStd(comm=comm, use_mpi=True)

    @property
    def loss(self):
        return self._loss

    def _build_intr_reward(self):
        logger.info('Building RND intrinisic reward...')
        intr_rew = tf.reduce_mean(tf.square(self._preds - self._targets), axis=-1, keep_dims=True)

        return intr_rew

    def compute_intr_rew(self, obs):
        self.ob_rms.update(obs[:, :, :, -1:])



        feed_dict = {
            self._obs_ph: obs,
            # self.ph_mean: self.ob_rms.mean,
            # self.ph_std: self.ob_rms.var ** 0.5,
            self._is_training_ph: False,
        }

        sess_obs = self._sess.run(self._intr_rew, feed_dict=feed_dict).T
        rffs_int = np.array([self.rff_int.update(rew) for rew in sess_obs.T])
        self.rff_rms_int.update(rffs_int.ravel())
        rews_int = sess_obs / np.sqrt(self.rff_rms_int.var)

        return np.squeeze(rews_int)

    def _build_loss(self):
        logger.info('Building RND loss...')
        loss = tf.reduce_mean(tf.square(self._preds - self._targets), axis=-1)
        keep_mask = tf.random_uniform(shape=tf.shape(loss), minval=0.0, maxval=1.0, dtype=tf.float32)
        keep_mask = tf.cast(keep_mask < self._rnd_predictor_update_proportion, tf.float32)
        loss = tf.reduce_sum(loss * keep_mask) / tf.maximum(tf.reduce_sum(keep_mask), 1.0)
        return loss


