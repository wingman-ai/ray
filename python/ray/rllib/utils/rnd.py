""" Implementation of Random Network Distillation (RND).
Original Paper: https://arxiv.org/abs/1810.12894

Author
------
Vishal Satish
"""
import logging

import numpy as np
import tensorflow as tf

from ray.rllib.models.catalog import ModelCatalog
from mpi4py import MPI


logger = logging.getLogger(__name__)

class RND(object):
    def __init__(self, obs_ph, is_training_ph, obs_space, action_space, logit_dim, model_cfg, sess, rnd_predictor_update_proportion=1.0):
        # set proportion of bonuses to actually use
        self._rnd_predictor_update_proportion = rnd_predictor_update_proportion

        # get handle to TF session for bonus inference
        self._sess = sess

        # get handles to RND net placeholders for bonus inference
        self._obs_ph = obs_ph
        self._is_training_ph = is_training_ph

        # normalize obs
#        obs_ph = tf.layers.batch_normalization(obs_ph, training=is_training_ph)

        # clip obs to [-5.0, 5.0]
#        obs_ph = tf.clip_by_value(obs_ph, -5.0, 5.0)

        # build target and predictor networks
        logger.info("Building RND networks...")
        with tf.variable_scope("rnd"):
            with tf.variable_scope("target"):
                self._targets = ModelCatalog.get_model(
                    input_dict={
                        "obs": obs_ph,
                        "is_training": is_training_ph,
                    },
                    obs_space=obs_space,
                    action_space=action_space,
                    num_outputs=logit_dim,
                    options=model_cfg).outputs
            self._targets = tf.stop_gradient(self._targets) # freeze target network

            with tf.variable_scope("predictor"):
                self._preds = ModelCatalog.get_model(
                    {
                        "obs": obs_ph,
                        "is_training": is_training_ph,
                    },
                    obs_space,
                    action_space,
                    logit_dim,
                    model_cfg).outputs

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
        # normalize intr reward
#        intr_rew = tf.layers.batch_normalization(intr_rew, training=self._is_training_ph)
        return intr_rew

    def compute_intr_rew(self, obs):
        feed_dict = {self._obs_ph: obs, self._is_training_ph: False}
        out_one = self._sess.run(self._intr_rew, feed_dict=feed_dict)
        out_one = out_one.T
        rffs_int = np.array([self.rff_int.update(rew) for rew in out_one.T])
        self.rff_rms_int.update(rffs_int.ravel())
        rews_int = out_one / np.sqrt(self.rff_rms_int.var)

        return np.squeeze(rews_int)

    def _build_loss(self):
        logger.info('Building RND loss...')
        loss = tf.reduce_mean(tf.square(self._preds - self._targets), axis=-1)
        keep_mask = tf.random_uniform(shape=tf.shape(loss), minval=0.0, maxval=1.0, dtype=tf.float32)
        keep_mask = tf.cast(keep_mask < self._rnd_predictor_update_proportion, tf.float32)
        loss = tf.reduce_sum(loss * keep_mask) / tf.maximum(tf.reduce_sum(keep_mask), 1.0)
        return loss

def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n+1, x.dtype)
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
        newshape = mean.shape[:axis] + mean.shape[axis+1:]
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
        return self.rewems



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
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
