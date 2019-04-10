"""Adapted from A3CPolicyGraph to add V-trace.

Keep in sync with changes to A3CPolicyGraph and VtraceSurrogatePolicyGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import ray
import tensorflow as tf
from ray.rllib.agents.impala import vtrace
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.action_dist import MultiCategorical, Categorical
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.explained_variance import explained_variance

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"


class VTraceLoss(object):
    def __init__(self,
                 actions,
                 actions_logp,
                 actions_entropy,
                 dones,
                 behaviour_logits,
                 target_logits,
                 discount,
                 rewards,
                 values,
                 bootstrap_value,
                 valid_mask,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0):
        """Policy gradient loss with vtrace importance weighting.

        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.

        Args:
            actions: An int32 tensor of shape [T, B, ACTION_SPACE].
            actions_logp: A float32 tensor of shape [T, B].
            actions_entropy: A float32 tensor of shape [T, B].
            dones: A bool tensor of shape [T, B].
            behaviour_logits: A list with length of ACTION_SPACE of float32
                tensors of shapes
                [T, B, ACTION_SPACE[0]],
                ...,
                [T, B, ACTION_SPACE[-1]]
            target_logits: A list with length of ACTION_SPACE of float32
                tensors of shapes
                [T, B, ACTION_SPACE[0]],
                ...,
                [T, B, ACTION_SPACE[-1]]
            discount: A float32 scalar.
            rewards: A float32 tensor of shape [T, B].
            values: A float32 tensor of shape [T, B].
            bootstrap_value: A float32 tensor of shape [B].
            valid_mask: A bool tensor of valid RNN input elements (#2992).
        """

        # Compute vtrace on the CPU for better perf.
        with tf.device("/cpu:0"):
            self.vtrace_returns = vtrace.multi_from_logits(
                behaviour_policy_logits=behaviour_logits,
                target_policy_logits=target_logits,
                actions=tf.unstack(tf.cast(actions, tf.int32), axis=2),
                discounts=tf.to_float(~dones) * discount,
                rewards=rewards,
                values=values,
                bootstrap_value=bootstrap_value,
                clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32),
                clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold,
                                              tf.float32))

        # The policy gradients loss
        self.pi_loss = -tf.reduce_sum(
            tf.boolean_mask(actions_logp * self.vtrace_returns.pg_advantages,
                            valid_mask), name='pi_loss')

        # The baseline loss
        delta = tf.boolean_mask(values - self.vtrace_returns.vs, valid_mask)
        self.vf_loss = tf.math.multiply(0.5, tf.reduce_sum(tf.square(delta)), name='vf_loss')

        # The entropy loss
        self.entropy = tf.reduce_sum(
            tf.boolean_mask(actions_entropy, valid_mask), name='entropy_loss')

        # The summed weighted loss
        self.total_loss = tf.math.add(self.pi_loss, self.vf_loss * vf_loss_coeff - self.entropy * entropy_coeff,
                                      name='total_loss')


class VTracePostprocessing(object):
    """Adds the policy logits to the trajectory."""

    @override(TFPolicyGraph)
    def extra_compute_action_fetches(self):
        return dict(
            TFPolicyGraph.extra_compute_action_fetches(self),
            **{BEHAVIOUR_LOGITS: self.model.outputs})

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        # not used, so save some bandwidth
        del sample_batch.data[SampleBatch.NEXT_OBS]
        return sample_batch


class VTracePolicyGraph(LearningRateSchedule, VTracePostprocessing,
                        TFPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 existing_inputs=None):
        config = dict(ray.rllib.agents.impala.impala.DEFAULT_CONFIG, **config)
        assert config["batch_mode"] == "truncate_episodes", \
            "Must use `truncate_episodes` batch mode with V-trace."
        self.config = config
        self.sess = tf.get_default_session()
        self.grads = None

        if isinstance(action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            actions_shape = [None]
            output_hidden_shape = [action_space.n]
        elif isinstance(action_space, gym.spaces.multi_discrete.MultiDiscrete):
            is_multidiscrete = True
            actions_shape = [None, len(action_space.nvec)]
            output_hidden_shape = action_space.nvec.astype(np.int32)
        else:
            raise UnsupportedSpaceException(
                "Action space {} is not supported for IMPALA.".format(
                    action_space))

        # Create input placeholders
        if existing_inputs:
            actions, dones, behaviour_logits, rewards, observations, \
                prev_actions, prev_rewards = existing_inputs[:7]
            existing_state_in = existing_inputs[7:-1]
            existing_seq_lens = existing_inputs[-1]
        else:
            actions = tf.placeholder(tf.int64, actions_shape, name="ac")
            dones = tf.placeholder(tf.bool, [None], name="dones")
            rewards = tf.placeholder(tf.float32, [None], name="rewards")
            behaviour_logits = tf.placeholder(
                tf.float32, [None, sum(output_hidden_shape)],
                name="behaviour_logits")
            observations = tf.placeholder(
                tf.float32, [None] + list(observation_space.shape),
                name='observations')
            existing_state_in = None
            existing_seq_lens = None

        # Unpack behaviour logits
        unpacked_behaviour_logits = tf.split(
            behaviour_logits, output_hidden_shape, axis=1)

        # Setup the policy
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        prev_actions = ModelCatalog.get_action_placeholder(action_space)
        prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")
        self.model = ModelCatalog.get_model(
            {
                "obs": observations,
                "prev_actions": prev_actions,
                "prev_rewards": prev_rewards,
                "is_training": self._get_is_training_placeholder(),
            },
            observation_space,
            action_space,
            logit_dim,
            self.config["model"],
            state_in=existing_state_in,
            seq_lens=existing_seq_lens)
        unpacked_outputs = tf.split(
            self.model.outputs, output_hidden_shape, axis=1)

        dist_inputs = unpacked_outputs if is_multidiscrete else \
            self.model.outputs
        action_dist = dist_class(dist_inputs)

        values = self.model.value_function()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

        def _make_time_major(tensor, drop_last=False):
            """Swaps batch and trajectory axis.
            Args:
                tensor: A tensor or list of tensors to reshape.
                drop_last: A bool indicating whether to drop the last
                trajectory item.
            Returns:
                res: A tensor with swapped axes or a list of tensors with
                swapped axes.
            """
            if isinstance(tensor, list):
                return [_make_time_major(t, drop_last) for t in tensor]

            if self.model.state_init:
                B = tf.shape(self.model.seq_lens)[0]
                T = tf.shape(tensor)[0] // B
            else:
                # Important: chop the tensor into batches at known episode cut
                # boundaries. TODO(ekl) this is kind of a hack
                T = self.config["sample_batch_size"]
                B = tf.shape(tensor)[0] // T
            rs = tf.reshape(tensor,
                            tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))

            # swap B and T axes
            res = tf.transpose(
                rs,
                [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))

            if drop_last:
                return res[:-1]
            return res

        def make_time_major(tensor, drop_last=False):
            with tf.name_scope('make_time_major'):
                return _make_time_major(tensor, drop_last=False)

        if self.model.state_in:
            max_seq_len = tf.reduce_max(self.model.seq_lens) - 1
            mask = tf.sequence_mask(self.model.seq_lens, max_seq_len)
            mask = tf.reshape(mask, [-1])
        else:
            mask = tf.ones_like(rewards, dtype=tf.bool)

        # Prepare actions for loss
        loss_actions = actions if is_multidiscrete else tf.expand_dims(
            actions, axis=1)

        if isinstance(action_dist, MultiCategorical):
            action_dist_inputs = list(map(lambda c: c.inputs, action_dist.cats))
        elif isinstance(action_dist, Categorical):
            action_dist_inputs = [action_dist.inputs]
        else:
            raise ValueError('Not supported action_dist distribution')

        # Inputs are reshaped from [B * T] => [T - 1, B] for V-trace calc.
        with tf.name_scope('vtrace_loss'):
            self.loss = VTraceLoss(
                actions=make_time_major(loss_actions, drop_last=True),
                actions_logp=make_time_major(
                    action_dist.logp(actions), drop_last=True),
                actions_entropy=make_time_major(
                    action_dist.entropy(), drop_last=True),
                dones=make_time_major(dones, drop_last=True),
                behaviour_logits=make_time_major(
                    unpacked_behaviour_logits, drop_last=True),
                target_logits=make_time_major(unpacked_outputs, drop_last=True),
                discount=config["gamma"],
                rewards=make_time_major(rewards, drop_last=True),
                values=make_time_major(values, drop_last=True),
                bootstrap_value=make_time_major(values)[-1],
                valid_mask=make_time_major(mask, drop_last=True),
                vf_loss_coeff=self.config["vf_loss_coeff"],
                entropy_coeff=self.config["entropy_coeff"],
                clip_rho_threshold=self.config["vtrace_clip_rho_threshold"],
                clip_pg_rho_threshold=self.config["vtrace_clip_pg_rho_threshold"])

        # Initialize TFPolicyGraph
        loss_in = [
            (SampleBatch.ACTIONS, actions),
            (SampleBatch.DONES, dones),
            (BEHAVIOUR_LOGITS, behaviour_logits),
            (SampleBatch.REWARDS, rewards),
            (SampleBatch.CUR_OBS, observations),
            (SampleBatch.PREV_ACTIONS, prev_actions),
            (SampleBatch.PREV_REWARDS, prev_rewards),
        ]

        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])

        with tf.name_scope('TFPolicyGraph.__init__'):
            TFPolicyGraph.__init__(
                self,
                observation_space,
                action_space,
                self.sess,
                obs_input=observations,
                action_sampler=action_dist.sample(),
                action_prob=action_dist.sampled_action_prob(),
                loss=self.loss.total_loss,
                model=self.model,
                loss_inputs=loss_in,
                state_inputs=self.model.state_in,
                state_outputs=self.model.state_out,
                prev_action_input=prev_actions,
                prev_reward_input=prev_rewards,
                seq_lens=self.model.seq_lens,
                max_seq_len=self.config["model"]["max_seq_len"],
                batch_divisibility_req=self.config["sample_batch_size"],
                values=values)

        with tf.name_scope('stats_fetches'):
            # KL divergence between worker and learner logits for debugging
            model_dist = MultiCategorical(unpacked_outputs)
            behaviour_dist = MultiCategorical(unpacked_behaviour_logits)

            kls = model_dist.kl(behaviour_dist)
            if len(kls) > 1:
                self.KL_stats = {}

                for i, kl in enumerate(kls):
                    self.KL_stats.update({
                        "mean_KL_{}".format(i): tf.reduce_mean(kl),
                        "max_KL_{}".format(i): tf.reduce_max(kl),
                        "median_KL_{}".format(i): tf.contrib.distributions.
                        percentile(kl, 50.0),
                    })
            else:
                self.KL_stats = {
                    "mean_KL": tf.reduce_mean(kls[0]),
                    "max_KL": tf.reduce_max(kls[0]),
                    "median_KL": tf.contrib.distributions.percentile(kls[0], 50.0),
                }

            action_probs = [tf.nn.softmax(adi, axis=1) for adi in action_dist_inputs]
            action_probs_means = [tf.reduce_mean(ap, axis=0) for ap in action_probs]
            action_probs_max = [tf.reduce_max(apm) for apm in action_probs_means]
            action_probs_min = [tf.reduce_min(apm) for apm in action_probs_means]
            values_mean = tf.reduce_mean(values)
            vtrace_pg_advantages_mean = tf.reduce_mean(self.loss.vtrace_returns.pg_advantages)
            vtrace_vs_mean = tf.reduce_mean(self.loss.vtrace_returns.vs)

            pi_loss_abs = tf.abs(self.loss.pi_loss)
            vf_loss_abs = tf.abs(self.loss.vf_loss * self.config["vf_loss_coeff"])
            en_loss_abs = tf.abs(self.loss.entropy * self.config["entropy_coeff"])
            total_influence_loss = pi_loss_abs + vf_loss_abs + en_loss_abs

            if hasattr(self.model, 'lp_loss'):
                lp_loss_abs = tf.abs(self.model.lp_loss)
                total_influence_loss += lp_loss_abs
            if hasattr(self.model, 'tae_loss'):
                tae_loss_abs = tf.abs(self.model.tae_loss)
                total_influence_loss += tae_loss_abs

            influence_stats = {
                "_influence_total_loss": total_influence_loss,
                "_influence_policy_loss": pi_loss_abs / total_influence_loss,
                "_influence_vf_loss": vf_loss_abs / total_influence_loss,
                "_influence_entropy_loss": en_loss_abs / total_influence_loss,
            }

            if hasattr(self.model, 'lp_loss'):
                influence_stats.update({
                    "influence_lp_loss": lp_loss_abs / total_influence_loss,
                })
            if hasattr(self.model, 'tae_loss'):
                influence_stats.update({
                    "influence_tae_loss": tae_loss_abs / total_influence_loss,
                })

            self.sess.run(tf.global_variables_initializer())

            self.stats_fetches = {
                LEARNER_STATS_KEY: dict({
                    "actions": {
                        **dict([(f'action_probs_min{i}', action_probs_min[i]) for i in range(len(action_probs_min))]),
                        **dict([(f'action_probs_max{i}', action_probs_max[i]) for i in range(len(action_probs_max))]),
                    },
                    "activations_abs_max": dict([(f'{v.name}', tf.reduce_max(tf.math.abs(v))) for _, v in self._grads_and_vars]),
                    "gradients_abs_max": dict([(f'{v.name}', tf.reduce_max(tf.math.abs(g))) for g, v in self._grads_and_vars]),
                    "histograms": {
                        "activations": dict([(f'{v.name}', v) for _, v in self._grads_and_vars]),
                        "gradients": dict([(f'{v.name}', g) for g, v in self._grads_and_vars]),
                    },
                    "monitoring": {
                        "cur_lr": tf.cast(self.cur_lr, tf.float64),
                        "grad_gnorm": tf.global_norm(self._grads),
                        "var_gnorm": tf.global_norm(self.var_list),
                        "vf_explained_var": explained_variance(
                            tf.reshape(self.loss.vtrace_returns.vs, [-1]),
                            tf.reshape(make_time_major(values, drop_last=True), [-1])),
                        "values_mean": values_mean,
                        "vtrace_pg_advantages_mean": vtrace_pg_advantages_mean,
                        "vtrace_vs_mean": vtrace_vs_mean,
                        **self.KL_stats,
                    },
                    "losses": {
                        "total_loss": self.loss.total_loss,
                        "policy_loss": self.loss.pi_loss,
                        "vf_loss": self.loss.vf_loss * self.config["vf_loss_coeff"],
                        "entropy_loss": self.loss.entropy * self.config["entropy_coeff"],
                        **influence_stats,
                    },
                }),
            }

    @override(TFPolicyGraph)
    def copy(self, existing_inputs):
        return VTracePolicyGraph(
            self.observation_space,
            self.action_space,
            self.config,
            existing_inputs=existing_inputs)

    @override(TFPolicyGraph)
    def optimizer(self):
        if self.config["opt_type"] == "adam":
            return tf.train.AdamOptimizer(self.cur_lr)
        else:
            return tf.train.RMSPropOptimizer(self.cur_lr, self.config["decay"],
                                             self.config["momentum"],
                                             self.config["epsilon"])

    @override(TFPolicyGraph)
    def gradients(self, optimizer, loss):
        grads = tf.gradients(loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        clipped_grads = list(zip(self.grads, self.var_list))
        return clipped_grads

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    @override(PolicyGraph)
    def get_initial_state(self):
        return self.model.state_init
