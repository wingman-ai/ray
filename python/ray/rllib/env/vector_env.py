from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.env.entangled_env import EntangledEnv


@PublicAPI
class VectorEnv(object):
    """An environment that supports batch evaluation.

    Subclasses must define the following attributes:

    Attributes:
        action_space (gym.Space): Action space of individual envs.
        observation_space (gym.Space): Observation space of individual envs.
        num_envs (int): Number of envs in this vector env.
    """

    @staticmethod
    def wrap(make_env=None, num_envs=1, remote_envs=False,
             entangled_envs=False, action_space=None, observation_space=None):
        if remote_envs:
            return _RemoteVectorizedGymEnv(make_env, num_envs,
                                           action_space, observation_space)
        if entangled_envs:
            return _EntangledVectorizedGymEnv(make_env, num_envs,
                                              action_space, observation_space)
        return _VectorizedGymEnv(make_env, [], num_envs,
                                 action_space, observation_space)

    @PublicAPI
    def vector_reset(self):
        """Resets all environments.

        Returns:
            obs (list): Vector of observations from each environment.
        """
        raise NotImplementedError

    @PublicAPI
    def reset_at(self, index):
        """Resets a single environment.

        Returns:
            obs (obj): Observations from the resetted environment.
        """
        raise NotImplementedError

    @PublicAPI
    def vector_step(self, actions):
        """Vectorized step.

        Arguments:
            actions (list): Actions for each env.

        Returns:
            obs (list): New observations for each env.
            rewards (list): Reward values for each env.
            dones (list): Done values for each env.
            infos (list): Info values for each env.
        """
        raise NotImplementedError

    @PublicAPI
    def get_unwrapped(self):
        """Returns the underlying env instances."""
        raise NotImplementedError


class _VectorizedGymEnv(VectorEnv):
    """Internal wrapper for gym envs to implement VectorEnv.

    Arguments:
        make_env (func|None): Factory that produces a new gym env. Must be
            defined if the number of existing envs is less than num_envs.
        existing_envs (list): List of existing gym envs.
        num_envs (int): Desired num gym envs to keep total.
    """

    def __init__(self, make_env, existing_envs, num_envs,
                 action_space=None, observation_space=None):
        self.make_env = make_env
        self.envs = existing_envs
        self.num_envs = num_envs
        while len(self.envs) < self.num_envs:
            self.envs.append(self.make_env(len(self.envs)))
        self.action_space = action_space or self.envs[0].action_space
        self.observation_space = observation_space or \
            self.envs[0].observation_space

    @override(VectorEnv)
    def vector_reset(self):
        return [e.reset() for e in self.envs]

    @override(VectorEnv)
    def reset_at(self, index):
        return self.envs[index].reset()

    @override(VectorEnv)
    def vector_step(self, actions):
        obs_batch, rew_batch, done_batch, info_batch = [], [], [], []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(actions[i])
            obs_batch.append(obs)
            rew_batch.append(rew)
            done_batch.append(done)
            info_batch.append(info)
        return obs_batch, rew_batch, done_batch, info_batch

    @override(VectorEnv)
    def get_unwrapped(self):
        return self.envs


class _RemoteVectorizedGymEnv(_VectorizedGymEnv):
    """Internal wrapper for gym envs to implement VectorEnv as remote workers.
    """

    def __init__(self, make_env, num_envs,
                 action_space=None, observation_space=None):
        _VectorizedGymEnv.__init__(self, make_env, [], num_envs,
                                   action_space, observation_space)

        for env in range(self.envs):
            assert isinstance(env, ray.actor.ActorClass), \
                "Your environment needs to be ray remote environment"

    @override(_VectorizedGymEnv)
    def vector_reset(self):
        return ray.get([env.reset.remote() for env in self.envs])

    @override(_VectorizedGymEnv)
    def reset_at(self, index):
        return ray.get(self.envs[index].reset.remote())

    @override(_VectorizedGymEnv)
    def vector_step(self, actions):
        step_outs = ray.get([env.step.remote(act)
                             for env, act in zip(self.envs, actions)])

        obs_batch, rew_batch, done_batch, info_batch = [], [], [], []
        for obs, rew, done, info in step_outs:
            obs_batch.append(obs)
            rew_batch.append(rew)
            done_batch.append(done)
            info_batch.append(info)
        return obs_batch, rew_batch, done_batch, info_batch


class _EntangledVectorizedGymEnv(_VectorizedGymEnv):
    """Internal wrapper for gym env to implement VectorEnv as entangled env
    (logically different envs that share the same physical env)."""

    def __init__(self, make_env, num_envs,
                 action_space=None, observation_space=None):
        self.make_env = make_env
        self.num_envs = num_envs
        self.env = self.make_env(num_envs)
        assert isinstance(self.env, EntangledEnv), \
            "Your environment needs to inherit EntangledEnv"

        self.action_space = action_space or self.env.action_space
        self.observation_space = observation_space or \
            self.env.observation_space

    @override(_VectorizedGymEnv)
    def vector_reset(self):
        return [self.env.reset(env_i) for env_i in range(self.num_envs)]

    @override(_VectorizedGymEnv)
    def reset_at(self, env_i):
        return self.env.reset(env_i)

    @override(_VectorizedGymEnv)
    def vector_step(self, actions):
        step_results = self.env.step(
            {i: act for i, act in enumerate(actions)})

        obs_batch, rew_batch, done_batch, info_batch = [], [], [], []
        for env_i in range(self.num_envs):
            obs, rew, done, info = step_results[env_i]
            obs_batch.append(obs)
            rew_batch.append(rew)
            done_batch.append(done)
            info_batch.append(info)
        return obs_batch, rew_batch, done_batch, info_batch

    @override(_VectorizedGymEnv)
    def get_unwrapped(self):
        return [self.env] * self.num_envs
