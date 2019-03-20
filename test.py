import ray
from ray.rllib.agents.impala import ImpalaAgent

ray.init()

agent = ImpalaAgent(config={
    'num_workers': 2,
    'num_envs_per_worker': 2,
    'num_gpus': 1,
    'rnd': 1,
    'entropy_coeff': 0.05,
    'vf_loss_coeff': 0.5,
}, env='PongDeterministic-v4')

while True:
    print(agent.train())