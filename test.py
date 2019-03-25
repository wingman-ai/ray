import ray
from ray.rllib.agents.impala import ImpalaAgent

ray.init(object_store_memory=int(3e9), redis_max_memory=int(1e8))

agent = ImpalaAgent(config={
    'num_workers': 1,
    'num_envs_per_worker': 3,
    'num_gpus': 1,
    'rnd': 1,
    'vf_loss_coeff': 0.5,
    'entropy_coeff': 0.2,
    # 'lr': 0.0003,
    'model': {'dim': 42},
    'sample_batch_size': 5,
}, env='PongNoFrameskip-v4')

while True:
    print(agent.train())