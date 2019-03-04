#!/usr/bin/env bash

# Cause the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

MEMORY_SIZE="20G"
SHM_SIZE="20G"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

DOCKER_SHA=$($ROOT_DIR/../../build-docker.sh --output-sha --no-cache)
echo "Using Docker image" $DOCKER_SHA

######################## RLLIB TESTS #################################

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/train.py \
    --env CartPole-v0 \
    --run MARWIL \
    --stop '{"training_iteration": 2}' \
    --config '{"input": "/ray/python/ray/rllib/test/data/cartpole_small", "learning_starts": 0, "input_evaluation": ["wis", "is"], "shuffle_buffer_size": 10}'

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/train.py \
    --env CartPole-v0 \
    --run DQN \
    --stop '{"training_iteration": 2}' \
    --config '{"input": "/ray/python/ray/rllib/test/data/cartpole_small", "learning_starts": 0, "input_evaluation": ["wis", "is"], "soft_q": true}'

######################## TUNE TESTS #################################

bash $ROOT_DIR/run_tune_tests.sh ${MEMORY_SIZE} ${SHM_SIZE} $DOCKER_SHA

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_io.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_checkpoint_restore.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_policy_evaluator.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_nested_spaces.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_external_env.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/parametric_action_cartpole.py --run=PG --stop=50

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/parametric_action_cartpole.py --run=PPO --stop=50

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/parametric_action_cartpole.py --run=DQN --stop=50

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/custom_loss.py --iters=2

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_lstm.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/batch_norm_model.py --num-iters=1 --run=PPO

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/batch_norm_model.py --num-iters=1 --run=PG

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/batch_norm_model.py --num-iters=1 --run=DQN

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/batch_norm_model.py --num-iters=1 --run=DDPG

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_multi_agent_env.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_supported_spaces.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_env_with_subprocess.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    /ray/python/ray/rllib/test/test_rollout.sh

# Run all single-agent regression tests (3x retry each)
for yaml in $(ls $ROOT_DIR/../../python/ray/rllib/tuned_examples/regression_tests); do
    docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
        python /ray/python/ray/rllib/test/run_regression_tests.py /ray/python/ray/rllib/tuned_examples/regression_tests/$yaml
done

# Try a couple times since it's stochastic
docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
        python /ray/python/ray/rllib/test/multiagent_pendulum.py || \
    docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
        python /ray/python/ray/rllib/test/multiagent_pendulum.py || \
    docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
        python /ray/python/ray/rllib/test/multiagent_pendulum.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/multiagent_cartpole.py --num-iters=2

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/multiagent_two_trainers.py --num-iters=2

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/test/test_avail_actions_qmix.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/cartpole_lstm.py --run=PPO --stop=200

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/cartpole_lstm.py --run=IMPALA --stop=100

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/cartpole_lstm.py --stop=200 --use-prev-action-reward

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/custom_metrics_and_callbacks.py --num-iters=2

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/contrib/random_agent/random_agent.py

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/twostep_game.py --stop=2000 --run=PG

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/twostep_game.py --stop=2000 --run=QMIX

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/examples/twostep_game.py --stop=2000 --run=APEX_QMIX

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/experimental/sgd/test_sgd.py --num-iters=2 \
        --batch-size=1 --strategy=simple

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/experimental/sgd/test_sgd.py --num-iters=2 \
        --batch-size=1 --strategy=ps

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/experimental/sgd/test_save_and_restore.py --num-iters=2 \
        --batch-size=1 --strategy=simple

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/experimental/sgd/test_save_and_restore.py --num-iters=2 \
        --batch-size=1 --strategy=ps

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/experimental/sgd/mnist_example.py --num-iters=1 \
        --num-workers=1 --devices-per-worker=1 --strategy=ps

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/experimental/sgd/mnist_example.py --num-iters=1 \
        --num-workers=1 --devices-per-worker=1 --strategy=ps --tune

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/train.py \
    --env PongDeterministic-v4 \
    --run A3C \
    --stop '{"training_iteration": 2}' \
    --config '{"num_workers": 2, "use_pytorch": true, "sample_async": false, "model": {"use_lstm": false, "grayscale": true, "zero_mean": false, "dim": 84}, "preprocessor_pref": "rllib"}'

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/train.py \
    --env CartPole-v1 \
    --run A3C \
    --stop '{"training_iteration": 2}' \
    --config '{"num_workers": 2, "use_pytorch": true, "sample_async": false}'

docker run --rm --shm-size=${SHM_SIZE} --memory=${MEMORY_SIZE} $DOCKER_SHA \
    python /ray/python/ray/rllib/train.py \
    --env PongDeterministic-v4 \
    --run IMPALA \
    --stop='{"timesteps_total": 40000}' \
    --ray-object-store-memory=500000000 \
    --config '{"num_workers": 1, "num_gpus": 0, "num_envs_per_worker": 64, "sample_batch_size": 50, "train_batch_size": 50, "learner_queue_size": 1}'

python3 $ROOT_DIR/multi_node_docker_test.py \
    --docker-image=$DOCKER_SHA \
    --num-nodes=5 \
    --num-redis-shards=10 \
    --test-script=/ray/test/jenkins_tests/multi_node_tests/test_0.py

python3 $ROOT_DIR/multi_node_docker_test.py \
    --docker-image=$DOCKER_SHA \
    --num-nodes=5 \
    --num-redis-shards=5 \
    --num-gpus=0,1,2,3,4 \
    --num-drivers=7 \
    --driver-locations=0,1,0,1,2,3,4 \
    --test-script=/ray/test/jenkins_tests/multi_node_tests/remove_driver_test.py

python3 $ROOT_DIR/multi_node_docker_test.py \
    --docker-image=$DOCKER_SHA \
    --num-nodes=5 \
    --num-redis-shards=2 \
    --num-gpus=0,0,5,6,50 \
    --num-drivers=100 \
    --test-script=/ray/test/jenkins_tests/multi_node_tests/many_drivers_test.py

python3 $ROOT_DIR/multi_node_docker_test.py \
    --docker-image=$DOCKER_SHA \
    --num-nodes=1 \
    --mem-size=60G \
    --shm-size=60G \
    --test-script=/ray/ci/jenkins_tests/multi_node_tests/large_memory_test.py
