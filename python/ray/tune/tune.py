from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import logging
import time

from ray.tune.error import TuneError
from ray.tune.experiment import convert_to_experiment_list, Experiment
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.suggest import BasicVariantGenerator
from ray.tune.trial import Trial, DEBUG_PRINT_INTERVAL
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.syncer import wait_for_sync
from ray.tune.trial_runner import TrialRunner
from ray.tune.schedulers import (HyperBandScheduler, AsyncHyperBandScheduler,
                                 FIFOScheduler, MedianStoppingRule)
from ray.tune.web_server import TuneServer

logger = logging.getLogger(__name__)

_SCHEDULERS = {
    "FIFO": FIFOScheduler,
    "MedianStopping": MedianStoppingRule,
    "HyperBand": HyperBandScheduler,
    "AsyncHyperBand": AsyncHyperBandScheduler,
}


def _make_scheduler(args):
    if args.scheduler in _SCHEDULERS:
        return _SCHEDULERS[args.scheduler](**args.scheduler_config)
    else:
        raise TuneError("Unknown scheduler: {}, should be one of {}".format(
            args.scheduler, _SCHEDULERS.keys()))


def _prompt_restore(checkpoint_dir, resume):
    restore = False
    if TrialRunner.checkpoint_exists(checkpoint_dir):
        if resume == "prompt":
            msg = ("Found incomplete experiment at {}. "
                   "Would you like to resume it?".format(checkpoint_dir))
            restore = click.confirm(msg, default=False)
            if restore:
                logger.info("Tip: to always resume, "
                            "pass resume=True to run()")
            else:
                logger.info("Tip: to always start a new experiment, "
                            "pass resume=False to run()")
        elif resume:
            restore = True
        else:
            logger.info("Tip: to resume incomplete experiments, "
                        "pass resume='prompt' or resume=True to run()")
    else:
        logger.info(
            "Did not find checkpoint file in {}.".format(checkpoint_dir))
    return restore


def run(run_or_experiment,
        name=None,
        stop=None,
        config=None,
        resources_per_trial=None,
        num_samples=1,
        local_dir=None,
        upload_dir=None,
        trial_name_creator=None,
        loggers=None,
        sync_to_cloud=None,
        sync_to_driver=None,
        checkpoint_freq=0,
        checkpoint_at_end=False,
        keep_checkpoints_num=None,
        checkpoint_score_attr=None,
        global_checkpoint_period=10,
        export_formats=None,
        max_failures=3,
        restore=None,
        search_alg=None,
        scheduler=None,
        with_server=False,
        server_port=TuneServer.DEFAULT_PORT,
        verbose=2,
        resume=False,
        queue_trials=False,
        reuse_actors=False,
        trial_executor=None,
        raise_on_failed_trial=True,
        return_trials=False,
        ray_auto_init=True,
        sync_function=None):
    """Executes training.

    Args:
        run_or_experiment (function|class|str|Experiment): If
            function|class|str, this is the algorithm or model to train.
            This may refer to the name of a built-on algorithm
            (e.g. RLLib's DQN or PPO), a user-defined trainable
            function or class, or the string identifier of a
            trainable function or class registered in the tune registry.
            If Experiment, then Tune will execute training based on
            Experiment.spec.
        name (str): Name of experiment.
        stop (dict): The stopping criteria. The keys may be any field in
            the return result of 'train()', whichever is reached first.
            Defaults to empty dict.
        config (dict): Algorithm-specific configuration for Tune variant
            generation (e.g. env, hyperparams). Defaults to empty dict.
            Custom search algorithms may ignore this.
        resources_per_trial (dict): Machine resources to allocate per trial,
            e.g. ``{"cpu": 64, "gpu": 8}``. Note that GPUs will not be
            assigned unless you specify them here. Defaults to 1 CPU and 0
            GPUs in ``Trainable.default_resource_request()``.
        num_samples (int): Number of times to sample from the
            hyperparameter space. Defaults to 1. If `grid_search` is
            provided as an argument, the grid will be repeated
            `num_samples` of times.
        local_dir (str): Local dir to save training results to.
            Defaults to ``~/ray_results``.
        upload_dir (str): Optional URI to sync training results
            to (e.g. ``s3://bucket``).
        trial_name_creator (func): Optional function for generating
            the trial string representation.
        loggers (list): List of logger creators to be used with
            each Trial. If None, defaults to ray.tune.logger.DEFAULT_LOGGERS.
            See `ray/tune/logger.py`.
        sync_to_cloud (func|str): Function for syncing the local_dir to and
            from upload_dir. If string, then it must be a string template
            that includes `{source}` and `{target}` for the syncer to run.
            If not provided, the sync command defaults to standard
            S3 or gsutil sync comamnds.
        sync_to_driver (func|str): Function for syncing trial logdir from
            remote node to local. If string, then it must be a string template
            that includes `{source}` and `{target}` for the syncer to run.
            If not provided, defaults to using rsync.
        checkpoint_freq (int): How many training iterations between
            checkpoints. A value of 0 (default) disables checkpointing.
        checkpoint_at_end (bool): Whether to checkpoint at the end of the
            experiment regardless of the checkpoint_freq. Default is False.
        keep_checkpoints_num (int): Number of checkpoints to keep. A value of
            `None` keeps all checkpoints. Defaults to `None`. If set, need
            to provide `checkpoint_score_attr`.
        checkpoint_score_attr (str): Specifies by which attribute to rank the
            best checkpoint. Default is increasing order. If attribute starts
            with `min-` it will rank attribute in decreasing order, i.e.
            `min-validation_loss`.
        global_checkpoint_period (int): Seconds between global checkpointing.
            This does not affect `checkpoint_freq`, which specifies frequency
            for individual trials.
        export_formats (list): List of formats that exported at the end of
            the experiment. Default is None.
        max_failures (int): Try to recover a trial from its last
            checkpoint at least this many times. Only applies if
            checkpointing is enabled. Setting to -1 will lead to infinite
            recovery retries. Defaults to 3.
        restore (str): Path to checkpoint. Only makes sense to set if
            running 1 trial. Defaults to None.
        search_alg (SearchAlgorithm): Search Algorithm. Defaults to
            BasicVariantGenerator.
        scheduler (TrialScheduler): Scheduler for executing
            the experiment. Choose among FIFO (default), MedianStopping,
            AsyncHyperBand, and HyperBand.
        with_server (bool): Starts a background Tune server. Needed for
            using the Client API.
        server_port (int): Port number for launching TuneServer.
        verbose (int): 0, 1, or 2. Verbosity mode. 0 = silent,
            1 = only status updates, 2 = status and trial results.
        resume (str|bool): One of "LOCAL", "REMOTE", "PROMPT", or bool.
            LOCAL/True restores the checkpoint from the local_checkpoint_dir.
            REMOTE restores the checkpoint from remote_checkpoint_dir.
            PROMPT provides CLI feedback. False forces a new
            experiment. If resume is set but checkpoint does not exist,
            ValueError will be thrown.
        queue_trials (bool): Whether to queue trials when the cluster does
            not currently have enough resources to launch one. This should
            be set to True when running on an autoscaling cluster to enable
            automatic scale-up.
        reuse_actors (bool): Whether to reuse actors between different trials
            when possible. This can drastically speed up experiments that start
            and stop actors often (e.g., PBT in time-multiplexing mode). This
            requires trials to have the same resource requirements.
        trial_executor (TrialExecutor): Manage the execution of trials.
        raise_on_failed_trial (bool): Raise TuneError if there exists failed
            trial (of ERROR state) when the experiments complete.
        ray_auto_init (bool): Automatically starts a local Ray cluster
            if using a RayTrialExecutor (which is the default) and
            if Ray is not initialized. Defaults to True.
        sync_function: Deprecated. See `sync_to_cloud` and
            `sync_to_driver`.

    Returns:
        List of Trial objects.

    Raises:
        TuneError if any trials failed and `raise_on_failed_trial` is True.

    Examples:
        >>> tune.run(mytrainable, scheduler=PopulationBasedTraining())

        >>> tune.run(mytrainable, num_samples=5, reuse_actors=True)

        >>> tune.run(
                "PG",
                num_samples=5,
                config={
                    "env": "CartPole-v0",
                    "lr": tune.sample_from(lambda _: np.random.rand())
                }
            )
    """
    trial_executor = trial_executor or RayTrialExecutor(
        queue_trials=queue_trials,
        reuse_actors=reuse_actors,
        ray_auto_init=ray_auto_init)
    experiment = run_or_experiment
    if not isinstance(run_or_experiment, Experiment):
        run_identifier = Experiment._register_if_needed(run_or_experiment)
        experiment = Experiment(
            name=name,
            run=run_identifier,
            stop=stop,
            config=config,
            resources_per_trial=resources_per_trial,
            num_samples=num_samples,
            local_dir=local_dir,
            upload_dir=upload_dir,
            sync_to_driver=sync_to_driver,
            trial_name_creator=trial_name_creator,
            loggers=loggers,
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=checkpoint_at_end,
            keep_checkpoints_num=keep_checkpoints_num,
            checkpoint_score_attr=checkpoint_score_attr,
            export_formats=export_formats,
            max_failures=max_failures,
            restore=restore,
            sync_function=sync_function)
    else:
        logger.debug("Ignoring some parameters passed into tune.run.")

    if sync_to_cloud:
        assert experiment.remote_checkpoint_dir, (
            "Need `upload_dir` if `sync_to_cloud` given.")

    checkpoint_dir = experiment.checkpoint_dir
    should_restore = _prompt_restore(checkpoint_dir, resume)

    runner = None
    if should_restore:
        try:
            runner = TrialRunner.restore(checkpoint_dir, search_alg, scheduler,
                                         trial_executor)

            import collections
            import sys
            import numpy as np

            def tae_lp_override_check_model(restored_value, new_value):
                restored_subnetwork_exists = not np.isclose(restored_value, 0)
                new_subnetwork_exists = not (np.isscalar(new_value) and np.isclose(new_value, 0))

                return restored_subnetwork_exists == new_subnetwork_exists

            def override_flags(restored_config, new_config, flags_to_override):
                for k, v in flags_to_override.items():
                    if isinstance(v, collections.Mapping):
                        override_flags(restored_config[k], new_config[k], flags_to_override[k])
                    elif v is None or callable(v):
                        if k in explicitly_defined_flags or k == '_fake_sampler':
                            assert v is None or v(restored_config[k], new_config[k]), "Restored and new model differ"
                            if any(isinstance(new_config[k], cls) for cls in [int, float, bool, str]):
                                restored_config[k] = new_config[k]
                    else:
                        raise ValueError('Values of flags_to_override dict should be dict, None or methods')

            restored_config = runner._trials[0].config
            new_config = experiment.spec['config']
            flags_to_override_to_model_check_dict = {
                'grad_clip': None,
                'lr': None,
                'entropy_coeff': None,
                'vf_loss_coeff': None,
                '_fake_sampler': None,
                'model': {
                    'custom_options': {
                        'lp_coeff': tae_lp_override_check_model,
                        'tae_coeff': tae_lp_override_check_model,
                        'debug': None,
                        'imit_coeff': None,
                        'dataset_path': None,
                        'val_dataset_path': None,
                    },
                },
                'num_workers': None,
                'num_envs_per_worker': None,
            }

            explicitly_defined_flags = set([arg[2:] for arg in sys.argv if arg.startswith('--')])
            override_flags(restored_config, new_config, flags_to_override_to_model_check_dict)

            if 'checkpoint_freq' in explicitly_defined_flags:
                runner._trials[0].checkpoint_freq = experiment.spec['checkpoint_freq']

            if runner._trials[0].status == Trial.ERROR:
                runner._trials[0].init_logger()
                runner._try_recover(runner._trials[0], error_msg=None)

            old_res = runner._trials[0].resources
            runner._trials[0].resources = Resources(old_res.cpu, old_res.gpu, 1.0, old_res.extra_gpu,
                                                    old_res.custom_resources, old_res.extra_custom_resources)
        except Exception:
            logger.exception("Runner restore failed. Restarting experiment.")
    else:
        logger.info("Starting a new experiment.")

    runner = TrialRunner(
        search_alg=search_alg or BasicVariantGenerator(),
        scheduler=scheduler or FIFOScheduler(),
        local_checkpoint_dir=experiment.checkpoint_dir,
        remote_checkpoint_dir=experiment.remote_checkpoint_dir,
        sync_to_cloud=sync_to_cloud,
        checkpoint_period=global_checkpoint_period,
        resume=resume,
        launch_web_server=with_server,
        server_port=server_port,
        verbose=bool(verbose > 1),
        trial_executor=trial_executor)

    runner.add_experiment(experiment)

    if verbose:
        print(runner.debug_string(max_debug=99999))

    last_debug = 0
    while not runner.is_finished():
        runner.step()
        if time.time() - last_debug > DEBUG_PRINT_INTERVAL:
            if verbose:
                print(runner.debug_string())
            last_debug = time.time()

    if verbose:
        print(runner.debug_string(max_debug=99999))

    wait_for_sync()

    errored_trials = []
    for trial in runner.get_trials():
        if trial.status != Trial.TERMINATED:
            errored_trials += [trial]

    if errored_trials:
        if raise_on_failed_trial:
            raise TuneError("Trials did not complete", errored_trials)
        else:
            logger.error("Trials did not complete: %s", errored_trials)

    trials = runner.get_trials()
    if return_trials:
        return trials
    logger.info("Returning an analysis object by default. You can call "
                "`analysis.trials` to retrieve a list of trials. "
                "This message will be removed in future versions of Tune.")
    return ExperimentAnalysis(runner.checkpoint_file, trials=trials)


def run_experiments(experiments,
                    search_alg=None,
                    scheduler=None,
                    with_server=False,
                    server_port=TuneServer.DEFAULT_PORT,
                    verbose=2,
                    resume=False,
                    queue_trials=False,
                    reuse_actors=False,
                    trial_executor=None,
                    raise_on_failed_trial=True):
    """Runs and blocks until all trials finish.

    Examples:
        >>> experiment_spec = Experiment("experiment", my_func)
        >>> run_experiments(experiments=experiment_spec)

        >>> experiment_spec = {"experiment": {"run": my_func}}
        >>> run_experiments(experiments=experiment_spec)

        >>> run_experiments(
        >>>     experiments=experiment_spec,
        >>>     scheduler=MedianStoppingRule(...))

        >>> run_experiments(
        >>>     experiments=experiment_spec,
        >>>     search_alg=SearchAlgorithm(),
        >>>     scheduler=MedianStoppingRule(...))

    Returns:
        List of Trial objects, holding data for each executed trial.

    """
    # This is important to do this here
    # because it schematize the experiments
    # and it conducts the implicit registration.
    experiments = convert_to_experiment_list(experiments)

    trials = []
    for exp in experiments:
        trials += run(
            exp,
            search_alg=search_alg,
            scheduler=scheduler,
            with_server=with_server,
            server_port=server_port,
            verbose=verbose,
            resume=resume,
            queue_trials=queue_trials,
            reuse_actors=reuse_actors,
            trial_executor=trial_executor,
            raise_on_failed_trial=raise_on_failed_trial,
            return_trials=True)
    return trials
