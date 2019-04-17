from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import csv
import json
import logging
import os
import subprocess
import yaml
import distutils.version
import numbers

import numpy as np

import ray.cloudpickle as cloudpickle
from ray.tune.log_sync import get_syncer
from ray.tune.result import NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S, \
    TIMESTEPS_TOTAL, EPISODE_REWARD_MEAN

logger = logging.getLogger(__name__)

tf = None
use_tf150_api = True


class Logger(object):
    """Logging interface for ray.tune.

    By default, the UnifiedLogger implementation is used which logs results in
    multiple formats (TensorBoard, rllab/viskit, plain json, custom loggers)
    at once.

    Arguments:
        config: Configuration passed to all logger creators.
        logdir: Directory for all logger creators to log to.
        upload_uri (str): Optional URI where the logdir is sync'ed to.
    """

    def __init__(self, config, logdir, upload_uri=None):
        self.config = config
        self.logdir = logdir
        self.uri = upload_uri
        self._init()

    def _init(self):
        pass

    def on_result(self, result):
        """Given a result, appends it to the existing log."""

        raise NotImplementedError

    def close(self):
        """Releases all resources used by this logger."""

        pass

    def flush(self):
        """Flushes all disk writes to storage."""

        pass


class NoopLogger(Logger):
    def on_result(self, result):
        pass


class JsonLogger(Logger):
    def _init(self):
        config_out = os.path.join(self.logdir, "params.json")
        with open(config_out, "w") as f:
            json.dump(
                self.config,
                f,
                indent=2,
                sort_keys=True,
                cls=_SafeFallbackEncoder)
        config_pkl = os.path.join(self.logdir, "params.pkl")
        with open(config_pkl, "wb") as f:
            cloudpickle.dump(self.config, f)
        local_file = os.path.join(self.logdir, "result.json")
        self.local_out = open(local_file, "a")

    def on_result(self, result):
        json.dump(result, self, cls=_SafeFallbackEncoder)
        self.write("\n")
        self.local_out.flush()

    def write(self, b):
        self.local_out.write(b)

    def flush(self):
        self.local_out.flush()

    def close(self):
        self.local_out.close()


def make_histogram(values, bins=1000):
    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return hist


def to_tf_values(result, path):
    values = []
    for attr, value in result.items():
        if value is not None:
            if use_tf150_api:
                type_list = [int, float, np.float32, np.float64, np.int32]
            else:
                type_list = [int, float]
            if type(value) in type_list:
                learner_path = ["ray", "tune", "info", "learner"]
                if path[:len(learner_path)] == learner_path:
                    tag = "/".join(path[len(learner_path):] + [attr])
                else:
                    tag = "/".join(path + [attr])

                values.append(
                    tf.Summary.Value(
                        tag=tag, simple_value=value))
            elif isinstance(value, np.ndarray):
                values.append(
                    tf.Summary.Value(
                        tag=f"{path[-1]}/{attr}", histo=make_histogram(value)))
            elif type(value) is dict:
                values.extend(to_tf_values(value, path + [attr]))
    return values


def sendmessage(message):
    subprocess.Popen(['notify-send', message])
    subprocess.Popen(['paplay', '--volume', '50000', '/usr/share/sounds/freedesktop/stereo/complete.oga'])
    return


class TFLogger(Logger):
    def _init(self):
        try:
            global tf, use_tf150_api
            import tensorflow
            tf = tensorflow
            use_tf150_api = (distutils.version.LooseVersion(tf.VERSION) >=
                             distutils.version.LooseVersion("1.5.0"))
        except ImportError:
            logger.warning("Couldn't import TensorFlow - "
                           "disabling TensorBoard logging.")
        self._file_writer = tf.summary.FileWriter(self.logdir)

        self.time_10_mins_notification_sent = False
        self.time_20_mins_notification_sent = False
        self.reward_notification_sent = False

    def on_result(self, result):
        if result[TRAINING_ITERATION] > 10:
            if result[TIME_TOTAL_S] > 60 * 10:
                if not self.time_10_mins_notification_sent:
                    sendmessage('10 minutes into training')
                    self.time_10_mins_notification_sent = True

            if result[TIME_TOTAL_S] > 60 * 20:
                if not self.time_20_mins_notification_sent:
                    sendmessage('20 minutes into training')
                    self.time_20_mins_notification_sent = True

            if result[EPISODE_REWARD_MEAN] > 0.5:
                if not self.reward_notification_sent:
                    sendmessage('reached episode mean reward of 0.5')
                    self.reward_notification_sent = True

        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to tf log these
        values = to_tf_values(tmp, ["ray", "tune"])
        train_stats = tf.Summary(value=values)
        t = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._file_writer.add_summary(train_stats, t)
        iteration_value = to_tf_values({
            "training_iteration": result[TRAINING_ITERATION]
        }, ["ray", "tune"])
        iteration_stats = tf.Summary(value=iteration_value)
        self._file_writer.add_summary(iteration_stats, t)
        self._file_writer.flush()

    def flush(self):
        self._file_writer.flush()

    def close(self):
        self._file_writer.close()


class CSVLogger(Logger):
    def _init(self):
        """CSV outputted with Headers as first set of results."""
        # Note that we assume params.json was already created by JsonLogger
        progress_file = os.path.join(self.logdir, "progress.csv")
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")
        self._csv_out = None

    def on_result(self, result):
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            if not self._continuing:
                self._csv_out.writeheader()
        self._csv_out.writerow(
            {k: v
             for k, v in result.items() if k in self._csv_out.fieldnames})

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()


DEFAULT_LOGGERS = (JsonLogger, CSVLogger, TFLogger)


class UnifiedLogger(Logger):
    """Unified result logger for TensorBoard, rllab/viskit, plain json.

    This class also periodically syncs output to the given upload uri.

    Arguments:
        config: Configuration passed to all logger creators.
        logdir: Directory for all logger creators to log to.
        upload_uri (str): Optional URI where the logdir is sync'ed to.
        loggers (list): List of logger creators. Defaults to CSV, Tensorboard,
            and JSON loggers.
        sync_function (func|str): Optional function for syncer to run.
            See ray/python/ray/tune/log_sync.py
    """

    def __init__(self,
                 config,
                 logdir,
                 upload_uri=None,
                 loggers=None,
                 sync_function=None):
        if loggers is None:
            self._logger_cls_list = DEFAULT_LOGGERS
        else:
            self._logger_cls_list = loggers
        self._sync_function = sync_function
        self._log_syncer = None

        Logger.__init__(self, config, logdir, upload_uri)

    def _init(self):
        self._loggers = []
        for cls in self._logger_cls_list:
            try:
                self._loggers.append(cls(self.config, self.logdir, self.uri))
            except Exception:
                logger.warning("Could not instantiate {} - skipping.".format(
                    str(cls)))
        self._log_syncer = get_syncer(
            self.logdir, self.uri, sync_function=self._sync_function)

    def on_result(self, result):
        for _logger in self._loggers:
            _logger.on_result(result)
        self._log_syncer.set_worker_ip(result.get(NODE_IP))
        self._log_syncer.sync_if_needed()

    def close(self):
        for _logger in self._loggers:
            _logger.close()
        self._log_syncer.sync_now(force=False)
        self._log_syncer.close()

    def flush(self):
        for _logger in self._loggers:
            _logger.flush()
        self._log_syncer.sync_now(force=False)

    def sync_results_to_new_location(self, worker_ip):
        """Sends the current log directory to the remote node.

        Syncing will not occur if the cluster is not started
        with the Ray autoscaler.
        """
        if worker_ip != self._log_syncer.worker_ip:
            self._log_syncer.set_worker_ip(worker_ip)
            self._log_syncer.sync_to_worker_if_possible()


class _SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(_SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(_SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result):
    result = copy.deepcopy(result)

    try:
        result['info']['learner']['histograms'] = '<not displayed>'
    except KeyError:
        pass

    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=_SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
