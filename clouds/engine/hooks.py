from detectron2.engine.train_loop import HookBase
from detectron2.evaluation.testing import flatten_results_dict
import detectron2.utils.comm as comm
from fvcore.common.checkpoint import Checkpointer
import logging
import operator
import math


class PersoEvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, eval_after_train=True):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still evaluate after the last iteration
                if `eval_after_train` is True).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            eval_after_train (bool): whether to evaluate after the last iteration

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self._eval_after_train = eval_after_train

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def before_train(self):
        """
        Called before the first iteration.
        """
        if "debug" in self.trainer.cfg.OUTPUT_DIR:
            pass
        else:
            results = self._func()

            if results:
                assert isinstance(
                    results, dict
                ), "Eval function must return a dict. Got {} instead.".format(results)

                flattened_results = flatten_results_dict(results)
                for k, v in flattened_results.items():
                    try:
                        v = float(v)
                    except Exception as e:
                        raise ValueError(
                            "[EvalHook] eval_function should return a nested dict of float. "
                            "Got '{}: {}' instead.".format(k, v)
                        ) from e
                self.trainer.storage.put_scalars(
                    **flattened_results, smoothing_hint=False
                )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class PersoBestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metrics: list,
        mode: str = "max",
        file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metrics = val_metrics
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        # metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        # metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        values_tuple = [self.trainer.storage.latest()[key] for key in self._val_metrics]
        # Calculate the mean of the first elements
        mean_first_elements = sum(x[0] for x in values_tuple) / len(values_tuple)
        # Get the second element from the first tuple
        second_element = values_tuple[0][1]
        metric_tuple = (mean_first_elements, second_element)
        if metric_tuple is None:
            # self._logger.warning(
            #     f"Given val metric does not seem to be computed/stored."
            #     "Will not be checkpointing based on it."
            # )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                # self._logger.info(
                #     f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                # )
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)
            # self._logger.info(
            #     f"Saved best model as latest eval score is "
            #     f"{latest_metric:0.5f}, better than last best score "
            #     f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            # )
            self._update_best(latest_metric, metric_iter)
        # else:
        # self._logger.info(
        #     f"Not saving as latest eval score is {latest_metric:0.5f}, "
        #     f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
        # )

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        # same conditions as `EvalHook`
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()
