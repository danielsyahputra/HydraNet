import numpy as np
import sklearn.metrics as metrics
from typing import Iterable, List

class MTLMetrics():
    def __init__(self, enabled_task: List,
                regression_metric: str = "mae", 
                classification_metric: str = "f1") -> None:
        self.enabled_task = enabled_task
        self._set_initial_container()
        self.regression_metric = regression_metric
        self.classification_metric = classification_metric

    def _set_initial_container(self) -> None:
        age_enabled, gender_enabled, race_enabled = self.enabled_task
        if age_enabled:
            self.age_outputs, self.age_targets = [], []
            self.age_loss = []
        if gender_enabled:
            self.gender_outputs, self.gender_targets = [], []
            self.gender_loss = []
        if race_enabled:
            self.race_outputs, self.race_targets = [], []
            self.race_loss = []
        self.total_loss = []

    def reset(self) -> None:
        self._set_initial_container()

    def insert(self, targets: Iterable, outputs: Iterable, losses: Iterable) -> None:
        """
        targets: Tuple[Age targets, Gender targets, Race targets]
        outputs: Tuple[Age outputs, Gender outputs, Race outputs]
        losses: Tuple[total loss, age loss, gender loss, race, loss] @ batch
        """
        age_enabled, gender_enabled, race_enabled = self.enabled_task
        ages_targets, gender_targets, race_targets = targets
        age_outputs, gender_outputs, race_outputs = outputs
        total_loss, age_loss, gender_loss, race_loss = losses
        if age_enabled:
            self.age_targets.extend(ages_targets)
            self.age_outputs.extend(age_outputs)
            self.age_loss.append(age_loss)
        if gender_enabled:
            self.gender_targets.extend(gender_targets)
            self.gender_outputs.extend(gender_outputs)
            self.gender_loss.append(gender_loss)
        if race_enabled:
            self.race_targets.extend(race_targets)
            self.race_outputs.extend(race_outputs)
            self.race_loss.append(race_loss)
        self.total_loss.append(total_loss)

    def evalute_model(self, step='train'):
        age_enabled, gender_enabled, race_enabled = self.enabled_task
        metrics = {f"{step}_total_loss": np.mean(self.total_loss)}
        if age_enabled:
            age_metric = self._evaluate_regression()
            metrics.update({
                f"{step}_age_{self.regression_metric}": age_metric,
                f"{step}_age_loss": np.mean(self.age_loss)
            })
        if gender_enabled:
            gender_metric = self._evaluate_classification(task="gender")
            metrics.update({
                f"{step}_gender_{self.classification_metric}": gender_metric,
                f"{step}_gender_loss": np.mean(self.gender_loss)
            })
        if race_enabled:
            race_metric = self._evaluate_classification(task="race")
            metrics.update({
                f"{step}_race_{self.classification_metric}": race_metric,
                f"{step}_race_loss": np.mean(self.race_loss)
            })
        return metrics

    def _evaluate_regression(self):
        if self.regression_metric == "mae":
            metric_value = self._mae()
        elif self.regression_metric == "mse":
            metric_value = self._mse()
        else:
            raise ValueError
        return metric_value

    def _evaluate_classification(self, task: str):
        metric = self.classification_metric
        if metric == "f1":
            metric_value = self._f1(task=task)
        elif metric == 'recall':
            metric_value = self._recall(task=task)
        elif metric == "precision":
            metric_value = self._precision(task=task)
        elif metric == "acc":
            metric_value = self._acc(task=task)
        else:
            raise ValueError
        return metric_value

    def _mse(self):
        return metrics.mean_squared_error(self.age_targets, self.age_outputs)

    def _mae(self):
        return metrics.mean_absolute_error(self.age_targets, self.age_outputs)

    def _acc(self, task: str):
        if task == "gender":
            return metrics.accuracy_score(self.gender_targets, self.gender_outputs)
        return metrics.accuracy_score(self.race_targets, self.race_outputs)

    def _precision(self, task: str):
        if task == "gender":
            return metrics.precision_score(self.gender_targets, self.gender_outputs, average='binary')
        return metrics.precision_score(self.race_targets, self.race_outputs, average='micro')

    def _recall(self, task: str):
        if task == "gender":
            return metrics.recall_score(self.gender_targets, self.gender_outputs, average='binary')
        return metrics.recall_score(self.race_targets, self.race_outputs, average='micro')

    def _f1(self, task: str):
        if task == "gender":
            return metrics.f1_score(self.gender_targets, self.gender_outputs, average='binary')
        return metrics.f1_score(self.race_targets, self.race_outputs, average='micro')