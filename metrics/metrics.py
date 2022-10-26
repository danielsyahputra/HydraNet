import sklearn.metrics as metrics

class MTLMetrics():
    def __init__(self) -> None:
        self.age_targets, self.gender_targets, self.race_targets = [],[],[]
        self.age_outputs, self.gender_outputs, self.race_outputs = [],[],[]
    
    def insert(self, values, task: str, mode: str = "target") -> None:
        if mode == "target":
            if task == "age":
                self.age_targets.extend(values)
            elif task == "gender":
                self.gender_targets.extend(values)
            else:
                self.race_targets.extend(values)
        else:
            if task == "age":
                self.age_outputs.extend(values)
            elif task == "gender":
                self.gender_outputs.extend(values)
            else:
                self.race_outputs.extend(values)

    def evalute_model(self, regression_metric: str, classification_metric: str, step='train'):
        age_metric = self._evaluate_regression(metric=regression_metric)
        gender_metric = self._evaluate_classification(task='gender', metric=classification_metric)
        race_metric = self._evaluate_classification(task='race', metric=classification_metric)
        metrics = {
            f"{step}_age_metric": age_metric,
            f"{step}_gender_metric": gender_metric,
            f"{step}_race_metric": race_metric
        }
        return metrics

    def _evaluate_regression(self, metric: str = "mae"):
        if metric == "mae":
            metric_value = self._mae()
        elif metric == "mse":
            metric_value = self._mse()
        else:
            pass
        return metric_value

    def _evaluate_classification(self, task: str, metric: str = "f1"):
        if metric == "f1":
            metric_value = self._f1(task=task)
        elif metric == 'recall':
            metric_value = self._recall(task=task)
        elif metric == "precision":
            metric_value = self._precision(task=task)
        elif metric == "acc":
            metric_value = self._acc(task=task)
        else:
            pass
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