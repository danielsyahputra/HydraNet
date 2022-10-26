from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

class MultiTaskLoss(nn.Module):
    def __init__(self, loss_type: str, task_num: int = 3, enabled_task=(True, True, True), custom_weights: Tuple = (1, 1, 1)) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.task_num = task_num
        self.enabled_task = enabled_task
        self.custom_weights = custom_weights
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.l1_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def _age_loss(self, age_outputs: Tensor, age_targets: Tensor):
        return self.l1_loss(age_outputs, age_targets)

    def _gender_loss(self, gender_outputs: Tensor, gender_targets: Tensor):
        return self.cross_entropy(gender_outputs, gender_targets)

    def _race_loss(self, race_outputs: Tensor, race_targets: Tensor):
        return self.cross_entropy(race_outputs, race_targets)

    def _calculate_total_loss(self, *losses):
        age_loss, gender_loss, race_loss = losses
        age_enabled, gender_enabled, race_enabled = self.enabled_tasks
        age_weight_loss, gender_weight_loss, race_weight_loss = self.custom_weights
        age_log_vars, gender_log_vars, race_log_vars = self.log_vars
        loss = 0
        if self.loss_type == "fixed":
            if age_enabled:
                loss += age_weight_loss * age_loss
            if gender_enabled:
                loss += gender_weight_loss * gender_loss
            if race_enabled:
                loss += race_loss * race_loss
        elif self.loss_type == "learned":
            if age_enabled:
                precision = torch.exp(age_log_vars)
                loss += precision * age_loss + age_log_vars
            if gender_enabled:
                precision = torch.exp(gender_log_vars)
                loss += precision * gender_loss + gender_log_vars
            if race_enabled:
                precision = torch.exp(race_log_vars)
                loss += precision * race_loss + race_log_vars
        else:
            raise ValueError

    def forward(self, outputs, targets):
        age_outputs, gender_outputs, race_outputs = outputs
        age_targets, gender_targets, race_targets = targets
        age_enabled, gender_enabled, race_enabled = self.enabled_task
        age_loss = self._age_loss(age_outputs, age_targets) if age_enabled else None
        gender_loss = self._gender_loss(gender_outputs, gender_targets) if gender_enabled else None
        race_loss = self._race_loss(race_outputs, race_targets) if race_enabled else None
        
        total_loss = self._calculate_total_loss(age_loss, gender_loss, race_loss)
        age_loss_value = age_loss.item() if age_enabled else None
        gender_loss_value = gender_loss.item() if gender_enabled else None
        race_loss_value = race_loss.item() if race_enabled else None
        return total_loss, (age_loss_value, gender_loss_value, race_loss_value)