import torch
import torch.nn as nn
from torch import Tensor

# class MTLLoss(nn.Module):
#     def __init__(self, model, num_tasks: int = 3, loss_type: str = "fixed") -> None:
#         super().__init__()
#         self.model = model
#         self.num_tasks = num_tasks
#         self.log_vars = nn.Parameter(torch.zeros((num_tasks)))

#     def forward(self, targets: Tensor, imgs: Tensor):
#         outputs = self.model(imgs)
#         loss = 0
#         for i in range(self.num_tasks):
#             precision = torch.exp(-self.log_vars[i])
#             loss += torch.sum(precision * (targets[i] - outputs[i]) ** 2 + self.log_vars[i], -1)
#         loss = torch.mean(loss)
#         return loss, self.log_vars.data.tolist()

class MultiTaskLoss(nn.Module):
    def __init__(self, loss_type, uncertainties, enabled_tasks=(True, True, True)) -> None:
        super().__init__()
        assert len(uncertainties) == 3
        assert len(enabled_tasks) == 3
        assert ((loss_type == "learned" and isinstance(uncertainties[0], nn.parameter.Parameter)) or (
            loss_type == "fixed" and isinstance(uncertainties[0], float)
        ))
        self.loss_type = loss_type
        self.uncertainties = uncertainties
        self.enabled_tasks = enabled_tasks
        self.l1_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def age_loss(self, age_outputs, age_targets):
        return self.l1_loss(age_outputs, age_targets)

    def gender_loss(self, gender_outputs, gender_targets):
        return self.bce_loss(gender_outputs, gender_targets)

    def race_loss(self, race_outputs, race_targets):
        return self.cross_entropy(race_outputs, race_targets)

    def calculate_total_loss(self, *losses):
        age_loss, gender_loss, race_loss = losses
        age_uncertainty, gender_uncertainty, race_uncertainty = self.uncertainties
        age_enabled, gender_enabled, race_enabled = self.enabled_tasks
        loss = 0
        
        if self.loss_type == "fixed":
            if age_enabled:
                loss += age_uncertainty * age_loss
            if gender_enabled:
                loss += gender_uncertainty * gender_loss
            if race_enabled:
                loss += race_uncertainty * race_loss
        elif self.loss_type == "learned":
            if age_enabled:
                loss += torch.exp(-age_uncertainty) * age_loss + 0.5 * age_uncertainty
            if gender_enabled:
                loss += 0.5 * (torch.exp(-gender_uncertainty) * gender_loss + gender_uncertainty)
            if race_enabled:
                loss += 0.5 * (torch.exp(-race_uncertainty) * race_loss + race_uncertainty)
        else:
            raise ValueError
        return loss

    def forward(self, outputs, targets):
        age_outputs, gender_outputs, race_outputs = outputs
        age_targets, gender_targets, race_targets = targets
        age_enabled, gender_enabled, race_enabled = self.enabled_tasks
        age_loss = self.age_loss(age_outputs, age_targets) if age_enabled else None
        gender_loss = self.gender_loss(gender_outputs, gender_targets) if gender_enabled else None
        race_loss = self.race_loss(race_outputs, race_targets) if race_enabled else None

        total_loss = self.calculate_total_loss(age_loss, gender_loss, race_loss)
        age_loss_value = age_loss.item() if age_enabled else None
        gender_loss_value = gender_loss.item() if gender_enabled else None
        race_loss_value = race_loss.item() if race_enabled else None
        return total_loss, (age_loss_value, gender_loss_value, race_loss_value)