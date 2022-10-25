import os
import time
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from timeit import default_timer as timer 
from .metrics import MTLMetrics
from typing import Dict, Tuple
from tqdm.auto import tqdm

# MLFlow
import mlflow
import mlflow.pytorch as mp

def train_one_epoch(model, 
                    loader, 
                    optimizer,
                    loss_fns: dict) -> Tuple:
    model.train()
    device = next(model.parameters()).device
    batch_losses = []
    batch_age_losses = []
    batch_gender_losses = []
    batch_race_losses = []

    mtl_metric = MTLMetrics()
    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = targets['age'].to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Batch Result
        optimizer.zero_grad()
        outputs = model(imgs)
        loss_age = loss_fns['age'](outputs[0], age_targets.unsqueeze(1).float())
        loss_gender = loss_fns['gender'](outputs[1], gender_targets.unsqueeze(1).float())
        loss_race = loss_fns['race'](outputs[2], race_targets)
        loss = loss_age + loss_gender + loss_race
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        batch_age_losses.append(loss_age.item())
        batch_gender_losses.append(loss_gender.item())
        batch_race_losses.append(loss_race.item())

        mtl_metric.insert(values=age_targets.cpu().numpy(), task="age", mode='target')
        mtl_metric.insert(values=gender_targets.cpu().numpy(), task="gender", mode='target')
        mtl_metric.insert(values=race_targets.cpu().numpy(), task="race", mode='target')
        mtl_metric.insert(values=outputs[0].squeeze(1).detach().numpy(), task="age", mode='output')
        mtl_metric.insert(values=outputs[1].squeeze(1).detach().numpy(), task="gender", mode='output')
        mtl_metric.insert(values=outputs[2].squeeze(1).detach().numpy(), task="race", mode='output')
    
    total_loss_value = np.mean(batch_losses)
    age_loss_value = np.mean(batch_age_losses)
    gender_loss_value = np.mean(batch_gender_losses)
    race_loss_value = np.mean(batch_race_losses.item())

    task_metrics = mtl_metric.evalute_model(regression_metric="mae", classification_metric="f1", step="train")
    metrics = {
        "train_loss_age": age_loss_value,
        "train_loss_gender": gender_loss_value,
        "train_loss_race": race_loss_value,
        "train_total_loss": total_loss_value
    }
    metrics.update(task_metrics)
    return metrics

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fns: dict, step: str = "val"):
    model.eval()
    device = next(model.parameters()).device
    batch_losses = []
    mtl_metric = MTLMetrics()

    batch_losses = []
    batch_age_losses = []
    batch_gender_losses = []
    batch_race_losses = []

    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = targets['age'].to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Outputs
        outputs = model(imgs)
        loss_age = loss_fns['age'](outputs[0], age_targets.unsqueeze(1).float())
        loss_gender = loss_fns['gender'](outputs[1], gender_targets.unsqueeze(1).float())
        loss_race = loss_fns['race'](outputs[2], race_targets)
        loss = loss_age + loss_gender + loss_race

        # Task Loss
        batch_losses.append(loss.item())
        batch_age_losses.append(loss_age.item())
        batch_gender_losses.append(loss_gender.item())
        batch_race_losses.append(loss_race.item())

        # Task metrics
        mtl_metric.insert(values=age_targets.cpu().numpy(), task="age", mode='target')
        mtl_metric.insert(values=gender_targets.cpu().numpy(), task="gender", mode='target')
        mtl_metric.insert(values=race_targets.cpu().numpy(), task="race", mode='target')
        mtl_metric.insert(values=outputs[0].squeeze(1).detach().numpy(), task="age", mode='output')
        mtl_metric.insert(values=outputs[1].squeeze(1).detach().numpy(), task="gender", mode='output')
        mtl_metric.insert(values=outputs[2].squeeze(1).detach().numpy(), task="race", mode='output')
    
    # Total loss for all dataset
    total_loss_value = np.mean(batch_losses)
    age_loss_value = np.mean(batch_age_losses)
    gender_loss_value = np.mean(batch_gender_losses)
    race_loss_value = np.mean(batch_race_losses.item())

    task_metrics = mtl_metric.evalute_model(regression_metric="mae", classification_metric="f1", step=step)
    metrics = {
        f"{step}_loss_age": age_loss_value,
        f"{step}_loss_gender": gender_loss_value,
        f"{step}_loss_race": race_loss_value,
        f"{step}_total_loss": total_loss_value
    }
    metrics.update(task_metrics)
    return metrics

def get_str_log(epoch: int, metrics: Dict, step:str = "train") -> str:
    total_loss = metrics[f'{step}_total_loss']
    loss_age = metrics[f'{step}_loss_age']
    loss_gender = metrics[f'{step}_loss_gender']
    loss_race = metrics[f'{step}_loss_race']
    age_metric = f"{step}_age_metric"
    gender_metric = f"{step}_gender_metric"
    race_metrics = f"{step}_race_metric"
    stepUp = step.upper()
    log = f"""
    Epoch: {epoch:3} | {stepUp} Total Loss: {total_loss:.3} | {stepUp} Loss Age: {loss_age:.3f} | {stepUp} Loss Gender: {loss_gender:.3f} | {stepUp} Loss Race: {loss_race}
    Epoch: {epoch:3} | {stepUp} Age Metric: {age_metric:.3} | {stepUp} Gender Metric: {gender_metric:.3f} | {stepUp} Race Metric: {race_metrics:.3f}"""
    return log

def train_model(model,
                loaders,
                optimizer,
                loss_fns,
                device,
                model_dir: str,
                model_name: str = "Model",
                experiment_name: str = 'Experiment 1',
                epochs: int = 10,
                verbose: bool = True):
    model.to(device)

    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id = current_experiment['experiment_id']
    
    with mlflow.start_run(experiment_id=experiment_id):
        start_time = timer()
        for epoch in tqdm(epochs + 1):
            train_metrics = train_one_epoch(model=model, loader=loaders['train'], 
                                    optimizer=optimizer, loss_fns=loss_fns)
            val_metrics = eval_one_epoch(model=model, loader=loaders['val'], loss_fns=loss_fns, step='val')
            mlflow.log_metrics(metrics=train_metrics)
            mlflow.log_metrics(metrics=val_metrics)
            
            if verbose:
                print(get_str_log(epoch=epoch, metrics=train_metrics, step='train'))
                print(get_str_log(epoch=epoch, metrics=val_metrics, step="val"))
            
            if epoch == 1:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=False)
            
            torch.save(model.state_dict(), f"{model_dir}/weight_epoch_{epoch}.pt")
        end_time = timer()
        eval_one_epoch()
        mp.log_model(model, model_name)
        mlflow.log_metrics({"time": end_time - start_time})
        del model
        mlflow.end_run()