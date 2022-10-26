import os
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from timeit import default_timer as timer 
from metrics.metrics import MTLMetrics
from utils.normalize import Transform
from losses.uncertainty_loss import MultiTaskLoss
from typing import Dict, Tuple
from tqdm.auto import tqdm

# MLFlow
import mlflow
import mlflow.pytorch as mp

def train_one_epoch(model, 
                    loader, 
                    optimizer,
                    loss_fns: dict,
                    loss_type: str = "learned",
                    regression_metric:str = "mae",
                    classification_metric: str = "f1",) -> Dict:
    model.train()
    device = next(model.parameters()).device
    batch_losses = []
    batch_age_losses = []
    batch_gender_losses = []
    batch_race_losses = []
    softmax = nn.Softmax(dim=1)
    mtl_metric = MTLMetrics()
    transform = Transform()
    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = transform.get_normalize(targets['age']).to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Batch Result
        optimizer.zero_grad()
        outputs = model(imgs)
        loss_age = loss_fns['age'](outputs[0], age_targets.unsqueeze(1).float())
        loss_gender = loss_fns['gender'](outputs[1], gender_targets)
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
        mtl_metric.insert(values=outputs[0].squeeze(1).detach().cpu().numpy(), task="age", mode='output')
        mtl_metric.insert(values=torch.argmax(softmax(outputs[1]), dim=1).cpu().numpy(), task="gender", mode='output')
        mtl_metric.insert(values=torch.argmax(softmax(outputs[2]), dim=1).cpu().numpy(), task="race", mode='output')
    
    total_loss_value = np.mean(batch_losses)
    age_loss_value = np.mean(batch_age_losses)
    gender_loss_value = np.mean(batch_gender_losses)
    race_loss_value = np.mean(batch_race_losses)

    task_metrics = mtl_metric.evalute_model(regression_metric=regression_metric, classification_metric=classification_metric, step="train")
    metrics = {
        "train_loss_age": age_loss_value,
        "train_loss_gender": gender_loss_value,
        "train_loss_race": race_loss_value,
        "train_total_loss": total_loss_value
    }
    metrics.update(task_metrics)
    return metrics

@torch.no_grad()
def eval_one_epoch(model, 
                    loader, 
                    loss_fns: dict, 
                    regression_metric:str = "mae",
                    classification_metric: str = "f1",
                    step: str = "val") -> Dict:
    model.eval()
    device = next(model.parameters()).device
    batch_losses = []
    mtl_metric = MTLMetrics()
    transform = Transform()

    batch_losses = []
    batch_age_losses = []
    batch_gender_losses = []
    batch_race_losses = []
    softmax = nn.Softmax(dim=1)
    
    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = transform.get_normalize(targets['age']).to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Outputs
        outputs = model(imgs)
        loss_age = loss_fns['age'](outputs[0], age_targets.unsqueeze(1).float())
        loss_gender = loss_fns['gender'](outputs[1], gender_targets)
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
        mtl_metric.insert(values=outputs[0].squeeze(1).detach().cpu().numpy(), task="age", mode='output')
        mtl_metric.insert(values=torch.argmax(softmax(outputs[1]), dim=1).cpu().numpy(), task="gender", mode='output')
        mtl_metric.insert(values=torch.argmax(softmax(outputs[2]), dim=1).cpu().numpy(), task="race", mode='output')
    
    # Total loss for all dataset
    total_loss_value = np.mean(batch_losses)
    age_loss_value = np.mean(batch_age_losses)
    gender_loss_value = np.mean(batch_gender_losses)
    race_loss_value = np.mean(batch_race_losses)

    task_metrics = mtl_metric.evalute_model(regression_metric=regression_metric, classification_metric=classification_metric, step=step)
    metrics = {
        f"{step}_loss_age": age_loss_value,
        f"{step}_loss_gender": gender_loss_value,
        f"{step}_loss_race": race_loss_value,
        f"{step}_total_loss": total_loss_value
    }
    metrics.update(task_metrics)
    return metrics

def get_str_log(epoch: int, metrics: Dict, regression_metric: str, 
                classification_metric:str, step:str = "train") -> str:
    total_loss = metrics[f'{step}_total_loss']
    loss_age = metrics[f'{step}_loss_age']
    loss_gender = metrics[f'{step}_loss_gender']
    loss_race = metrics[f'{step}_loss_race']
    age_metric = metrics[f"{step}_age_metric"]
    gender_metric = metrics[f"{step}_gender_metric"]
    race_metrics = metrics[f"{step}_race_metric"]
    stepUp = step.title()
    log = f"""Epoch: {epoch:3} | {stepUp} Total Loss: {total_loss:.3f} | {stepUp} Loss Age: {loss_age:.3f} | {stepUp} Loss Gender: {loss_gender:.3f} | {stepUp} Loss Race: {loss_race:.3f} | {stepUp} Age {regression_metric.upper()}: {age_metric:.3f} | {stepUp} Gender {classification_metric.title()}: {gender_metric:.3f} | {stepUp} Race {classification_metric.title()}: {race_metrics:.3f}"""
    return log

def train_model(model,
                loaders,
                optimizer,
                loss_fns,
                device,
                model_dir: str,
                model_name: str = "Model",
                experiment_name: str = 'Experiment 1',
                regression_metric: str = "mae",
                classification_metric: str = "f1",
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
        for epoch in tqdm(range(1, epochs + 1)):
            train_metrics = train_one_epoch(model=model, loader=loaders['train'], 
                                    optimizer=optimizer, loss_fns=loss_fns,
                                    regression_metric=regression_metric, classification_metric=classification_metric)
            val_metrics = eval_one_epoch(model=model, loader=loaders['val'], 
                                        loss_fns=loss_fns, step='val',
                                        regression_metric=regression_metric, classification_metric=classification_metric)
            mlflow.log_metrics(metrics=train_metrics, step=epoch)
            mlflow.log_metrics(metrics=val_metrics, step=epoch)
            
            if verbose:
                print(get_str_log(epoch=epoch, metrics=train_metrics, regression_metric=regression_metric, classification_metric=classification_metric, step='train'))
                print(get_str_log(epoch=epoch, metrics=val_metrics, regression_metric=regression_metric, classification_metric=classification_metric, step="val"))
                print("-"*100)
            
            if epoch == 1:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=False)
            
            torch.save(model.state_dict(), f"{model_dir}/weight_epoch_{epoch}.pt")
        end_time = timer()
        mp.log_model(model, model_name)
        mlflow.log_metrics({"time": end_time - start_time})
        del model
        mlflow.end_run()