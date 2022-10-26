import os
import shutil
import torch
import torch.nn as nn
from timeit import default_timer as timer 
from utils.normalize import Transform
from typing import Dict
from tqdm.auto import tqdm

# MLFlow
import mlflow
import mlflow.pytorch as mp

def train_one_epoch(model, 
                    loader, 
                    optimizer,
                    loss_fn,
                    mtl_metric) -> Dict:
    model.train()
    device = next(model.parameters()).device
    softmax = nn.Softmax(dim=1)
    transform = Transform()
    mtl_metric.reset()

    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = transform.get_normalize(targets['age']).to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Batch Result
        optimizer.zero_grad()
        outputs = model(imgs)
        target_tasks = (age_targets.unsqueeze(1).float(), gender_targets, race_targets)
        loss, (age_loss_value, gender_loss_value, race_loss_value) = loss_fn(outputs, target_tasks)
        loss.backward()
        optimizer.step()

        targets_cpu = [age_targets.cpu().numpy(), gender_targets.cpu().numpy(), race_targets.cpu().numpy()]
        outputs_cpu = [
            outputs[0].squeeze(1).detach().cpu().numpy(),
            torch.argmax(softmax(outputs[1]), dim=1).cpu().numpy(),
            torch.argmax(softmax(outputs[2]), dim=1).cpu().numpy()
        ]
        losses_values = [loss.item(), age_loss_value, gender_loss_value, race_loss_value]
        mtl_metric.insert(targets=targets_cpu, outputs=outputs_cpu, losses=losses_values)

    metrics = mtl_metric.evalute_model(step="train")    
    return metrics

@torch.no_grad()
def eval_one_epoch(model, 
                    loader, 
                    loss_fn,
                    mtl_metric,
                    step: str = "val") -> Dict:
    model.eval()
    device = next(model.parameters()).device
    transform = Transform()
    softmax = nn.Softmax(dim=1)
    mtl_metric.reset()
    
    for imgs, targets in loader:
        imgs = imgs.to(device)
        age_targets = transform.get_normalize(targets['age']).to(device)
        gender_targets = targets['gender'].to(device)
        race_targets = targets['race'].to(device)

        # Outputs
        outputs = model(imgs)
        target_tasks = (age_targets.unsqueeze(1).float(), gender_targets, race_targets)
        loss, (age_loss_value, gender_loss_value, race_loss_value) = loss_fn(outputs, target_tasks)

        # Evaluate result
        targets_cpu = [age_targets.cpu().numpy(), gender_targets.cpu().numpy(), race_targets.cpu().numpy()]
        outputs_cpu = [
            outputs[0].squeeze(1).detach().cpu().numpy(),
            torch.argmax(softmax(outputs[1]), dim=1).cpu().numpy(),
            torch.argmax(softmax(outputs[2]), dim=1).cpu().numpy()
        ]
        losses_values = [loss.item(), age_loss_value, gender_loss_value, race_loss_value]
        mtl_metric.insert(targets=targets_cpu, outputs=outputs_cpu, losses=losses_values)

    metrics = mtl_metric.evalute_model(step=step)
    return metrics

def get_str_log(epoch: int, train_metrics: Dict, val_metrics: Dict) -> str:
    train_metrics = list(train_metrics.items())
    val_metrics = list(val_metrics.items())
    train_metrics = [(" ".join(metric.split("_")).title(), value) for metric, value in train_metrics]
    val_metrics = [(" ".join(metric.split("_")).title(), value) for metric, value in val_metrics]
    train_log = f"Epoch: {epoch:3} "
    val_log = f"Epoch: {epoch:3} "
    for metric, value in train_metrics:
        train_log += f"| {metric}: {value}"
    for metric, value in val_metrics:
        val_log += f"| {metric}: {value}"
    return f"""
    {train_log}
    {val_log}
    """

def train_model(model,
                loaders,
                optimizer,
                loss_fn,
                mtl_metric,
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
        for epoch in tqdm(range(1, epochs + 1)):
            train_metrics = train_one_epoch(
                model=model,
                loader=loaders['train'],
                optimizer=optimizer,
                loss_fn=loss_fn,
                mtl_metric=mtl_metric
            )
            val_metrics = eval_one_epoch(
                model=model,
                loader=loaders['val'],
                loss_fn=loss_fn,
                mtl_metric=mtl_metric,
                step='val'
            )
            mlflow.log_metrics(metrics=train_metrics, step=epoch)
            mlflow.log_metrics(metrics=val_metrics, step=epoch)
            
            if verbose:
                print(get_str_log(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))
                print("-"*100)
            
            if epoch == 1:
                shutil.rmtree(f"output/{model_dir}", ignore_errors=True)
                os.makedirs(f"output/{model_dir}", exist_ok=False)
            
        end_time = timer()
        mp.log_model(model, model_name)
        mlflow.log_metrics({"time": end_time - start_time})
        del model
        mlflow.end_run()