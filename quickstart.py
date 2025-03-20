__author__ = "Muhammed Patel"
__contributor__ = "Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park"
__copyright__ = ["university of waterloo"]
__contact__ = ["m32patel@uwaterloo.ca", "xinweic@uwaterloo.ca"]
__version__ = "1.0.0"
__date__ = "2024-04-05"

import argparse
import json
import random
import os
import os.path as osp
import shutil
from icecream import ic
import pathlib
import warnings

import numpy as np
import torch
from mmengine import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb

# Functions to calculate metrics and show the relevant chart colorbar.
from functions import (
    compute_metrics,
    save_best_model,
    load_model,
    slide_inference,
    batched_slide_inference,
    water_edge_metric,
    class_decider,
    create_train_validation_and_test_scene_list,
    get_scheduler,
    get_optimizer,
    get_loss,
    get_model,
    seed_worker,
    r2_metric,
    create_edge_cent_flat,
)

# Load costume loss function
from losses import WaterConsistencyLoss, WeightedMSELoss

# Custom dataloaders for regular training and validation.
from loaders import get_variable_options, AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset

#  get_variable_options

# -- Built-in modules -- #
from utils import colour_str
from test_upload_function import test


def parse_args():
    parser = argparse.ArgumentParser(description="Train Default U-NET segmenter")

    # Mandatory arguments
    parser.add_argument("config", type=pathlib.Path, help="train config file path")
    parser.add_argument("--wandb-project", required=True, help="Name of wandb project")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--seed", default=None, help="the seed to use, if not provided, seed from config file will be taken")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume-from",
        type=pathlib.Path,
        default=None,
        help="Resume Training from checkpoint, it will use the optimizer and scheduler defined on checkpoint",
    )
    group.add_argument(
        "--finetune-from", type=pathlib.Path, default=None, help="Start new training using the weights from checkpoint"
    )

    args = parser.parse_args()

    return args


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0):
    """
    Trains the model.
    """
    best_combined_score = -np.Inf  # Best weighted model score.
    patience = train_options["patience"]  # Number of epochs to wait before early stopping.
    patience_counter = 0  # Counter for early stopping.

    loss_ce_functions = {
        chart: get_loss(train_options["chart_loss"][chart]["type"], chart=chart, **train_options["chart_loss"][chart])
        for chart in train_options["charts"]
    }
    weighted_mse_loss = WeightedMSELoss()

    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(start_epoch, train_options["epochs"]), desc="Training"):

        train_loss_sum = torch.tensor([0.0])  # To sum the training batch losses during the epoch.

        val_loss_sum = torch.tensor([0.0])  # To sum the validation batch losses during the epoch.

        net.train()  # Set network to evaluation mode.

        # Loops though batches in queue.
        for i, (batch_x, batch_y, sic_weight_map) in enumerate(
            tqdm(iterable=dataloader_train, total=train_options["epoch_len"], colour="red", desc="Batch")
        ):
            train_loss_batch = torch.tensor([0.0]).to(device)  # Reset from previous batch.

            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.amp.autocast(device_type=device.type):
                # - Forward pass.
                output = net(batch_x)

                # - Calculate loss.
                for chart, weight in zip(train_options["charts"], train_options["task_weights"]):

                    # only calculate loss when there is at least a valid target value
                    if (batch_y[chart] != train_options["class_fill_values"][chart]).any():

                        if chart == "SIC" and train_options["sic_label_mask"]["train"]:
                            train_loss_batch += weight * weighted_mse_loss(
                                output[chart].squeeze(), batch_y[chart].to(device), sic_weight_map.to(device)
                            )
                        else:
                            train_loss_batch += weight * loss_ce_functions[chart](output[chart], batch_y[chart].to(device))

            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            train_loss_batch.backward()

            # - Optimizer step
            optimizer.step()

            # - Scheduler step
            scheduler.step()

            # - Add batch loss.
            train_loss_sum += train_loss_batch.detach().item()

        # - Average loss for displaying
        train_loss_epoch = torch.true_divide(train_loss_sum, i + 1).detach().item()

        # -- Validation Loop -- #
        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options["charts"]}
        inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options["charts"]}

        sic_cent_flat = torch.Tensor().to(device)
        inf_y_sic_cent_flat = torch.Tensor().to(device)
        sic_edge_flat = torch.Tensor().to(device)
        inf_y_sic_edge_flat = torch.Tensor().to(device)

        net.eval()  # Set network to evaluation mode.

        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, sic_weight_map, cfv_masks, name, original_size) in enumerate(
            tqdm(iterable=dataloader_val, total=len(train_options["validate_list"]), colour="green", desc="Validation")
        ):
            torch.cuda.empty_cache()

            val_loss_batch = torch.tensor([0.0]).to(device)

            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                inf_x = inf_x.to(device, non_blocking=True)
                if train_options["model_selection"] == "swin":
                    output = slide_inference(inf_x, net, train_options, "val")
                else:
                    output = net(inf_x)

                for chart, weight in zip(train_options["charts"], train_options["task_weights"]):

                    # only calculate loss when there is at least a valid target value
                    if (inf_y[chart] != train_options["class_fill_values"][chart]).any():

                        if chart == "SIC" and train_options["sic_label_mask"]["val"]:
                            val_loss_batch += weight * weighted_mse_loss(
                                output[chart].squeeze(), inf_y[chart].to(device), sic_weight_map.to(device)
                            )
                        else:
                            val_loss_batch += weight * loss_ce_functions[chart](
                                output[chart], inf_y[chart].unsqueeze(0).long().to(device)
                            )

            # - Final output layer, and storing of non masked pixels.
            for chart in train_options["charts"]:
                output[chart] = class_decider(output[chart], train_options, chart)
                outputs_flat[chart] = torch.cat((outputs_flat[chart], output[chart][~cfv_masks[chart]]))
                inf_ys_flat[chart] = torch.cat(
                    (inf_ys_flat[chart], inf_y[chart][~cfv_masks[chart]].to(device, non_blocking=True))
                )

            # - Store SIC center and edge pixels for r2_metric
            _sic_cent_flat, _inf_y_sic_cent_flat, _sic_edge_flat, _inf_y_sic_edge_flat = create_edge_cent_flat(
                edge_weights=train_options["sic_label_mask"]["edge_weights"],
                sic_weight_map=sic_weight_map,
                output=output,
                inf_y=inf_y,
                chart="SIC",
            )
            sic_cent_flat = torch.cat((sic_cent_flat, _sic_cent_flat.to(device, non_blocking=True)))
            inf_y_sic_cent_flat = torch.cat((inf_y_sic_cent_flat, _inf_y_sic_cent_flat.to(device, non_blocking=True)))
            sic_edge_flat = torch.cat((sic_edge_flat, _sic_edge_flat.to(device, non_blocking=True)))
            inf_y_sic_edge_flat = torch.cat((inf_y_sic_edge_flat, _inf_y_sic_edge_flat.to(device, non_blocking=True)))

            # - Add batch loss.
            val_loss_sum += val_loss_batch.detach().item()

        # - Average loss for displaying
        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.
        combined_score, scores = compute_metrics(
            true=inf_ys_flat,
            pred=outputs_flat,
            charts=train_options["charts"],
            metrics=train_options["chart_metric"],
            num_classes=train_options["n_classes"],
        )

        sic_cent_r2 = (
            torch.round(r2_metric(inf_y_sic_cent_flat, sic_cent_flat) * 100, decimals=3)
            if sic_cent_flat.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        sic_edge_r2 = (
            torch.round(r2_metric(inf_y_sic_edge_flat, sic_edge_flat) * 100, decimals=3)
            if sic_edge_flat.numel() > 0
            else torch.tensor(0.0, device=device)
        )

        if train_options["compute_classwise_f1score"]:
            from functions import compute_classwise_f1score

            # dictionary key = chart, value = tensor; e.g  key = SOD, value = tensor([0, 0.2, 0.4, 0.2, 0.1])
            classwise_scores = compute_classwise_f1score(
                true=inf_ys_flat, pred=outputs_flat, charts=train_options["charts"], num_classes=train_options["n_classes"]
            )
        print("")
        print(f"Epoch {epoch} score:")

        for chart in train_options["charts"]:
            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            # if classwise_f1score is True,
            if train_options["compute_classwise_f1score"]:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"SIC Center R2: {sic_cent_r2}%")
        print(f"SIC Edge R2: {sic_edge_r2}%")

        # Log combine score and epoch loss to wandb
        wandb.log(
            {
                "Combined score": combined_score,
                "Train Epoch Loss": train_loss_epoch,
                "Validation Epoch Loss": val_loss_epoch,
                "SIC Center R2": sic_cent_r2,
                "SIC Edge R2": sic_edge_r2,
                "Learning Rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        # If the scores is better than the previous epoch, then save the model
        if combined_score > best_combined_score:
            best_combined_score = combined_score

            # Log the best combine score, and the metrics that make that best combine score in summary in wandb.
            wandb.run.summary[f"While training/Best Combined Score"] = best_combined_score
            for chart in train_options["charts"]:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[
                    chart
                ]
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch
            wandb.run.summary[f"While training/SIC Center R2"] = sic_cent_r2
            wandb.run.summary[f"While training/SIC Edge R2"] = sic_edge_r2
            wandb.run.summary[f"while training/Best Epoch"] = epoch

            # Save the best model in work_dirs
            model_path = save_best_model(cfg, train_options, net, optimizer, scheduler, epoch)

            wandb.save(model_path)

            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if (patience != 0) and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model_path


def create_dataloaders(train_options):
    """
    Create train and validation dataloader based on the train and validation list inside train_options.
    """

    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(options=train_options, files=train_options["train_list"], do_transform=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        num_workers=train_options["num_workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )

    # Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.
    dataset_val = AI4ArcticChallengeTestDataset(options=train_options, files=train_options["validate_list"], mode="train")
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=None,
        num_workers=train_options["num_workers_val"],
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )

    return dataloader_train, dataloader_val


def main():
    args = parse_args()
    ic(args.config)

    cfg = Config.fromfile(args.config)

    train_options = cfg.train_options
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)

    # generate wandb run wandb_id, to be used to link the run with test_upload
    wandb_id = wandb.util.generate_id()

    # Seed handling: Prioritize CLI seed, then config seed if not -1
    seed = None
    if args.seed is not None:
        seed = int(args.seed)
    elif train_options.get("seed", -1) != -1:
        seed = train_options["seed"]

    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if train_options["deterministic"]:
            # Set deterministic algorithms for cudnn
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

            # Set deterministic algorithms for torch
            torch.use_deterministic_algorithms(True, warn_only=True)

        print(f"Seed: {seed}")
    else:
        print("Random Seed Chosen")

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dir", osp.splitext(osp.basename(args.config))[0])

    # Append run ID for cross-validation runs
    if train_options["cross_val_run"]:
        cfg.work_dir = osp.join(cfg.work_dir, wandb_id)

    ic(osp.abspath(cfg.work_dir))
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))

    # Setup device with GPU/CPU detection and print detailed device info
    if torch.cuda.is_available():
        # Get GPU information
        device = torch.device(f"cuda:{train_options.get('gpu_id', 0)}")

        # Print GPU information
        print(colour_str(f"\n[GPU Information]", "blue"))
        print(colour_str(f"Device: {torch.cuda.get_device_name(device)}", "blue"))
        print(colour_str(f"CUDA Version: {torch.version.cuda}", "blue"))
        print(colour_str(f"Compute Capability: {torch.cuda.get_device_capability(device)}", "blue"))
        print(colour_str(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory/1e9:.2f} GB", "blue"))

        # Compilation warning
        if train_options.get("compile_model", False):
            device_cap = torch.cuda.get_device_capability(device)
            if device_cap not in ((7, 0), (8, 0), (9, 0)):
                warnings.warn(colour_str(f"Compilation may not be optimal on {device_cap} compute capability", "red"))
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        print(colour_str(f"\n[CPU Information]", "blue"))
        print(colour_str(f"Available Cores: {os.cpu_count()}", "blue"))

    print(colour_str(f"Device setup completed: {device}", "blue"))

    net = get_model(train_options, device)
    if train_options["compile_model"]:
        net = torch.compile(net)

    optimizer = get_optimizer(train_options, net)
    scheduler = get_scheduler(train_options, optimizer)

    epoch_start = 0
    if args.resume_from:
        print(colour_str(f"Resuming work from {args.resume_from}", "green"))
        epoch_start = load_model(net, args.resume_from, optimizer, scheduler)
    elif args.finetune_from:
        print(colour_str(f"Finetune model from {args.finetune_from}", "green"))
        _ = load_model(net, args.finetune_from)

    # WandB initialization
    config_basename = osp.splitext(osp.basename(args.config))[0]
    wandb_name = f"{config_basename}-{wandb_id}" if train_options["cross_val_run"] else config_basename
    wandb_group = config_basename if train_options["cross_val_run"] else None
    wandb.init(
        name=wandb_name,
        project=args.wandb_project,
        entity="junf-default",
        config=train_options,
        id=wandb_id,
        group=wandb_group,
        resume="allow",
    )

    # Define the metrics and make them such that they are not added to the summary
    wandb.define_metric("Train Epoch Loss", summary="none")
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Train Water Consistency Epoch Loss", summary="none")
    wandb.define_metric("Validation Epoch Loss", summary="none")
    wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none")
    wandb.define_metric("Validation Water Consistency Epoch Loss", summary="none")
    wandb.define_metric("Combined score", summary="none")
    wandb.define_metric("SIC r2_metric", summary="none")
    wandb.define_metric("SIC Center R2", summary="none")
    wandb.define_metric("SIC Edge R2", summary="none")
    wandb.define_metric("SOD f1_metric", summary="none")
    wandb.define_metric("FLOE f1_metric", summary="none")
    wandb.define_metric("Water Consistency Accuracy", summary="none")
    wandb.define_metric("Learning Rate", summary="none")
    if train_options["compute_classwise_f1score"]:
        for chart in train_options["charts"]:
            for index in range(train_options["n_classes"][chart]):
                wandb.define_metric(f"{chart}/Class: {index}", summary="none")

    wandb.save(str(args.config))
    print(colour_str("Save Config File", "green"))

    # Create the train, validation and test scene list
    create_train_validation_and_test_scene_list(train_options)

    # Create dataloaders
    dataloader_train, dataloader_val = create_dataloaders(train_options)

    # Update Config
    wandb.config["validate_list"] = train_options["validate_list"]

    print("Starting Training")
    checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, epoch_start)

    print("Staring Validation with best model")
    # this is for valset 1 visualization along with gt
    test("val", net, checkpoint_path, device, cfg.deepcopy(), train_options["validate_list"], "Cross Validation")

    print("Starting testing with best model")
    # this is for test path along with gt after the gt has been released
    test("test", net, checkpoint_path, device, cfg.deepcopy(), train_options["test_list"], "Test")

    # finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
