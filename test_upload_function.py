# # AutoICE - test model and prepare upload package
# This notebook tests the 'best_model', created in the quickstart notebook,
# with the tests scenes exempt of reference data.
# The model outputs are stored per scene and chart in an xarray Dataset in individual Dataarrays.
# The xarray Dataset is saved and compressed in an .nc file ready to be uploaded to the AI4EO.eu platform.
# Finally, the scene chart inference is shown.
#
# The first cell imports necessary packages:

# -- Built-in modules -- #
import json
import os
import os.path as osp

# -- Third-part modules -- #
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from mmengine import mkdir_or_exist
import wandb

# --Proprietary modules -- #
from functions import (
    chart_cbar,
    water_edge_plot_overlay,
    compute_metrics,
    water_edge_metric,
    class_decider,
    cat_edge_cent_metrics,
    calc_edge_cent_metrics,
)
from loaders import AI4ArcticChallengeTestDataset, get_variable_options
from functions import slide_inference, batched_slide_inference, seed_worker


def test(mode: str, net: torch.nn.modules, checkpoint: str, device: str, cfg, test_list, test_name):
    """_summary_

    Args:
        net (torch.nn.modules): The model
        checkpoint (str): The checkpoint to the model
        device (str): The device to run the inference on
        cfg (Config): mmcv based Config object, Can be considered dict
    """
    if mode not in ["val", "test"]:
        raise ValueError("Mode must be 'val' or 'test'")

    train_options = cfg.train_options
    train_options = get_variable_options(train_options)

    weights = torch.load(f=checkpoint, weights_only=False)["model_state_dict"]
    net.load_state_dict(weights)
    print("Model successfully loaded.")

    experiment_name = osp.splitext(osp.basename(cfg.work_dir))[0]
    artifact = wandb.Artifact(experiment_name, "dataset")
    table = wandb.Table(columns=["ID", "Image"])

    # Initialize storage for results
    outputs_flat = {chart: [] for chart in train_options["charts"]}
    inf_ys_flat = {chart: [] for chart in train_options["charts"]}

    # Create data structure to store edge and center predictions and true values
    cent_edge_flat = {
        "cent": {
            "pred": {chart: torch.Tensor() for chart in train_options["charts"]},
            "true": {chart: torch.Tensor() for chart in train_options["charts"]},
        },
        "edge": {
            "pred": {chart: torch.Tensor() for chart in train_options["charts"]},
            "true": {chart: torch.Tensor() for chart in train_options["charts"]},
        },
    }

    # Prepare dataset and dataloader
    dataset = AI4ArcticChallengeTestDataset(options=train_options, files=test_list, mode="train" if mode == "val" else "test")
    asid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=train_options["num_workers_val"],
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(torch.initial_seed()),
    )
    print("Setup ready")

    inference_name = "inference_val" if mode == "val" else "inference_test"
    os.makedirs(osp.join(cfg.work_dir, inference_name), exist_ok=True)

    net.eval()
    for inf_x, inf_y, weight_maps, cfv_masks, scene_name, original_size in tqdm(
        iterable=asid_loader, total=len(test_list), colour="green", position=0, desc=inference_name
    ):
        scene_name = scene_name[:19]  # Remove '_prep.nc' from name
        torch.cuda.empty_cache()

        inf_x = inf_x.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            if train_options["model_selection"] == "swin":
                output = slide_inference(inf_x, net, train_options, "test")
            else:
                output = net(inf_x)
                if train_options["model_selection"] == "UNet_regression_var":
                    # sic_output_var = output["SIC"]["variance"].unsqueeze(-1).to(device)  # Variance of SIC
                    output["SIC"] = output["SIC"]["mean"].unsqueeze(-1).to(device)  # Mean of SIC

            inf_x = inf_x.to("cpu")

        if train_options["down_sample_scale"] != 1:
            for chart in train_options["charts"]:
                # Up sample the masks
                masks_int = cfv_masks[chart].to(torch.uint8)
                masks_int = (
                    torch.nn.functional.interpolate(masks_int.unsqueeze(0).unsqueeze(0), size=original_size, mode="nearest")
                    .squeeze()
                    .squeeze()
                )
                cfv_masks[chart] = torch.gt(masks_int, 0)
                cfv_masks[chart] = cfv_masks[chart].to("cpu")

                # Upsample data
                # check if the output is regression output, if yes, permute the dimension
                if output[chart].size(3) == 1:
                    output[chart] = output[chart].permute(0, 3, 1, 2)
                    output[chart] = torch.nn.functional.interpolate(output[chart], size=original_size, mode="nearest")
                    output[chart] = output[chart].permute(0, 2, 3, 1)
                else:
                    output[chart] = torch.nn.functional.interpolate(output[chart], size=original_size, mode="nearest")
                output[chart] = output[chart].to("cpu")

                inf_y[chart] = torch.nn.functional.interpolate(
                    inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0), size=original_size, mode="nearest"
                ).squeeze()
                inf_y[chart] = inf_y[chart].to("cpu")

                # Upsample the weight maps
                weight_maps[chart] = torch.nn.functional.interpolate(
                    weight_maps[chart].unsqueeze(0).unsqueeze(0), size=original_size, mode="nearest"
                ).squeeze()
                weight_maps[chart] = weight_maps[chart].to("cpu")

        # Process and move results to CPU immediately
        output_class = {}
        for chart in train_options["charts"]:
            output_class[chart] = class_decider(output[chart], train_options, chart).detach()
            outputs_flat[chart].append(output_class[chart][~cfv_masks[chart]])
            inf_ys_flat[chart].append(inf_y[chart][~cfv_masks[chart]])

        # Process edge and center metrics for all charts
        cent_edge_flat = cat_edge_cent_metrics(cent_edge_flat, weight_maps, output_class, inf_y, train_options)

        # Visualization
        fig, axs2d = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

        axs = axs2d.flat

        # HH and HV
        for j in range(0, 2):
            ax = axs[j]
            img = inf_x.squeeze(0).numpy()[j]
            ax.set_title(f"Scene {scene_name}, HH" if j == 0 else f"Scene {scene_name}, HV")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img, cmap="gray")

        # Plot (0, 2)
        ax = axs[2]
        ax.set_xticks([])
        ax.set_yticks([])

        # Output from the model
        for idx, chart in enumerate(train_options["charts"]):
            ax = axs[idx + 3]
            output_np = output_class[chart].numpy().astype(float)
            output_np[cfv_masks[chart].numpy()] = np.nan
            ax.imshow(output_np, vmin=0, vmax=train_options["n_classes"][chart] - 2, cmap="jet", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Scene {scene_name}, {chart}: Model Prediction")
            chart_cbar(ax=ax, n_classes=train_options["n_classes"][chart], chart=chart, cmap="jet")

        # Labels
        for idx, chart in enumerate(train_options["charts"]):
            ax = axs[idx + 6]
            inf_y_np = inf_y[chart].numpy().astype(float)
            inf_y_np[cfv_masks[chart].numpy()] = np.nan
            ax.imshow(inf_y_np, vmin=0, vmax=train_options["n_classes"][chart] - 2, cmap="jet", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Scene {scene_name}, {chart}: Ground Truth")
            chart_cbar(ax=ax, n_classes=train_options["n_classes"][chart], chart=chart, cmap="jet")

        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.75, wspace=0.5, hspace=-0)
        fig.savefig(f"{osp.join(cfg.work_dir, inference_name, scene_name)}.png", format="png", dpi=128, bbox_inches="tight")
        plt.close("all")
        table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir, inference_name, scene_name)}.png"))

        # Clean up GPU memory
        del inf_x, output
        torch.cuda.empty_cache()

    # Concatenate results
    for chart in train_options["charts"]:
        outputs_flat[chart] = torch.cat(outputs_flat[chart])
        inf_ys_flat[chart] = torch.cat(inf_ys_flat[chart])

    # Compute metrics
    combined_score, scores = compute_metrics(
        true=inf_ys_flat,
        pred=outputs_flat,
        charts=train_options["charts"],
        metrics=train_options["chart_metric"],
        num_classes=train_options["n_classes"],
    )

    # Compute edge and center metrics for all charts
    edge_cent_comb_scores, edge_cent_scores = calc_edge_cent_metrics(cent_edge_flat, train_options)

    if train_options["compute_classwise_f1score"]:
        from functions import compute_classwise_f1score

        classwise_scores = compute_classwise_f1score(
            true=inf_ys_flat, pred=outputs_flat, charts=train_options["charts"], num_classes=train_options["n_classes"]
        )

    if train_options["plot_confusion_matrix"]:
        from torchmetrics.functional.classification import multiclass_confusion_matrix
        import seaborn as sns
        from utils import GROUP_NAMES

        for chart in train_options["charts"]:
            cm = multiclass_confusion_matrix(
                preds=outputs_flat[chart], target=inf_ys_flat[chart], num_classes=train_options["n_classes"][chart]
            )
            # Calculate percentages
            cm = cm.numpy()
            row_sums = cm.sum(axis=1)
            # Handle rows with zero sum to avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_percent = np.round(cm / row_sums[:, np.newaxis] * 100, 2)
            # Replace NaN values with zeros
            cm_percent = np.nan_to_num(cm_percent, nan=0.0)

            # Plot the confusion matrix
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_percent, annot=True, cmap="Blues")

            # Customize the plot
            class_names = list(GROUP_NAMES[chart].values()) + ["255"]
            tick_marks = np.arange(len(class_names)) + 0.5
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names, rotation=45 if chart in ["FLOE", "SOD"] else 0)
            plt.xlabel("Predicted Labels")
            plt.ylabel("Actual Labels")
            plt.title(f"{chart} Confusion Matrix (%)")

            cbar = ax.collections[0].colorbar
            cbar.set_ticks(np.linspace(0, 100, 6))
            cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

            mkdir_or_exist(f"{osp.join(cfg.work_dir)}/{test_name}")
            plt.savefig(
                f"{osp.join(cfg.work_dir)}/{test_name}/{chart}_confusion_matrix.png", format="png", dpi=128, bbox_inches="tight"
            )
            plt.close()

    # Save the results to the wandb
    wandb.run.summary[f"{test_name}/Best Combined Score"] = combined_score
    print(f"{test_name}/Best Combined Score = {combined_score}")

    wandb.run.summary[f"{test_name}/Best Combined Score Center"] = edge_cent_comb_scores["cent"]
    print(f"{test_name}/Best Combined Score Center = {edge_cent_comb_scores['cent']}")

    wandb.run.summary[f"{test_name}/Best Combined Score Edge"] = edge_cent_comb_scores["edge"]
    print(f"{test_name}/Best Combined Score Edge = {edge_cent_comb_scores['edge']}")

    for chart in train_options["charts"]:
        metric_name = train_options["chart_metric"][chart]["name"]
        wandb.run.summary[f"{test_name}/{chart} {metric_name}"] = scores[chart]
        print(f"{test_name}/{chart} {metric_name} = {scores[chart]}")
        wandb.run.summary[f"{test_name}/{chart} Center {metric_name}"] = edge_cent_scores["cent"][chart]
        print(f"{test_name}/{chart} Center {metric_name} = {edge_cent_scores['cent'][chart]}")
        wandb.run.summary[f"{test_name}/{chart} Edge {metric_name}"] = edge_cent_scores["edge"][chart]
        print(f"{test_name}/{chart} Edge {metric_name} = {edge_cent_scores['edge'][chart]}")

        if train_options["compute_classwise_f1score"]:
            wandb.run.summary[f"{test_name}/{chart}: classwise score:"] = classwise_scores[chart]
            print(f"{test_name}/{chart}: classwise score: = {classwise_scores[chart]}")

    if mode == "test":
        artifact.add(table, experiment_name + "_test")
    elif mode == "val":
        artifact.add(table, experiment_name + "_val")

    wandb.log_artifact(artifact)
