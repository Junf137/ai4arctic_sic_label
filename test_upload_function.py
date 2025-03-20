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
from functions import chart_cbar, water_edge_plot_overlay, compute_metrics, water_edge_metric, class_decider
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
        raise ValueError("String variable must be one of 'train_val', 'test_val', or 'train'")

    train_options = cfg.train_options
    train_options = get_variable_options(train_options)

    weights = torch.load(f=checkpoint, weights_only=False)["model_state_dict"]
    net.load_state_dict(weights)
    print("Model successfully loaded.")

    experiment_name = osp.splitext(osp.basename(cfg.work_dir))[0]
    artifact = wandb.Artifact(experiment_name, "dataset")

    table = wandb.Table(columns=["ID", "Image"])

    # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
    output_class = {chart: torch.Tensor().to(device) for chart in train_options["charts"]}
    outputs_flat = {chart: torch.Tensor().to(device) for chart in train_options["charts"]}
    inf_ys_flat = {chart: torch.Tensor().to(device) for chart in train_options["charts"]}

    # ### Prepare the scene list, dataset and dataloaders
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

    if mode == "val":
        inference_name = "inference_val"
    elif mode == "test":
        inference_name = "inference_test"

    os.makedirs(osp.join(cfg.work_dir, inference_name), exist_ok=True)

    net.eval()
    for inf_x, inf_y, cfv_masks, scene_name, original_size in tqdm(
        iterable=asid_loader, total=len(test_list), colour="green", position=0
    ):
        scene_name = scene_name[:19]  # Removes the _prep.nc from the name.
        torch.cuda.empty_cache()

        inf_x = inf_x.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            if train_options["model_selection"] == "swin":
                output = slide_inference(inf_x, net, train_options, "test")
            else:
                output = net(inf_x)

        # Up sample the masks
        for chart in train_options["charts"]:
            masks_int = cfv_masks[chart].to(torch.uint8)
            masks_int = (
                torch.nn.functional.interpolate(masks_int.unsqueeze(0).unsqueeze(0), size=original_size, mode="nearest")
                .squeeze()
                .squeeze()
            )
            cfv_masks[chart] = torch.gt(masks_int, 0)

        # Upsample data
        if train_options["down_sample_scale"] != 1:
            for chart in train_options["charts"]:
                # check if the output is regression output, if yes, permute the dimension
                if output[chart].size(3) == 1:
                    output[chart] = output[chart].permute(0, 3, 1, 2)
                    output[chart] = torch.nn.functional.interpolate(output[chart], size=original_size, mode="nearest")
                    output[chart] = output[chart].permute(0, 2, 3, 1)
                else:
                    output[chart] = torch.nn.functional.interpolate(output[chart], size=original_size, mode="nearest")

                inf_y[chart] = torch.nn.functional.interpolate(
                    inf_y[chart].unsqueeze(dim=0).unsqueeze(dim=0), size=original_size, mode="nearest"
                ).squeeze()

        for chart in train_options["charts"]:
            output_class[chart] = class_decider(output[chart], train_options, chart).detach()
            outputs_flat[chart] = torch.cat((outputs_flat[chart], output_class[chart][~cfv_masks[chart]]))
            inf_ys_flat[chart] = torch.cat((inf_ys_flat[chart], inf_y[chart][~cfv_masks[chart]].to(device, non_blocking=True)))

        for chart in train_options["charts"]:
            inf_y[chart] = inf_y[chart].cpu().numpy()
            output_class[chart] = output_class[chart].squeeze().cpu().numpy()

        # - Show the scene inference.
        fig, axs2d = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

        axs = axs2d.flat

        for j in range(0, 2):
            ax = axs[j]
            img = torch.squeeze(inf_x, dim=0).cpu().numpy()[j]
            if j == 0:
                ax.set_title(f"Scene {scene_name}, HH")
            else:
                ax.set_title(f"Scene {scene_name}, HV")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img, cmap="gray")

        for idx, chart in enumerate(train_options["charts"]):

            ax = axs[idx + 3]
            output_class[chart] = output_class[chart].astype(float)
            output_class[chart][cfv_masks[chart]] = np.nan
            ax.imshow(
                output_class[chart], vmin=0, vmax=train_options["n_classes"][chart] - 2, cmap="jet", interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title([f"Scene {scene_name}, {chart}: Model Prediction"])
            chart_cbar(ax=ax, n_classes=train_options["n_classes"][chart], chart=chart, cmap="jet")

        for idx, chart in enumerate(train_options["charts"]):

            ax = axs[idx + 6]
            inf_y[chart] = inf_y[chart].astype(float)
            inf_y[chart][cfv_masks[chart]] = np.nan
            ax.imshow(inf_y[chart], vmin=0, vmax=train_options["n_classes"][chart] - 2, cmap="jet", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title([f"Scene {scene_name}, {chart}: Ground Truth"])
            chart_cbar(ax=ax, n_classes=train_options["n_classes"][chart], chart=chart, cmap="jet")

        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.75, wspace=0.5, hspace=-0)
        fig.savefig(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png", format="png", dpi=128, bbox_inches="tight")
        plt.close("all")
        table.add_data(scene_name, wandb.Image(f"{osp.join(cfg.work_dir,inference_name,scene_name)}.png"))

    # compute combine score
    combined_score, scores = compute_metrics(
        true=inf_ys_flat,
        pred=outputs_flat,
        charts=train_options["charts"],
        metrics=train_options["chart_metric"],
        num_classes=train_options["n_classes"],
    )

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
            cm = cm.cpu().numpy()
            cm_percent = np.round(cm / cm.sum(axis=1)[:, np.newaxis] * 100, 2)
            # Plot the confusion matrix
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_percent, annot=True, cmap="Blues")
            # Customize the plot
            class_names = list(GROUP_NAMES[chart].values())
            class_names.append("255")
            tick_marks = np.arange(len(class_names)) + 0.5
            plt.xticks(tick_marks, class_names, rotation=45)
            if chart in ["FLOE", "SOD"]:
                plt.yticks(tick_marks, class_names, rotation=45)
            else:
                plt.yticks(tick_marks, class_names)

            plt.xlabel("Predicted Labels")
            plt.ylabel("Actual Labels")
            plt.title("Confusion Matrix")
            cbar = ax.collections[0].colorbar
            cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
            mkdir_or_exist(f"{osp.join(cfg.work_dir)}/{test_name}")
            plt.savefig(
                f"{osp.join(cfg.work_dir)}/{test_name}/{chart}_confusion_matrix.png", format="png", dpi=128, bbox_inches="tight"
            )

    # Save the results to the wandb
    wandb.run.summary[f"{test_name}/Best Combined Score"] = combined_score
    print(f"{test_name}/Best Combined Score = {combined_score}")

    for chart in train_options["charts"]:
        wandb.run.summary[f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
        print(f"{test_name}/{chart} {train_options['chart_metric'][chart]['func'].__name__} = {scores[chart]}")

        if train_options["compute_classwise_f1score"]:
            wandb.run.summary[f"{test_name}/{chart}: classwise score:"] = classwise_scores[chart]
            print(f"{test_name}/{chart}: classwise score: = {classwise_scores[chart]}")

    if mode == "test":
        artifact.add(table, experiment_name + "_test")
    elif mode == "val":
        artifact.add(table, experiment_name + "_val")

    wandb.log_artifact(artifact)
