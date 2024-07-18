import os
from models import CIRPlus
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time
from data_utils import squarepad_transform, CIRDataset, targetpad_transform
from utils import extract_index_features, collate_fn, device


def compute_fiq_val_metrics(relative_val_dataset: CIRDataset,
                            blip_model: torch.nn.Module,
                            index_features: torch.tensor, index_features_normed_pooled: torch.tensor,
                            index_names: List[str]) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_model: stage 1 model
    :param index_features: validation index features
    :param index_features_normed_pooled: validation index features, pooled to 256 and normalized
    :param index_names: validation index names
    :return: the computed validation metrics
    """

    # Generate predictions with reference image and text
    predicted_features, target_names = generate_fiq_val_predictions(blip_model, relative_val_dataset,
                                                                    index_names, index_features)

    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the target candidates' index features
    index_features = index_features_normed_pooled.float()  # already normed

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(blip_model: torch.nn.Module,
                                 relative_val_dataset: CIRDataset,
                                 index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param blip_model: stage1 model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=8, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(batch_target_names) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = blip_model.img_txt_fusion(reference_image_features.to(device), None,
                                                                 input_captions,
                                                                 train=False)

        predicted_features = torch.vstack((predicted_features, batch_predicted_features))  # already normed
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str, model: torch.nn.Module, preprocess: callable, train=False):
    """
    Perform retrieval on FashionIQ validation set computing the metrics.
    :param dress_type: FashionIQ category on which perform the retrieval
    :param model: stageI model
    :param preprocess: preprocess pipeline
    """

    model = model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRDataset('fiq', 'val', 'classic', preprocess, args.data_path, [dress_type],
                                     fiq_val_type=args.fiq_val_type)
    index_features, index_features_p, index_names = extract_index_features(classic_val_dataset, model)
    relative_val_dataset = CIRDataset('fiq', 'val', 'relative', preprocess, args.data_path, [dress_type])

    return compute_fiq_val_metrics(relative_val_dataset, model, index_features, index_features_p,
                                   index_names)


def compute_cirr_val_metrics(relative_val_dataset: CIRDataset,
                             blip_model: torch.nn.Module,
                             index_features: torch.tensor, index_features_normed_pooled: torch.tensor,
                             index_names: List[str]) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param blip_model: stageI model
    :param index_features: validation index features
    :param index_features_normed_pooled: validation index features after normalization and pooling; if not using this feature then pass in the vanilla index features
    :param index_names: validation index names
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features)

    print(f"[{datetime.now()}] Compute CIRR validation metrics")

    # Normalize the target candidates' index features
    print(f"[{datetime.now()}] Compute the index features")
    index_features = index_features_normed_pooled.float()  # already normed

    # Compute the distances and sort the results
    print(f"[{datetime.now()}] Compute the distances and sort the results")
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    print(f"[{datetime.now()}] Compute the ground-truth labels wrt the predictions")
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    print(f"[{datetime.now()}] Compute subset predictions and ground-truth labels")
    group_members = np.array(group_members)
    print(f"[{datetime.now()}] Compute group_mask")
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    print(f"[{datetime.now()}] Compute group_labels")
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    print(f"[{datetime.now()}] Compute assert torch.equal")
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    print(f"[{datetime.now()}] Compute metrics")
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(blip_model: torch.nn.Module,
                                  relative_val_dataset: CIRDataset,
                                  index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param blip_model: stage1 model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print(f"[{datetime.now()}] Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(batch_target_names) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = blip_model.img_txt_fusion(reference_image_features.to(device), None, captions,
                                                                 train=False)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(model: torch.nn.Module, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics.
    :param model: stage1 model
    :param preprocess: preprocess pipeline
    """

    model = model.float().eval()
    classic_val_dataset = CIRDataset('cirr', 'val', 'classic', preprocess, args.data_path)
    index_features, index_features_p, index_names = extract_index_features(classic_val_dataset, model)
    relative_val_dataset = CIRDataset('cirr', 'val', 'relative', preprocess, args.data_path)

    return compute_cirr_val_metrics(relative_val_dataset, model, index_features, index_features_p,
                                    index_names, )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'cirr' or 'fiq'")
    parser.add_argument("--combiner", type=str, default='sum',
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--blip-model-name", default="/root_path/models/blip/model_base.pth", type=str,
                        help="CLIP model to use, e.g 'RN50', 'RN50x4','ViT-B/16'")
    parser.add_argument("--model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--model_path")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--fiq_val_type", default=0, type=int)
    parser.add_argument("--load_origin", action='store_true')
    args = parser.parse_args()

    model = CIRPlus(args.blip_model_name)
    model.eval()
    input_dim = model.input_dim
    feature_dim = model.output_dim

    if args.model_path:
        model.load_ckpt(args.model_path, args.load_origin)
    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = model.preprocess

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(model.blip, preprocess)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fiq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', model.blip,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', model.blip,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', model.blip,
                                                                       preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    else:
        raise ValueError("Dataset should be either 'cirr' or 'fiq")
