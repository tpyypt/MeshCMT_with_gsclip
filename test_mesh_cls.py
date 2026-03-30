import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from logger import get_logger
from mesh_cls_common import (
    build_mesh_components,
    compute_global_logits,
    encode_geometry_text,
    move_mesh_inputs_to_device,
    VisualClassifierHead,
)
from mesh_encoder import aggregate_view_features
from metrics_cls import classification_metrics, per_class_accuracy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_classnames(meta_path):
    if not meta_path or not os.path.isfile(meta_path):
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("seen_classnames"), meta.get("unseen_classnames")


def test(args):
    logger = get_logger(args.save_path)
    logger.info("test_primary_branch: visual_only, use_depth_branch=%s", args.use_depth_branch)
    dataset, model, prompt_learner, device = build_mesh_components(args, split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.stage2_checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    model.fusion.load_state_dict(checkpoint["fusion"])
    model.visual_depth.load_state_dict(checkpoint["visual_depth"])
    visual_classifier = VisualClassifierHead(model.text_projection.shape[1], len(dataset.classnames)).to(device)
    visual_classifier.load_state_dict(checkpoint["visual_classifier"])
    prompt_learner.eval()
    model.eval()
    visual_classifier.eval()

    geometry_logits_all = []
    direct_logits_all = []
    visual_logits_all = []
    combined_logits_all = []
    labels_all = []

    for batch in tqdm(dataloader, desc=f"test {args.split}"):
        class_id = batch["class_id"].to(device)
        mesh_inputs = move_mesh_inputs_to_device(batch["mesh_inputs"], device)
        render_images = batch["render_images"].to(device)
        depth_images = batch["depth_images"].to(device)
        batch_size, num_views, channels, height, width = render_images.shape
        render_images = render_images.view(batch_size * num_views, channels, height, width)
        depth_images = depth_images.view(batch_size * num_views, channels, height, width)

        with torch.no_grad():
            text_features, _, _, geometry_logits = encode_geometry_text(model, prompt_learner, mesh_inputs)
            render_global, _ = model.encode_image(render_images, args.features_list, DPAM_layer=args.dpam_layer)
            render_global = F.normalize(render_global, dim=-1)
            if args.use_depth_branch:
                depth_global, _ = model.encode_depth(depth_images, args.features_list)
                depth_global = F.normalize(depth_global, dim=-1)
                fused_view_global = model.fusion_r_d(render_global, depth_global)
            else:
                fused_view_global = render_global
            fused_view_global = F.normalize(fused_view_global, dim=-1)
            fused_global = aggregate_view_features(fused_view_global, batch_size, num_views)
            direct_logits = visual_classifier(fused_global)
            visual_logits = compute_global_logits(fused_global, text_features)
            combined_logits = direct_logits + args.text_logit_weight * visual_logits

        geometry_logits_all.append(geometry_logits.cpu())
        direct_logits_all.append(direct_logits.cpu())
        visual_logits_all.append(visual_logits.cpu())
        combined_logits_all.append(combined_logits.cpu())
        labels_all.append(class_id.cpu())

    geometry_logits_all = torch.cat(geometry_logits_all).numpy()
    direct_logits_all = torch.cat(direct_logits_all).numpy()
    visual_logits_all = torch.cat(visual_logits_all).numpy()
    combined_logits_all = torch.cat(combined_logits_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    summaries = {
        "geometry_only": classification_metrics(labels_all, geometry_logits_all, topk=(1, min(3, len(dataset.classnames)))),
        "direct_only": classification_metrics(labels_all, direct_logits_all, topk=(1, min(3, len(dataset.classnames)))),
        "visual_only": classification_metrics(labels_all, visual_logits_all, topk=(1, min(3, len(dataset.classnames)))),
        "combined": classification_metrics(labels_all, combined_logits_all, topk=(1, min(3, len(dataset.classnames)))),
    }
    primary_branch = "visual_only"
    primary_logits_all = visual_logits_all
    per_class = per_class_accuracy(labels_all, primary_logits_all, dataset.classnames)
    seen_classes, unseen_classes = load_split_classnames(args.meta_path)

    logger.info("primary_result_branch: %s", primary_branch)
    primary_table = [[metric, f"{value * 100:.2f}"] for metric, value in summaries[primary_branch].items()]
    logger.info("\nprimary_result")
    logger.info("\n%s", tabulate(primary_table, headers=["metric", "value"], tablefmt="pipe"))

    for branch_name, summary in summaries.items():
        table = [[metric, f"{value * 100:.2f}"] for metric, value in summary.items()]
        logger.info("\n%s", branch_name)
        logger.info("\n%s", tabulate(table, headers=["metric", "value"], tablefmt="pipe"))

    class_rows = [[class_name, f"{acc * 100:.2f}"] for class_name, acc in per_class.items()]
    logger.info("\nvisual_only_per_class")
    logger.info("\n%s", tabulate(class_rows, headers=["class", "accuracy"], tablefmt="pipe"))

    if seen_classes or unseen_classes:
        split_rows = []
        preds = primary_logits_all.argmax(axis=1)
        if seen_classes:
            seen_ids = [dataset.class_to_idx[name] for name in seen_classes if name in dataset.class_to_idx]
            seen_mask = np.isin(labels_all, seen_ids)
            if seen_mask.any():
                split_rows.append(["seen_accuracy", f"{(preds[seen_mask] == labels_all[seen_mask]).mean() * 100:.2f}"])
        if unseen_classes:
            unseen_ids = [dataset.class_to_idx[name] for name in unseen_classes if name in dataset.class_to_idx]
            unseen_mask = np.isin(labels_all, unseen_ids)
            if unseen_mask.any():
                split_rows.append(["unseen_accuracy", f"{(preds[unseen_mask] == labels_all[unseen_mask]).mean() * 100:.2f}"])
        if split_rows:
            logger.info("\n%s", tabulate(split_rows, headers=["split", "value"], tablefmt="pipe"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mesh Classification Test", add_help=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default="./.cache/manifold40")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--stage2_checkpoint_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--clip_path", type=str, default="pretrained_weights/ViT-L-14-336px.pt")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--point_size", type=int, default=336)
    parser.add_argument("--num_sampled_faces", type=int, default=500)
    parser.add_argument("--num_views", type=int, default=9)
    parser.add_argument("--mesh_mask_ratio", type=float, default=0.0)
    parser.add_argument("--render_backend", type=str, default="pyrender_egl")
    parser.add_argument("--disable_render_cache_generation", action="store_true")
    parser.add_argument("--use_depth_branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--text_logit_weight", type=float, default=0.3)
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
