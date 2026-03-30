import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from logger import get_logger
from mesh_cls_common import (
    build_mesh_components,
    collect_trainable_params,
    compute_global_logits,
    encode_geometry_text,
    freeze_prompt_semantics,
    move_mesh_inputs_to_device,
    VisualClassifierHead,
)
from mesh_encoder import aggregate_view_features


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    logger = get_logger(args.save_path)
    logger.info(
        "stage2_primary_train_branch: visual_only (visual_text_loss_weight=%.3f, direct_loss_weight=%.3f, combined_loss_weight=%.3f, use_depth_branch=%s)",
        args.text_loss_weight,
        args.direct_loss_weight,
        args.combined_loss_weight,
        args.use_depth_branch,
    )
    dataset, model, prompt_learner, device = build_mesh_components(args, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    visual_classifier = VisualClassifierHead(model.text_projection.shape[1], len(dataset.classnames)).to(device)

    checkpoint = torch.load(args.stage1_checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    freeze_prompt_semantics(prompt_learner)
    for param in prompt_learner.mesh_encoder.parameters():
        param.requires_grad = False

    params_to_optimize = [
        {"params": collect_trainable_params(model.fusion), "lr": args.learning_rate},
        {"params": visual_classifier.parameters(), "lr": args.learning_rate},
    ]
    if args.use_depth_branch:
        params_to_optimize.insert(1, {"params": collect_trainable_params(model.visual_depth), "lr": args.learning_rate * 0.1})
    params_to_optimize = [group for group in params_to_optimize if group["params"]]
    optimizer = torch.optim.Adam(params_to_optimize, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        direct_losses = []
        visual_losses = []
        combined_losses = []
        geometry_distill_losses = []
        consistency_losses = []
        total_losses = []
        direct_acc_values = []
        visual_acc_values = []
        combined_acc_values = []
        geometry_acc_values = []
        prompt_learner.eval()
        model.eval()
        model.fusion.train()
        if args.use_depth_branch:
            model.visual_depth.train()
        visual_classifier.train()

        for batch in tqdm(dataloader, desc=f"stage2 epoch {epoch + 1}/{args.epochs}"):
            class_id = batch["class_id"].to(device)
            mesh_inputs = move_mesh_inputs_to_device(batch["mesh_inputs"], device)
            render_images = batch["render_images"].to(device)
            depth_images = batch["depth_images"].to(device)
            batch_size, num_views, channels, height, width = render_images.shape
            render_images = render_images.view(batch_size * num_views, channels, height, width)
            depth_images = depth_images.view(batch_size * num_views, channels, height, width)

            with torch.no_grad():
                text_features, _, _, geometry_logits = encode_geometry_text(model, prompt_learner, mesh_inputs)

            with torch.no_grad():
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

            direct_loss = F.cross_entropy(direct_logits, class_id)
            visual_loss = F.cross_entropy(visual_logits, class_id)
            combined_loss = F.cross_entropy(combined_logits, class_id)
            geometry_distill_loss = F.kl_div(
                F.log_softmax(visual_logits / args.distill_temperature, dim=-1),
                F.softmax(geometry_logits.detach() / args.distill_temperature, dim=-1),
                reduction="batchmean",
            ) * (args.distill_temperature ** 2)

            view_features = fused_view_global.view(batch_size, num_views, -1)
            mean_feature = view_features.mean(dim=1, keepdim=True)
            consistency_loss = (1.0 - F.cosine_similarity(view_features, mean_feature, dim=-1).mean()) * args.consistency_weight

            loss = (
                args.direct_loss_weight * direct_loss
                + args.text_loss_weight * visual_loss
                + args.combined_loss_weight * combined_loss
                + args.geometry_distill_weight * geometry_distill_loss
                + consistency_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            direct_preds = direct_logits.argmax(dim=1)
            visual_preds = visual_logits.argmax(dim=1)
            combined_preds = combined_logits.argmax(dim=1)
            geometry_preds = geometry_logits.argmax(dim=1)
            direct_losses.append(direct_loss.item())
            visual_losses.append(visual_loss.item())
            combined_losses.append(combined_loss.item())
            geometry_distill_losses.append(geometry_distill_loss.item())
            consistency_losses.append(consistency_loss.item())
            total_losses.append(loss.item())
            direct_acc_values.append((direct_preds == class_id).float().mean().item())
            visual_acc_values.append((visual_preds == class_id).float().mean().item())
            combined_acc_values.append((combined_preds == class_id).float().mean().item())
            geometry_acc_values.append((geometry_preds == class_id).float().mean().item())

        logger.info(
            "epoch [%d/%d], total_loss: %.4f, direct_loss: %.4f, visual_text_loss: %.4f, combined_loss: %.4f, geom_distill_loss: %.4f, cons_loss: %.4f, direct_acc: %.4f, visual_text_acc: %.4f, combined_acc: %.4f, geometry_teacher_acc: %.4f",
            epoch + 1,
            args.epochs,
            float(np.mean(total_losses)),
            float(np.mean(direct_losses)),
            float(np.mean(visual_losses)),
            float(np.mean(combined_losses)),
            float(np.mean(geometry_distill_losses)),
            float(np.mean(consistency_losses)),
            float(np.mean(direct_acc_values)),
            float(np.mean(visual_acc_values)),
            float(np.mean(combined_acc_values)),
            float(np.mean(geometry_acc_values)),
        )
        if (epoch + 1) % args.save_freq == 0:
            torch.save(
                {
                    "fusion": model.fusion.state_dict(),
                    "visual_depth": model.visual_depth.state_dict(),
                    "visual_classifier": visual_classifier.state_dict(),
                    "prompt_learner": prompt_learner.state_dict(),
                    "classnames": dataset.classnames,
                    "args": vars(args),
                },
                os.path.join(args.save_path, f"epoch_{epoch + 1}.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mesh Classification Stage2", add_help=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default="./.cache/manifold40")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--stage1_checkpoint_path", type=str, required=True)
    parser.add_argument("--clip_path", type=str, default="pretrained_weights/ViT-L-14-336px.pt")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
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
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
    parser.add_argument("--geometry_distill_weight", type=float, default=0.2)
    parser.add_argument("--direct_loss_weight", type=float, default=0.3)
    parser.add_argument("--text_loss_weight", type=float, default=1.0)
    parser.add_argument("--combined_loss_weight", type=float, default=0.0)
    parser.add_argument("--text_logit_weight", type=float, default=0.3)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
