import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from logger import get_logger
from mesh_cls_common import (
    build_mesh_components,
    compute_alignment_loss,
    encode_geometry_text,
    move_mesh_inputs_to_device,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_prompt_tuning_modules(
    prompt_learner,
    freeze_mesh_encoder: bool,
    tune_mesh_global_proj_only: bool,
    freeze_geometry_classifier: bool,
):
    if not freeze_mesh_encoder:
        pass
    else:
        for param in prompt_learner.mesh_encoder.parameters():
            param.requires_grad = False

        if tune_mesh_global_proj_only:
            for param in prompt_learner.mesh_encoder.global_proj.parameters():
                param.requires_grad = True

    if freeze_geometry_classifier:
        for param in prompt_learner.geometry_classifier.parameters():
            param.requires_grad = False


def train(args):
    logger = get_logger(args.save_path)
    args.load_views = False
    dataset, model, prompt_learner, device = build_mesh_components(args, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start_epoch = 0
    if args.resume_checkpoint_path:
        checkpoint = torch.load(args.resume_checkpoint_path, map_location="cpu")
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        start_epoch = int(checkpoint.get("epoch", 0))
        has_resume_state = all(key in checkpoint for key in ("optimizer", "scheduler", "epoch"))
        if has_resume_state:
            logger.info("resume stage1 from %s at epoch %d", args.resume_checkpoint_path, start_epoch)
        else:
            logger.info(
                "load stage1 weights from %s without optimizer/scheduler state; restart training from epoch 0",
                args.resume_checkpoint_path,
            )
            start_epoch = 0
    elif args.init_geometry_checkpoint_path:
        checkpoint = torch.load(args.init_geometry_checkpoint_path, map_location="cpu")
        prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=False)
        logger.info("initialize stage1 prompt tuning from geometry checkpoint %s", args.init_geometry_checkpoint_path)
    configure_prompt_tuning_modules(
        prompt_learner,
        freeze_mesh_encoder=args.freeze_mesh_encoder,
        tune_mesh_global_proj_only=args.tune_mesh_global_proj_only,
        freeze_geometry_classifier=args.freeze_geometry_classifier,
    )
    prompt_params = [
        prompt_learner.ctx,
        *list(prompt_learner.compound_prompts_text),
    ]
    prompt_param_ids = {id(param) for param in prompt_params}
    geometry_params = [
        param for param in prompt_learner.parameters() if id(param) not in prompt_param_ids and param.requires_grad
    ]
    prompt_params = [param for param in prompt_params if param.requires_grad]
    optimizer_groups = []
    if geometry_params:
        optimizer_groups.append({"params": geometry_params, "lr": args.learning_rate, "weight_decay": args.weight_decay})
    if prompt_params:
        optimizer_groups.append({"params": prompt_params, "lr": args.prompt_learning_rate, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)
    if args.resume_checkpoint_path and has_resume_state:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    trainable_mesh_encoder_params = sum(param.numel() for param in prompt_learner.mesh_encoder.parameters() if param.requires_grad)
    trainable_geometry_classifier_params = sum(
        param.numel() for param in prompt_learner.geometry_classifier.parameters() if param.requires_grad
    )
    total_trainable_params = sum(param.numel() for param in prompt_learner.parameters() if param.requires_grad)
    logger.info(
        "stage1_prompt_tuning_config: freeze_mesh_encoder=%s, tune_mesh_global_proj_only=%s, freeze_geometry_classifier=%s, trainable_mesh_encoder_params=%d, trainable_geometry_classifier_params=%d, total_trainable_params=%d",
        args.freeze_mesh_encoder,
        args.tune_mesh_global_proj_only,
        args.freeze_geometry_classifier,
        trainable_mesh_encoder_params,
        trainable_geometry_classifier_params,
        total_trainable_params,
    )
    if start_epoch >= args.epochs:
        raise ValueError(
            f"resume checkpoint epoch {start_epoch} is not smaller than target epochs {args.epochs}; "
            "increase --epochs to continue training"
        )
    model.eval()
    prompt_learner.train()

    for epoch in range(start_epoch, args.epochs):
        geo_cls_loss_values = []
        align_loss_values = []
        total_loss_values = []
        acc_values = []
        for batch in tqdm(dataloader, desc=f"stage1 epoch {epoch + 1}/{args.epochs}"):
            class_id = batch["class_id"].to(device)
            mesh_inputs = move_mesh_inputs_to_device(batch["mesh_inputs"], device)
            text_features, _, geometry_embedding, geometry_logits = encode_geometry_text(model, prompt_learner, mesh_inputs)
            geo_cls_loss = F.cross_entropy(geometry_logits, class_id, label_smoothing=args.label_smoothing)
            align_loss = compute_alignment_loss(geometry_embedding, text_features, class_id)
            loss = geo_cls_loss + args.align_loss_weight * align_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(prompt_learner.parameters(), args.grad_clip_norm)
            optimizer.step()

            preds = geometry_logits.argmax(dim=1)
            geo_cls_loss_values.append(geo_cls_loss.item())
            align_loss_values.append(align_loss.item())
            total_loss_values.append(loss.item())
            acc_values.append((preds == class_id).float().mean().item())

        logger.info(
            "epoch [%d/%d], stage1_total_loss: %.4f, stage1_geo_cls_loss: %.4f, stage1_align_loss: %.4f, stage1_acc: %.4f",
            epoch + 1,
            args.epochs,
            float(np.mean(total_loss_values)),
            float(np.mean(geo_cls_loss_values)),
            float(np.mean(align_loss_values)),
            float(np.mean(acc_values)),
        )
        scheduler.step()
        if (epoch + 1) % args.save_freq == 0:
            torch.save(
                {
                    "prompt_learner": prompt_learner.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "classnames": dataset.classnames,
                    "args": vars(args),
                },
                os.path.join(args.save_path, f"epoch_{epoch + 1}.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mesh Classification Stage1", add_help=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default="./.cache/manifold40")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--clip_path", type=str, default="pretrained_weights/ViT-L-14-336px.pt")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--point_size", type=int, default=336)
    parser.add_argument("--num_sampled_faces", type=int, default=500)
    parser.add_argument("--num_views", type=int, default=9)
    parser.add_argument("--mesh_mask_ratio", type=float, default=0.0)
    parser.add_argument("--render_backend", type=str, default="pyrender_egl")
    parser.add_argument("--disable_render_cache_generation", action="store_true")
    parser.add_argument("--init_geometry_checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_checkpoint_path", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--align_loss_weight", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--freeze_mesh_encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune_mesh_global_proj_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_geometry_classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
