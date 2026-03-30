from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import AnomalyCLIP_lib
from dataset_mesh import MeshMultiViewDataset
from prompt_mesh import GeometryPromptLearner
from utils import get_transform


def move_mesh_inputs_to_device(mesh_inputs: Dict[str, torch.Tensor], device: str):
    return {key: value.to(device) for key, value in mesh_inputs.items()}


def build_mesh_components(args, split="train"):
    preprocess, _, _ = get_transform(args)
    dataset = MeshMultiViewDataset(
        root=args.data_root,
        split=split,
        transform=preprocess,
        image_size=args.image_size,
        cache_root=args.cache_root,
        num_sampled_faces=args.num_sampled_faces,
        num_views=args.num_views,
        meta_path=getattr(args, "meta_path", None),
        render_on_the_fly=not getattr(args, "disable_render_cache_generation", False),
        render_backend=getattr(args, "render_backend", "pyrender_egl"),
        load_views=getattr(args, "load_views", True),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomalyclip_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
        "mesh_mask_ratio": getattr(args, "mesh_mask_ratio", 0.0),
    }
    model, _ = AnomalyCLIP_lib.load(args.clip_path, device=device, design_details=anomalyclip_parameters)
    prompt_learner = GeometryPromptLearner(model.to("cpu"), anomalyclip_parameters, dataset.classnames)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)
    return dataset, model, prompt_learner, device


class VisualClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def encode_geometry_text(model, prompt_learner, mesh_inputs):
    (
        prompts,
        tokenized_prompts,
        compound_prompts_text,
        mesh_global,
        geometry_embedding,
        geometry_logits,
    ) = prompt_learner(mesh_inputs)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    batch_size = mesh_global.shape[0]
    text_features = text_features.view(batch_size, prompt_learner.n_cls, -1)
    text_features = F.normalize(text_features, dim=-1)
    geometry_embedding = F.normalize(geometry_embedding, dim=-1)
    return text_features, mesh_global, geometry_embedding, geometry_logits


def compute_global_logits(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    image_features = F.normalize(image_features, dim=-1)
    return torch.einsum("bd,bcd->bc", image_features, text_features) / 0.07


def compute_alignment_loss(geometry_embedding: torch.Tensor, text_features: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
    geometry_embedding = F.normalize(geometry_embedding, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    alignment_logits = torch.einsum("bd,bcd->bc", geometry_embedding, text_features) / 0.07
    return F.cross_entropy(alignment_logits, class_id)


def freeze_prompt_semantics(prompt_learner):
    frozen_modules = [
        prompt_learner.mesh_feature_to_prompt,
        prompt_learner.mesh_feature_to_text,
        prompt_learner.ctx,
        prompt_learner.compound_prompts_text,
        prompt_learner.geometry_classifier,
    ]
    for module in frozen_modules:
        if isinstance(module, torch.nn.Parameter):
            module.requires_grad = False
        elif isinstance(module, torch.nn.ParameterList):
            for param in module:
                param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False


def collect_trainable_params(module):
    return [param for param in module.parameters() if param.requires_grad]
