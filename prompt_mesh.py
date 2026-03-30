from typing import List, Sequence, Union

import torch
import torch.nn as nn
from packaging import version

from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from mesh_encoder import MeshNetEncoder

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result_dtype = torch.long if version.parse(torch.__version__) < version.parse("1.8.0") else torch.int
    result = torch.zeros(len(all_tokens), context_length, dtype=result_dtype)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result


class GeometryPromptLearner(nn.Module):
    def __init__(self, clip_model, design_details, classnames: Sequence[str]):
        super().__init__()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(self.classnames)
        self.n_ctx = design_details["Prompt_length"]
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]

        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.mesh_encoder = MeshNetEncoder(mask_ratio=design_details.get("mesh_mask_ratio", 0.0))
        self.mesh_feature_to_prompt = nn.Sequential(
            nn.Linear(512, ctx_dim),
            nn.GELU(),
            nn.LayerNorm(ctx_dim),
            nn.Dropout(p=0.1),
        )
        self.mesh_feature_to_text = nn.Sequential(
            nn.Linear(512, clip_model.text_projection.shape[1]),
            nn.GELU(),
            nn.LayerNorm(clip_model.text_projection.shape[1]),
            nn.Dropout(p=0.1),
        )
        self.geometry_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.n_cls),
        )

        self.ctx = nn.Parameter(torch.empty(self.n_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx, std=0.02)

        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim)) for _ in range(self.compound_prompts_depth - 1)]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        prompt_prefix = " ".join(["X"] * (self.n_ctx + 1))
        prompts = [f"{prompt_prefix} a 3d mesh model of {name}." for name in self.classnames]
        tokenized_prompts = torch.cat([tokenize(prompt) for prompt in prompts], dim=0)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 2 + self.n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self, mesh_inputs):
        global_feature, _ = self.mesh_encoder(mesh_inputs)
        dynamic_token = self.mesh_feature_to_prompt(global_feature).unsqueeze(1)
        geometry_embedding = self.mesh_feature_to_text(global_feature)
        geometry_logits = self.geometry_classifier(global_feature)

        batch_size = dynamic_token.shape[0]
        shared_ctx = self.ctx.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_cls, -1, -1)
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        dynamic_token = dynamic_token.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prompts = torch.cat([prefix, shared_ctx, dynamic_token, suffix], dim=2)
        prompts = prompts.reshape(batch_size * self.n_cls, prompts.shape[2], prompts.shape[3])
        tokenized_prompts = self.tokenized_prompts.repeat(batch_size, 1)
        return (
            prompts,
            tokenized_prompts,
            self.compound_prompts_text,
            global_feature,
            geometry_embedding,
            geometry_logits,
        )
