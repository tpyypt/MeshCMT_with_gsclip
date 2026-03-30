from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceRotateConvolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        feature = (
            self.rotate_mlp(corners[:, :6])
            + self.rotate_mlp(corners[:, 3:9])
            + self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], dim=1))
        ) / 3.0
        return self.fusion_mlp(feature)


class FaceKernelCorrelation(nn.Module):
    def __init__(self, num_kernel: int = 64, sigma: float = 0.2):
        super().__init__()
        self.num_kernel = num_kernel
        self.sigma = sigma
        self.weight_alpha = nn.Parameter(torch.rand(1, num_kernel, 4) * np.pi)
        self.weight_beta = nn.Parameter(torch.rand(1, num_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(num_kernel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, normals: torch.Tensor, neighbor_index: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_faces = normals.size()
        center = normals.unsqueeze(2).expand(-1, -1, self.num_kernel, -1).unsqueeze(4)
        neighbor = torch.gather(
            normals.unsqueeze(3).expand(-1, -1, -1, neighbor_index.size(-1)),
            2,
            neighbor_index.unsqueeze(1).expand(-1, 3, -1, -1),
        )
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.num_kernel, -1, -1)
        feature = torch.cat([center, neighbor], dim=4)
        feature = feature.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4)

        weight = torch.cat(
            [
                torch.sin(self.weight_alpha) * torch.cos(self.weight_beta),
                torch.sin(self.weight_alpha) * torch.sin(self.weight_beta),
                torch.cos(self.weight_alpha),
            ],
            dim=0,
        )
        weight = weight.unsqueeze(0).expand(batch_size, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, num_faces, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, neighbor_index.size(-1) + 1, -1)

        distance = torch.sum((feature - weight) ** 2, dim=1)
        feature = torch.sum(
            torch.sum(torch.exp(distance / (-2 * (self.sigma ** 2))), dim=4),
            dim=3,
        ) / float((neighbor_index.size(-1) + 1) * 4)
        return self.relu(self.bn(feature))


class SpatialDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.spatial_mlp(centers)


class StructuralDescriptor(nn.Module):
    def __init__(self, num_kernel: int = 64, sigma: float = 0.2):
        super().__init__()
        self.face_rotate_convolution = FaceRotateConvolution()
        self.face_kernel_correlation = FaceKernelCorrelation(num_kernel=num_kernel, sigma=sigma)
        self.structural_mlp = nn.Sequential(
            nn.Conv1d(64 + 3 + num_kernel, 131, kernel_size=1),
            nn.BatchNorm1d(131),
            nn.ReLU(inplace=True),
            nn.Conv1d(131, 131, kernel_size=1),
            nn.BatchNorm1d(131),
            nn.ReLU(inplace=True),
        )

    def forward(self, corners: torch.Tensor, normals: torch.Tensor, neighbor_index: torch.Tensor) -> torch.Tensor:
        structural_feature_1 = self.face_rotate_convolution(corners)
        structural_feature_2 = self.face_kernel_correlation(normals, neighbor_index)
        return self.structural_mlp(torch.cat([structural_feature_1, structural_feature_2, normals], dim=1))


class MeshConvolution(nn.Module):
    def __init__(
        self,
        spatial_in: int,
        structural_in: int,
        spatial_out: int,
        structural_out: int,
        aggregation_method: str = "Concat",
    ):
        super().__init__()
        self.structural_in = structural_in
        self.aggregation_method = aggregation_method
        self.combination_mlp = nn.Sequential(
            nn.Conv1d(spatial_in + structural_in, spatial_out, kernel_size=1),
            nn.BatchNorm1d(spatial_out),
            nn.ReLU(inplace=True),
        )
        if aggregation_method == "Concat":
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(structural_in * 2, structural_in, kernel_size=1),
                nn.BatchNorm2d(structural_in),
                nn.ReLU(inplace=True),
            )
        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(structural_in, structural_out, kernel_size=1),
            nn.BatchNorm1d(structural_out),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        structural_features: torch.Tensor,
        neighbor_index: torch.Tensor,
    ):
        spatial_features = self.combination_mlp(torch.cat([spatial_features, structural_features], dim=1))
        neighbors = torch.gather(
            structural_features.unsqueeze(3).expand(-1, -1, -1, neighbor_index.size(-1)),
            2,
            neighbor_index.unsqueeze(1).expand(-1, self.structural_in, -1, -1),
        )

        if self.aggregation_method == "Concat":
            structural_features = torch.cat(
                [
                    structural_features.unsqueeze(3).expand(-1, -1, -1, neighbor_index.size(-1)),
                    neighbors,
                ],
                dim=1,
            )
            structural_features = self.concat_mlp(structural_features)
            structural_features = torch.max(structural_features, dim=3)[0]
        elif self.aggregation_method == "Max":
            structural_features = torch.cat([structural_features.unsqueeze(3), neighbors], dim=3)
            structural_features = torch.max(structural_features, dim=3)[0]
        elif self.aggregation_method == "Average":
            structural_features = torch.cat([structural_features.unsqueeze(3), neighbors], dim=3)
            structural_features = torch.sum(structural_features, dim=3) / float(neighbor_index.size(-1) + 1)
        else:
            raise ValueError(f"Unsupported aggregation_method: {self.aggregation_method}")

        structural_features = self.aggregation_mlp(structural_features)
        return spatial_features, structural_features


class MeshNetEncoder(nn.Module):
    """MeshNet-style encoder adapted closely from iMoonLab/MeshNet."""

    def __init__(
        self,
        mask_ratio: float = 0.0,
        num_kernel: int = 64,
        sigma: float = 0.2,
        aggregation_method: str = "Concat",
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(num_kernel=num_kernel, sigma=sigma)
        self.mesh_conv1 = MeshConvolution(64, 131, 256, 256, aggregation_method=aggregation_method)
        self.mesh_conv2 = MeshConvolution(256, 256, 512, 512, aggregation_method=aggregation_method)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(256 + 512 + 1024, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
        )
        self.local_proj = nn.Sequential(
            nn.Conv1d(1024, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, mesh_inputs: Dict[str, torch.Tensor]):
        centers = mesh_inputs["centers"].transpose(1, 2)
        corners = mesh_inputs["corners"].transpose(1, 2)
        normals = mesh_inputs["normals"].transpose(1, 2)
        neighbor_index = mesh_inputs["neighbors"].long()
        valid_faces = mesh_inputs.get("valid_faces")

        spatial_feature_0 = self.spatial_descriptor(centers)
        structural_feature_0 = self.structural_descriptor(corners, normals, neighbor_index)
        spatial_feature_1, structural_feature_1 = self.mesh_conv1(spatial_feature_0, structural_feature_0, neighbor_index)
        spatial_feature_2, structural_feature_2 = self.mesh_conv2(spatial_feature_1, structural_feature_1, neighbor_index)
        spatial_feature_3 = self.fusion_mlp(torch.cat([spatial_feature_2, structural_feature_2], dim=1))
        fused = self.concat_mlp(torch.cat([spatial_feature_1, spatial_feature_2, spatial_feature_3], dim=1))
        if self.training and self.mask_ratio > 0:
            num_faces = fused.shape[2]
            keep_faces = max(1, int(num_faces * (1.0 - self.mask_ratio)))
            sampled_idx = torch.randperm(num_faces, device=fused.device)[:keep_faces]
            fused = fused[:, :, sampled_idx]
            if valid_faces is not None:
                valid_faces = valid_faces[:, sampled_idx]
        if valid_faces is not None:
            face_mask = valid_faces.unsqueeze(1).to(fused.device)
            masked_fused = fused.masked_fill(face_mask == 0, float("-inf"))
            max_pooled = torch.max(masked_fused, dim=2)[0]
            max_pooled = torch.where(torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled))
        else:
            max_pooled = F.adaptive_max_pool1d(fused, 1).squeeze(-1)
        global_feature = self.global_proj(max_pooled)
        local_feature = self.local_proj(fused).transpose(1, 2)
        return global_feature, local_feature


def aggregate_view_features(image_features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
    if image_features.shape[0] != batch_size * num_views:
        raise ValueError(
            f"Feature count {image_features.shape[0]} does not match batch_size*num_views {batch_size * num_views}"
        )
    return image_features.view(batch_size, num_views, -1).mean(dim=1)
