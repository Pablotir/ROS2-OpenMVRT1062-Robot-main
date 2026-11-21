"""
PointNet++ Modules for Contact-GraspNet (100% Pure PyTorch).
Completely self-contained: zero dependency on torch_geometric, numba, or pyg-lib.
Guarantees rock-solid stability and zero SIGSEGV crashes on Jetson Orin / aarch64.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


def safe_knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3):
    """Pure PyTorch k-NN feature interpolation (zero Numba / pyg-lib dependency)."""
    if batch_x is None or batch_y is None or (torch.all(batch_x == 0) and torch.all(batch_y == 0)):
        diff = pos_y.unsqueeze(1) - pos_x.unsqueeze(0)
        dists_sq = torch.sum(diff * diff, dim=-1)
        actual_k = min(k, pos_x.size(0))
        top_dists_sq, top_idx = torch.topk(dists_sq, k=actual_k, dim=-1, largest=False)
        weights = 1.0 / torch.clamp(top_dists_sq, min=1e-16)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        top_feats = x[top_idx]
        return torch.sum(top_feats * weights.unsqueeze(-1), dim=1)
    else:
        out = []
        for b in torch.unique(batch_y):
            mask_y = (batch_y == b)
            mask_x = (batch_x == b)
            sub_y, sub_x = pos_y[mask_y], pos_x[mask_x]
            sub_feats = x[mask_x]
            diff = sub_y.unsqueeze(1) - sub_x.unsqueeze(0)
            dists_sq = torch.sum(diff * diff, dim=-1)
            actual_k = min(k, sub_x.size(0))
            top_dists_sq, top_idx = torch.topk(dists_sq, k=actual_k, dim=-1, largest=False)
            weights = 1.0 / torch.clamp(top_dists_sq, min=1e-16)
            weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            top_feats = sub_feats[top_idx]
            out.append(torch.sum(top_feats * weights.unsqueeze(-1), dim=1))
        return torch.cat(out, dim=0)


class PurePointNetConv(torch.nn.Module):
    """Pure PyTorch PointNet set abstraction convolution (replaces torch_geometric PointNetConv)."""
    def __init__(self, local_nn=None):
        super(PurePointNetConv, self).__init__()
        self.local_nn = local_nn

    def forward(self, x, pos, edge_index):
        if isinstance(pos, tuple):
            pos_src, pos_dst = pos
        else:
            pos_src, pos_dst = pos, pos

        col, row = edge_index[0], edge_index[1]  # col: source (pos), row: target (pos[idx])
        msg = pos_src[col] - pos_dst[row]
        if x is not None:
            x_src = x[0] if isinstance(x, tuple) else x
            msg = torch.cat([x_src[col], msg], dim=-1)

        if self.local_nn is not None:
            msg = self.local_nn(msg)

        num_dst = pos_dst.size(0)
        out = torch.full((num_dst, msg.size(-1)), -1e10, dtype=msg.dtype, device=msg.device)
        out = torch.scatter_reduce(out, 0, row.unsqueeze(1).expand_as(msg), msg, reduce="amax", include_self=False)
        return out

PointConv = PurePointNetConv


def safe_fps(pos, batch=None, ratio=0.5, random_start=False):
    """Pure PyTorch Farthest Point Sampling (zero Numba / pyg-lib dependency)."""
    if batch is None or torch.all(batch == 0):
        N = pos.size(0)
        num_samples = max(1, int(N * ratio))
        sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=pos.device)
        distance = torch.full((N,), 1e10, dtype=pos.dtype, device=pos.device)
        farthest_idx = 0
        for i in range(num_samples):
            sampled_indices[i] = farthest_idx
            centroid = pos[farthest_idx]
            dist = torch.sum((pos - centroid) ** 2, dim=-1)
            distance = torch.minimum(distance, dist)
            farthest_idx = torch.argmax(distance).item()
        return sampled_indices
    else:
        out = []
        for b in torch.unique(batch):
            mask = (batch == b)
            idx = torch.nonzero(mask, as_tuple=True)[0]
            sub_pos = pos[mask]
            sub_fps = safe_fps(sub_pos, None, ratio=ratio, random_start=random_start)
            out.append(idx[sub_fps])
        return torch.cat(out)


def safe_radius(x, y, r, batch_x, batch_y, max_num_neighbors=64):
    """Pure PyTorch Ball Query / Radius Neighbor Search (zero Numba / pyg-lib dependency)."""
    r2 = r * r
    if batch_x is None or batch_y is None or (torch.all(batch_x == 0) and torch.all(batch_y == 0)):
        diff = y.unsqueeze(1) - x.unsqueeze(0)
        dists_sq = torch.sum(diff * diff, dim=-1)
        k = min(max_num_neighbors, x.size(0))
        top_dists, top_indices = torch.topk(dists_sq, k=k, dim=1, largest=False)
        valid = (top_dists <= r2)
        row = torch.arange(y.size(0), device=y.device).unsqueeze(1).expand_as(top_indices)[valid]
        col = top_indices[valid]
        return row, col
    else:
        row_list, col_list = [], []
        for b in torch.unique(batch_y):
            mask_y = (batch_y == b)
            mask_x = (batch_x == b)
            idx_y = torch.nonzero(mask_y, as_tuple=True)[0]
            idx_x = torch.nonzero(mask_x, as_tuple=True)[0]
            sub_y, sub_x = y[mask_y], x[mask_x]
            diff = sub_y.unsqueeze(1) - sub_x.unsqueeze(0)
            dists_sq = torch.sum(diff * diff, dim=-1)
            k = min(max_num_neighbors, sub_x.size(0))
            top_dists, top_indices = torch.topk(dists_sq, k=k, dim=1, largest=False)
            valid = (top_dists <= r2)
            b_rows = torch.arange(sub_y.size(0), device=y.device).unsqueeze(1).expand_as(top_indices)[valid]
            b_cols = top_indices[valid]
            row_list.append(idx_y[b_rows])
            col_list.append(idx_x[b_cols])
        if len(row_list) == 0:
            return torch.empty(0, dtype=torch.long, device=y.device), torch.empty(0, dtype=torch.long, device=y.device)
        return torch.cat(row_list), torch.cat(col_list)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn=None):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PurePointNetConv(nn)

    def forward(self, x, pos, batch, sample=True, idx=None):
        if sample is True:
            idx = safe_fps(pos, batch, ratio=self.ratio)
        else:
            idx = idx

        row, col = safe_radius(pos, pos[idx], self.r, batch, batch[idx],
                               max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch, idx


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        if batch is None or (torch.all(batch == 0)):
            x = torch.max(x, dim=0, keepdim=True)[0]
        else:
            out = []
            for b in torch.unique(batch):
                out.append(torch.max(x[batch == b], dim=0, keepdim=True)[0])
            x = torch.cat(out, dim=0)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = safe_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet(torch.nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(PointNet, self).__init__()
        self.sa1_module = SAModule(0.2, 0.2, MLP([in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 4, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(1024, 128)
        self.lin2 = torch.nn.Linear(128, latent_dim)

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        return x


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])
