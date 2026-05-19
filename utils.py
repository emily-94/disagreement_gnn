

import copy
import pandas as pd
import math
import os
import random
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader




def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = int(patience)
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.bad = 0
        self.state = None

    def step(self, metric_val: float, model: nn.Module) -> bool:
        improved = (metric_val < self.best) if self.mode == "min" else (metric_val > self.best)
        if improved:
            self.best = float(metric_val)
            self.bad = 0
            # Deepcopy can be large; but is the standard approach.
            self.state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.bad += 1
            return self.bad >= self.patience




def train_one_epoch(
    model,
    loader,
    loss_fn,
    opt,
    grad_clip,
    device= None):
 

    if device is None:
        device = get_device()

    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        pred = model(batch)
        y = batch.y.view(-1).to(pred.dtype)
        loss = loss_fn(pred, y)


        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device=None):
    if device is None:
        device = get_device()

    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)
        y = batch.y.view(-1).to(pred.dtype)
        loss = loss_fn(pred, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_mae += (pred - y).abs().sum().item()
        n += bs

    return total_loss / max(n, 1), total_mae / max(n, 1)


def collect_predictions_single(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> "pd.DataFrame":
    """
    Returns DataFrame with columns: conv_id, fold, y_true_rating, y_pred_rating
    """
  
    if device is None:
        device = get_device()

    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)                       # [B]
            y = batch.y.view(-1).to(pred.dtype)       # [B]
            B = int(batch.num_graphs)

            conv_ids = getattr(batch, "conv_id", None)
            if isinstance(conv_ids, str):
                conv_ids = [conv_ids] * B
            elif conv_ids is None:
                conv_ids = [f"g{i}" for i in range(B)]
            else:
                conv_ids = list(conv_ids)

            folds = getattr(batch, "fold", None)
            if isinstance(folds, (int, np.integer)):
                folds = [int(folds)] * B
            elif folds is None:
                folds = [-1] * B
            else:
                folds = list(folds)

            pred_cpu = pred.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()

            for i in range(B):
                rows.append({
                    "conv_id": conv_ids[i],
                    "fold": int(folds[i]),
                    "y_true_rating": float(y_cpu[i]),
                    "y_pred_rating": float(pred_cpu[i]),
                })

            del batch, pred, y

    return pd.DataFrame(rows, columns=["conv_id", "fold", "y_true_rating", "y_pred_rating"])


def collect_edge_attention(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    topk: Optional[int] = 5,
) -> "pd.DataFrame":
    """
    Collect per-edge attention weights into a DataFrame.

    """

    import pandas as pd
    if device is None:
        device = get_device()

    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            pred, extras = model(batch, return_attn=True)

            # extras already CPU in your model; but be defensive:
            alpha = extras["alpha_all"]
            edge_index = extras["edge_index"]
            edge_type = extras.get("edge_type", None)
            node2g = extras["batch"]

            src = edge_index[0]
            dst = edge_index[1]
            edge_graph = node2g[dst]  # graph id for each edge

            B = int(batch.num_graphs)
            conv_ids = getattr(batch, "conv_id", None)
            if isinstance(conv_ids, str):
                conv_ids = [conv_ids] * B
            elif conv_ids is None:
                conv_ids = [f"g{i}" for i in range(B)]
            else:
                conv_ids = list(conv_ids)

            folds = getattr(batch, "fold", None)
            if isinstance(folds, (int, np.integer)):
                folds = [int(folds)] * B
            elif folds is None:
                folds = [-1] * B
            else:
                folds = list(folds)

            # Optional node context
            speakers = batch.speakers.detach().cpu() if hasattr(batch, "speakers") else None
            time_idx = batch.time_idx.detach().cpu() if hasattr(batch, "time_idx") else None

            E = int(alpha.numel())
            for e in range(E):
                g = int(edge_graph[e].item())
                s = int(src[e].item())
                d = int(dst[e].item())
                row = {
                    "conv_id": conv_ids[g],
                    "fold": int(folds[g]),
                    "src": s,
                    "dst": d,
                    "alpha": float(alpha[e].item()),
                }
                if edge_type is not None:
                    row["edge_type"] = int(edge_type[e].item())

                if time_idx is not None:
                    row["src_time"] = int(time_idx[s].item())
                    row["dst_time"] = int(time_idx[d].item())
                    row["lag"] = int(time_idx[d].item() - time_idx[s].item())

                if speakers is not None:
                    row["src_spk"] = int(speakers[s].item())
                    row["dst_spk"] = int(speakers[d].item())

                rows.append(row)

            del batch, pred, extras

    df = pd.DataFrame(rows)

    if topk is not None and not df.empty:
        df = (df.sort_values(["conv_id", "dst", "alpha"], ascending=[True, True, False])
                .groupby(["conv_id", "dst"], as_index=False)
                .head(int(topk))
                .reset_index(drop=True))

    return df


def collect_node_attention(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    topk: Optional[int] = 10,
) -> "pd.DataFrame":
    """
    Collect per-node pooling attention weights into a DataFrame.
  
    """
    import pandas as pd
    if device is None:
        device = get_device()

    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            pred, extras = model(batch, return_attn=True)

            alpha_nodes = extras.get("alpha_nodes", None)
            if alpha_nodes is None:
                # pooling not providing node weights
                del batch, pred, extras
                continue

            node2g = extras["batch"]

            speakers = batch.speakers.detach().cpu() if hasattr(batch, "speakers") else None
            time_idx = batch.time_idx.detach().cpu() if hasattr(batch, "time_idx") else None

            B = int(batch.num_graphs)
            conv_ids = getattr(batch, "conv_id", None)
            if isinstance(conv_ids, str):
                conv_ids = [conv_ids] * B
            elif conv_ids is None:
                conv_ids = [f"g{i}" for i in range(B)]
            else:
                conv_ids = list(conv_ids)

            folds = getattr(batch, "fold", None)
            if isinstance(folds, (int, np.integer)):
                folds = [int(folds)] * B
            elif folds is None:
                folds = [-1] * B
            else:
                folds = list(folds)

            N = int(alpha_nodes.numel())
            for n in range(N):
                g = int(node2g[n].item())
                row = {
                    "conv_id": conv_ids[g],
                    "fold": int(folds[g]),
                    "node": int(n),
                    "alpha_node": float(alpha_nodes[n].item()),
                }
                if time_idx is not None:
                    row["time"] = int(time_idx[n].item())
                if speakers is not None:
                    row["spk"] = int(speakers[n].item())
                rows.append(row)

            del batch, pred, extras

    df = pd.DataFrame(rows)

    if topk is not None and not df.empty:
        df = (df.sort_values(["conv_id", "alpha_node"], ascending=[True, False])
                .groupby(["conv_id"], as_index=False)
                .head(int(topk))
                .reset_index(drop=True))

    return df


