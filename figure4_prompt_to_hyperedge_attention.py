# -*- coding: utf-8 -*-

import os
import math
import argparse
import warnings
from typing import Tuple, List, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import ReHypeNet
from utils import load_EOD_data, load_hypergraph, prepare_batch_data

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', -1) + 1}")
    print(f"Seed: {checkpoint.get('seed', 'N/A')}")
    print(f"Valid F1: {safe_float(checkpoint.get('valid_f1')):.4f}")
    print(f"Test F1: {safe_float(checkpoint.get('test_f1')):.4f}")
    print(f"Test AUC: {safe_float(checkpoint.get('test_auc')):.4f}")
    return checkpoint


def build_model_from_checkpoint(checkpoint, cli_args):
    if "args" not in checkpoint:
        raise KeyError("The checkpoint does not contain checkpoint['args'].")

    saved_args = argparse.Namespace(**checkpoint["args"])

    if cli_args.data_path is not None:
        saved_args.data_path = cli_args.data_path
    if cli_args.opath is not None:
        saved_args.opath = cli_args.opath
    if cli_args.market_name is not None:
        saved_args.market_name = cli_args.market_name
    if cli_args.hypergraph_ind_path is not None:
        saved_args.hypergraph_ind_path = cli_args.hypergraph_ind_path

    H_ind = load_hypergraph(saved_args.hypergraph_ind_path, saved_args.num_stocks)

    model = ReHypeNet(
        num_nodes=saved_args.num_stocks,
        feature_dim=saved_args.feature_dim,
        lookback_length=saved_args.lookback_length,
        H_ind=H_ind,
        temporal_hidden_dim=saved_args.temporal_hidden_dim,
        temporal_dropout=saved_args.temporal_dropout,
        k_proto=saved_args.k_proto,
        topk_edges=getattr(saved_args, "topk_edges", 3),
        hgnn_hidden_dim=saved_args.hgnn_hidden_dim,
        mlp_hidden_dim=saved_args.mlp_hidden_dim,
        num_prompt_tokens=saved_args.num_prompt_tokens,
        attn_heads=saved_args.attn_heads,
        attn_dropout=saved_args.attn_dropout,
        resid_dropout=saved_args.resid_dropout,
        num_hgnn_layers=getattr(saved_args, "num_hgnn_layers", 1),
        num_attention_layers=getattr(saved_args, "num_attention_layers", 2),
        hgnn_dropout=getattr(saved_args, "hgnn_dropout", 0.1),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, saved_args, H_ind


def load_split_data(saved_args, split: str):
    tickers_fname = saved_args.market_name + "_tickers_qualify_dr-0.98_min-5_smooth.csv"
    tickers = np.genfromtxt(
        os.path.join(saved_args.data_path, tickers_fname),
        dtype=str,
        delimiter="\t",
        skip_header=False,
    )

    stocks_data = load_EOD_data(saved_args.opath, saved_args.market_name, tickers)

    train_start, train_end = 1, 746
    valid_start, valid_end = 762, 1002
    test_start, test_end = 1003, 1243

    train_data_raw = np.concatenate(
        [
            stocks_data[: saved_args.num_stocks, i:j, :]
            for (i, j) in [(train_start, 241), (251, 491), (506, train_end)]
        ],
        axis=1,
    )
    valid_data_raw = stocks_data[: saved_args.num_stocks, valid_start:valid_end, :]
    test_data_raw = stocks_data[: saved_args.num_stocks, test_start:test_end, :]

    train_X, train_y = prepare_batch_data(train_data_raw, saved_args.lookback_length)
    valid_X, valid_y = prepare_batch_data(valid_data_raw, saved_args.lookback_length)
    test_X, test_y = prepare_batch_data(test_data_raw, saved_args.lookback_length)

    if split == "train":
        return tickers, train_X, train_y
    if split == "valid":
        return tickers, valid_X, valid_y
    if split == "test":
        return tickers, test_X, test_y
    raise ValueError("split must be one of {'train', 'valid', 'test'}")

@torch.no_grad()
def extract_prompt_to_hyperedge_attention_per_layer(
    model: ReHypeNet,
    x_batch: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
    """
    Extract prompt-query -> hyperedge attention from every PRI layer.

    Returns:
        layer_attn_list: list of (B, K, M_total) tensors on CPU
        H_time_dyn:      (B, N, M_time) tensor on CPU
        M_ind, M_time
    """
    E_time = model.temporal_encoder(x_batch)
    H_time_dyn = model.build_dynamic_H_improved(E_time, model.time_prototypes)

    batch_size = x_batch.shape[0]
    H_ind_expanded = model.H_ind.unsqueeze(0).expand(batch_size, -1, -1)
    H_combined = torch.cat([H_ind_expanded, H_time_dyn], dim=2)

    M_ind = int(model.H_ind.shape[1])
    M_total = int(H_combined.shape[2])
    M_time = M_total - M_ind

    Z = model.input_proj(E_time)
    edge_weights_1 = model.edge_weight_gen_1(Z, H_combined)

    Z_1 = Z
    for block_idx, block in enumerate(model.hgnn_blocks_1):
        dynamic_weights = edge_weights_1 if block_idx == 0 else None
        Z_1 = block(Z_1, H_combined, dynamic_weights=dynamic_weights)

    Z_cur = Z_1
    layer_attn_list: List[torch.Tensor] = []

    for prompt_gen, attn_block in zip(
        model.deep_prompt_attention.prompt_generators,
        model.deep_prompt_attention.attention_blocks,
    ):
        batch_size_2, _, hidden_dim = Z_cur.shape
        eps = 1e-8

        Dv = H_combined.sum(dim=2).clamp_min(eps)
        De = H_combined.sum(dim=1).clamp_min(eps)
        Dv_inv_sqrt = 1.0 / torch.sqrt(Dv)
        De_inv_sqrt = 1.0 / torch.sqrt(De)

        H_norm = Dv_inv_sqrt.unsqueeze(-1) * H_combined * De_inv_sqrt.unsqueeze(1)
        H_norm = torch.nan_to_num(H_norm, nan=0.0, posinf=1.0, neginf=0.0)

        edge_features = torch.bmm(H_norm.transpose(1, 2), Z_cur)
        edge_features = prompt_gen.edge_proj(edge_features)

        queries = prompt_gen.query_vectors.unsqueeze(0).expand(batch_size_2, -1, -1)
        attn_scores = torch.bmm(queries, edge_features.transpose(1, 2)) / math.sqrt(hidden_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        layer_attn_list.append(attn_weights.detach().cpu())

        prompts = torch.bmm(attn_weights, edge_features)
        Z_cur = attn_block(Z_cur, prompts)

    return layer_attn_list, H_time_dyn.detach().cpu(), M_ind, M_time

def set_plot_style(font_size: int = 15):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = font_size
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def save_figure(fig, out_path: str, dpi: int = 300):
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".png":
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)

@torch.no_grad()
def collect_average_attention(
    model: ReHypeNet,
    X: np.ndarray,
    batch_size: int = 32,
    max_batches: int = -1,
):
    """Average prompt-to-hyperedge attention over the selected split."""
    if X.ndim != 4:
        raise ValueError(
            f"X must have shape (samples, stocks, lookback, features), got {X.shape}"
        )

    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    if max_batches is not None and max_batches > 0:
        num_batches = min(num_batches, max_batches)

    sum_list = None
    total_count = 0
    M_ind = None
    M_time = None

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        x_tensor = torch.from_numpy(X[start:end]).float().to(DEVICE)
        layer_attn_list, _, M_ind_batch, M_time_batch = (
            extract_prompt_to_hyperedge_attention_per_layer(model, x_tensor)
        )

        if sum_list is None:
            sum_list = [attn.sum(dim=0) for attn in layer_attn_list]
            M_ind = M_ind_batch
            M_time = M_time_batch
        else:
            if M_ind != M_ind_batch or M_time != M_time_batch:
                raise RuntimeError("Hyperedge dimensions changed across batches.")
            for layer_idx, attn in enumerate(layer_attn_list):
                sum_list[layer_idx] += attn.sum(dim=0)

        total_count += end - start
        print(f"[{batch_idx + 1}/{num_batches}] processed {end}/{num_samples} samples")

    if sum_list is None or M_ind is None or M_time is None:
        raise RuntimeError("No samples were processed.")

    avg_list = [tensor.numpy() / float(total_count) for tensor in sum_list]
    return avg_list, int(M_ind), int(M_time), int(total_count)

def build_all_edge_labels(M_ind: int, M_time: int) -> List[str]:
    labels = [f"Ind-e{m}" for m in range(M_ind)]
    labels.extend(f"Time-p{m}" for m in range(M_time))
    return labels


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sum = x.sum(axis=1, keepdims=True)
    return x / np.clip(row_sum, eps, None)


def zscore_by_row(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / np.clip(std, eps, None)


def make_sparse_xticks(num_edges: int, labels: List[str], max_ticks: int = 30):
    if num_edges <= max_ticks:
        pos = np.arange(num_edges)
        lab = labels
    else:
        step = int(np.ceil(num_edges / max_ticks))
        pos = np.arange(0, num_edges, step)
        if pos[-1] != num_edges - 1:
            pos = np.append(pos, num_edges - 1)
        lab = [labels[p] for p in pos]
    return pos, lab


def draw_static_dynamic_annotation(
    ax,
    M_ind: int,
    M_time: int,
    K: int,
    show_bottom_text: bool = False,
):
    if M_ind > 0 and M_time > 0:
        ax.axvline(x=M_ind - 0.5, color="white", linestyle="--", linewidth=2)
        if show_bottom_text:
            y_text = K + 1.8
            ax.text(
                (M_ind - 1) / 2.0,
                y_text,
                "Static industry hyperedges",
                ha="center",
                va="top",
                fontsize=18,
                transform=ax.transData,
                fontweight="bold",
            )
            ax.text(
                M_ind + (M_time - 1) / 2.0,
                y_text,
                "Dynamic temporal hyperedges",
                ha="center",
                va="top",
                fontsize=18,
                transform=ax.transData,
                fontweight="bold",
            )


def plot_raw_and_zscore_combined_heatmap(
    layer0_raw: np.ndarray,
    layer1_raw: np.ndarray,
    layer0_zscore: np.ndarray,
    layer1_zscore: np.ndarray,
    M_ind: int,
    M_time: int,
    out_prefix: str,
    max_xticks: int = 20,
):
    """
    Create a compact publication figure containing raw attention and row-wise
    standardized attention.

    Layout (top to bottom):
        (a) Raw attention: Layer 0, Layer 1
        (b) Row-wise standardized attention: Layer 0, Layer 1

    To avoid redundant horizontal-axis information, hyperedge tick labels,
    the shared x-axis title, and the static/dynamic group labels are displayed
    only once, below the final heatmap. All numerical analyses and attention
    values remain unchanged; this function modifies visualization only.
    """
    matrices = [layer0_raw, layer1_raw, layer0_zscore, layer1_zscore]
    shapes = [matrix.shape for matrix in matrices]
    if len(set(shapes)) != 1:
        raise ValueError(
            "All raw and z-score matrices must have the same shape, "
            f"but received {shapes}."
        )

    K, M_total = layer0_raw.shape
    if M_total != M_ind + M_time:
        raise ValueError(
            f"Matrix width ({M_total}) does not equal M_ind + M_time "
            f"({M_ind + M_time})."
        )

    all_labels = build_all_edge_labels(M_ind, M_time)
    xticks, xticklabels = make_sparse_xticks(
        M_total,
        all_labels,
        max_ticks=max_xticks,
    )

    raw_vmin = min(layer0_raw.min(), layer1_raw.min())
    raw_vmax = max(layer0_raw.max(), layer1_raw.max())
    z_vmin = min(layer0_zscore.min(), layer1_zscore.min())
    z_vmax = max(layer0_zscore.max(), layer1_zscore.max())

    fig_w = max(14.0, 0.16 * M_total + 3.8)
    fig_h = max(9.2, 0.38 * K + 7.6)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        width_ratios=[1.0, 0.018],
        height_ratios=[1.0, 1.0, 0.16, 1.0, 1.0],
        hspace=0.06,
        wspace=0.035,
    )

    ax_raw0 = fig.add_subplot(gs[0, 0])
    ax_raw1 = fig.add_subplot(gs[1, 0], sharex=ax_raw0)
    ax_z0 = fig.add_subplot(gs[3, 0], sharex=ax_raw0)
    ax_z1 = fig.add_subplot(gs[4, 0], sharex=ax_raw0)
    cax_raw = fig.add_subplot(gs[0:2, 1])
    cax_z = fig.add_subplot(gs[3:5, 1])
    axes = [ax_raw0, ax_raw1, ax_z0, ax_z1]

    prompt_labels = [f"Prompt-{i + 1}" for i in range(K)]

    layer_fontsize = 18
    prompt_fontsize = 17
    panel_fontsize = 21
    x_tick_fontsize = 15
    cbar_label_fontsize = 18
    cbar_tick_fontsize = 14
    # -------------------------
    # (a) Raw attention
    # -------------------------
    raw_images = []
    for ax, matrix, layer_name in zip(
        [ax_raw0, ax_raw1],
        [layer0_raw, layer1_raw],
        ["Layer 0", "Layer 1"],
    ):
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=raw_vmin,
            vmax=raw_vmax,
        )
        raw_images.append(image)
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels(prompt_labels, fontsize=prompt_fontsize)
        ax.tick_params(axis="y", pad=2)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        draw_static_dynamic_annotation(
            ax,
            M_ind,
            M_time,
            K,
            show_bottom_text=False,
        )
        layer_y = 0.93 if layer_name == "Layer 1" else 1.01

        ax.text(
            -0.018,
            layer_y,
            layer_name,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=layer_fontsize,
            fontweight="bold",
        )

    ax_raw0.text(
        0.5,
        1.05,
        "(a) Raw prompt-to-hyperedge attention",
        transform=ax_raw0.transAxes,
        ha="center",
        va="bottom",
        fontsize=panel_fontsize,
        fontweight="bold",
    )

    raw_cbar = fig.colorbar(raw_images[0], cax=cax_raw)
    raw_cbar.ax.set_ylabel(
        "Average attention weight",
        rotation=90,
        fontsize=cbar_label_fontsize,
        labelpad=10,
    )
    raw_cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    # -------------------------
    # (b) Row-wise standardized attention
    # -------------------------
    z_images = []
    for ax, matrix, layer_name in zip(
        [ax_z0, ax_z1],
        [layer0_zscore, layer1_zscore],
        ["Layer 0", "Layer 1"],
    ):
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=z_vmin,
            vmax=z_vmax,
        )
        z_images.append(image)
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels(prompt_labels, fontsize=prompt_fontsize)
        ax.tick_params(axis="y", pad=2)
        draw_static_dynamic_annotation(
            ax,
            M_ind,
            M_time,
            K,
            show_bottom_text=False,
        )
        layer_y = 0.93 if layer_name == "Layer 1" else 1.01

        ax.text(
            -0.018,
            layer_y,
            layer_name,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=layer_fontsize,
            fontweight="bold",
        )

    ax_z0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_z1.set_xticks(xticks)
    ax_z1.set_xticklabels(
        xticklabels,
        rotation=50,
        ha="right",
        fontsize=x_tick_fontsize,
    )
    ax_z1.tick_params(axis="x", pad=2)

    ax_z1.text(
        (M_ind - 1) / 2.0,
        -0.54,
        "Static industry hyperedges",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        transform=ax_z1.get_xaxis_transform(),
    )
    ax_z1.text(
        M_ind + (M_time - 1) / 2.0,
        -0.54,
        "Dynamic temporal hyperedges",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        transform=ax_z1.get_xaxis_transform(),
    )

    ax_z0.text(
        0.5,
        1.05,
        "(b) Row-wise standardized attention",
        transform=ax_z0.transAxes,
        ha="center",
        va="bottom",
        fontsize=panel_fontsize,
        fontweight="bold",
    )

    z_cbar = fig.colorbar(z_images[0], cax=cax_z)
    z_cbar.ax.set_ylabel(
        "Row z-score",
        rotation=90,
        fontsize=cbar_label_fontsize,
        labelpad=10,
    )
    z_cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    fig.subplots_adjust(left=0.075, right=0.955, top=0.95, bottom=0.19)

    save_figure(fig, out_prefix + ".png")
    save_figure(fig, out_prefix + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate only the final Figure 4 raw/z-score attention visualization."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./NASDAQ/best_model.pt",
    )
    parser.add_argument("--data_path", default="../data")
    parser.add_argument("--opath", default="../data/2013-01-01")
    parser.add_argument("--market_name", default="NASDAQ")
    parser.add_argument("--hypergraph_ind_path", default="../data/industry_NASDAQ.npy")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Use >0 only for quick debugging; -1 uses the full split.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./vis_prompt_to_hyperedge",
    )
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--max_xticks", type=int, default=20)

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_xticks <= 0:
        raise ValueError("max_xticks must be positive")

    set_plot_style(font_size=15)
    ensure_dir(args.out_dir)

    checkpoint = load_checkpoint(args.checkpoint_path)
    set_seed(int(checkpoint.get("seed", args.seed)))

    model, saved_args, _ = build_model_from_checkpoint(checkpoint, args)
    _, X, _ = load_split_data(saved_args, args.split)

    print("=" * 88)
    print(f"Device: {DEVICE}")
    print(f"Split: {args.split}")
    print(f"X shape: {X.shape}")
    print(f"Samples: {len(X)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max batches: {args.max_batches}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 88)

    avg_list, M_ind, M_time, total_count = collect_average_attention(
        model=model,
        X=X,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    if len(avg_list) < 2:
        raise ValueError(
            f"Expected at least 2 PRI layers for Figure 4, got {len(avg_list)}."
        )

    layer0_raw = avg_list[0]
    layer1_raw = avg_list[1]
    layer0_z = zscore_by_row(layer0_raw)
    layer1_z = zscore_by_row(layer1_raw)

    out_prefix = os.path.join(
        args.out_dir,
        "prompt_to_hyperedge_raw_zscore_vertical_combined",
    )
    plot_raw_and_zscore_combined_heatmap(
        layer0_raw=layer0_raw,
        layer1_raw=layer1_raw,
        layer0_zscore=layer0_z,
        layer1_zscore=layer1_z,
        M_ind=M_ind,
        M_time=M_time,
        out_prefix=out_prefix,
        max_xticks=args.max_xticks,
    )

    print("\nFigure generation complete.")
    print(f"Samples used: {total_count}")
    print("Generated ONLY:")
    print(f"  {out_prefix}.png")
    print(f"  {out_prefix}.pdf")


if __name__ == "__main__":
    main()
