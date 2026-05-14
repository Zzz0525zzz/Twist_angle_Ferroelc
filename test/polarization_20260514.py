#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
polarization_twist_dump_grid.py

纯 dump 版 BaTiO3 core-shell 极化后处理，针对扭角 / moire 模型输出 Tecplot 网格和极化统计。

当前版本的核心调整：
    1) 默认直接运行：python polarization_twist_dump_grid.py
       常用参数都集中在 USER_CONFIG 中，不需要输入长命令。
    2) ALL 主输出改为 layer-by-layer multi-zone FE quadrilateral surface：
       每一个 z 层单独作为 Tecplot surface zone 输出，只连接真实节点，避免 blank 节点连到原点。
    3) padded 3D I×J×K 网格保留为可选 debug 输出，不再作为主结果。
    4) XY/XZ/YZ 截面支持 coordinate / logical 两种模式；ordered 截面默认写为 FE quadrilateral surface。
    5) 统计只使用真实 Ti cell；padding / blank 节点不进入统计。
    6) 增加边界裁剪和邻居完整性过滤，避免边界或配位不完整位置污染统计。
    7) 输出 grid report，便于判断网格识别是否可靠。
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("\n[致命错误] 本程序需要 scipy。请运行: pip install scipy --user\n")
    sys.exit(1)


# ==============================================================================
# 用户常用参数：正常情况下只需要改这里，然后直接运行本脚本
# ==============================================================================

USER_CONFIG = {
    # ---------- 输入 / 输出 ----------
    "pattern": "twist_bto_P12_longhold_Load.*.dump",
    "out_dir": "Pol_out_twist_grid",
    "out_prefix": "Pol_grid",

    # 主推荐输出：ALL_layers，每一层一个 Tecplot FEQUADRILATERAL surface zone。
    # 这样不会把 blank 节点连到原点，云图比 ordered+padding 干净很多。
    "write_all_layers": True,
    "write_sections": True,
    "write_stats": True,
    "write_grid_report": True,

    # padded 3D 网格只作为 debug / 特殊展示；默认关闭。
    # 如果你确实需要单个 I×J×K zone，可设 True。
    "write_padded_debug": False,

    # ---------- 极化近邻重建 ----------
    # 有压缩 / 拉开 / 层间距离变化时，不建议固定 reference 邻居。
    # 1 = 每帧重建，最稳但最慢；5 = 默认折中。
    "rebuild_freq": 5,
    "max_knn_dist": 8.0,

    # ---------- Ti 拓扑网格识别 ----------
    "ti_link_cutoff": 5.2,
    "z_layer_tol": 1.2,

    # 网格识别方法：
    #   "lattice" : 推荐。利用 BTO 晶格常数和 Ti-Ti 近邻距离，直接建立层内网格，速度快。
    #   "search"  : 旧方法。尝试多个角度和 I/J 形状，较慢，仅作备用。
    "grid_build_method": "lattice",

    # BTO 近似晶格常数，单位 Å。常温四方相 a 约 4 Å；这里作为 Ti-Ti 层内近邻距离的初值。
    # 如果加载很大，可适当放宽 lattice_dist_tol_A。
    "lattice_a_A": 4.0,
    "lattice_dist_tol_A": 1.2,

    # 如果你知道每层 Ti 网格尺寸，建议直接填上，例如 80, 80。
    # 不知道就设为 None，让程序根据投影坐标和晶格常数自动估计。
    "grid_ni": None,
    "grid_nj": None,

    # layer_grid_mode:
    #   "auto"        : 每层自己判断 I/J 和排序，推荐先用。
    #   "user_shape"  : 每层都使用 grid_ni/grid_nj 指定形状。
    "layer_grid_mode": "auto",

    # 若某层点数少于目标 I×J，是否允许该层内部用空节点补齐；
    # 空节点只用于 Tecplot ordered zone，不进入统计。
    "allow_layer_padding": True,

    # ordered 网格质量判据。mean_edge 太大说明排序可能跳线。
    "layer_mean_edge_max": 7.5,

    # ---------- 可选 padded 3D debug 网格 ----------
    # padded_grid_mode:
    #   "pad_to_max"  : 按点数最多的层建立 I×J×K。
    #   "pad_to_mode" : 按出现次数最多的 I/J 建立 I×J×K。
    #   "user_shape"  : 使用 grid_ni/grid_nj。
    "padded_grid_mode": "pad_to_max",

    # ---------- 截面设置 ----------
    # section_mode:
    #   "coordinate" : 按 xy_z / xz_y / yz_x 坐标找最近截面。
    #   "logical"    : 按 slice_k / slice_j / slice_i 逻辑索引取截面。
    "section_mode": "coordinate",

    # coordinate 模式：None 表示取中间截面。
    "xy_z": None,
    "xz_y": None,
    "yz_x": None,

    # logical 模式：None 表示取中间索引。
    "slice_k": None,   # XY: 第 k 层
    "slice_j": None,   # XZ: 第 j 行
    "slice_i": None,   # YZ: 第 i 列

    # 无可靠 ordered 截面时，用最近坐标带 scatter，section_tol 是半宽 Å。
    "section_tol": 1.2,

    # ---------- 边界 / 无效区域排除控制 ----------
    # 总开关。False 表示不裁剪，完全按原始全部 Ti 统计。
    "boundary_enable": False,

    # logical 裁剪：需要层网格识别成功。
    # 例如 cut_i_low=2, cut_i_high=2 表示去掉 I 方向两侧各 2 列。
    "cut_i_low": 0,
    "cut_i_high": 0,
    "cut_j_low": 0,
    "cut_j_high": 0,
    "cut_k_low": 0,
    "cut_k_high": 0,

    # coordinate 裁剪：按参考构型坐标范围裁剪，单位 Å。
    "cut_x_low_A": 0.0,
    "cut_x_high_A": 0.0,
    "cut_y_low_A": 0.0,
    "cut_y_high_A": 0.0,
    "cut_z_low_A": 0.0,
    "cut_z_high_A": 0.0,

    # 裁剪作用对象：
    #   polarization=True : 被裁剪 Ti 不计算 P，P 输出为 0。
    #   stats=True        : 被裁剪 Ti 不进入 overall / section 统计。推荐 True。
    #   output=True       : 被裁剪 Ti 在 plt 中置 0。可视化想看完整外形时可设 False。
    "apply_boundary_to_polarization": False,
    "apply_boundary_to_stats": True,
    "apply_boundary_to_output": False,

    # 邻居完整性过滤：低于该完整率的 Ti 不进入统计。
    # None 表示不启用；0.95 表示完整率低于 95% 的 cell 不进入统计。
    "min_neighbor_complete_for_stats": None,

    # ---------- 速度优化 ----------
    # 刚开始运行很慢，主要来自第一帧拓扑网格识别。开启 cache 后，第二次运行会直接读取拓扑。
    "use_topology_cache": True,
    # None 表示自动写入 out_dir/out_prefix_topology_cache.pkl。
    "topology_cache_file": None,

    # 自动网格搜索的候选数量。数值越大越稳但越慢；当前默认偏快。
    "max_angle_candidates": 4,
    "max_shape_candidates": 8,

    # True：冲突点不再做慢速逐半径寻找空位，直接保留最近点并把冲突计入报告；启动速度会明显提高。
    "fast_grid_assignment": True,

    # 只处理部分帧可显著加快全流程；1 表示每帧都处理，10 表示每 10 帧处理一帧。
    "frame_stride": 1,
    # None 表示不限制；整数表示最多处理多少帧，适合测试。
    "max_frames": None,

    # ---------- 进度输出 ----------
    # 在服务器 / Slurm / tee 日志中，覆盖式进度条经常会变成刷屏。
    # progress_mode:
    #   "none"      : 不输出逐帧进度，只输出开始和结束报告。
    #   "line"      : 每隔 progress_interval_frames 帧输出一行，推荐服务器运行。
    #   "overwrite" : 单行覆盖式进度条，适合本地终端，不适合日志文件。
    "progress_enable": True,
    "progress_mode": "line",
    "progress_interval_frames": 10,

    # ---------- 命令行覆盖 ----------
    # False：完全使用 USER_CONFIG；True：允许少量命令行覆盖。
    "allow_cli_override": False,
}


DEFAULT_CENTER_TYPE = 3
DEFAULT_SHOW_PROGRESS = True
PROGRESS_BAR_LEN = 30

CHARGE_BASE: Dict[int, float] = {
    1: 4.784,
    2: -2.948,
    3: 4.488,
    4: -1.614,
    5: 1.039,
    6: -2.609,
}
DEFAULT_CHARGE: Dict[int, float] = {t: q * 1.0 for t, q in CHARGE_BASE.items()}

DEFAULT_SHARE: Dict[int, float] = {
    1: 1.0 / 8.0,
    2: 1.0 / 8.0,
    3: 1.0,
    4: 1.0,
    5: 1.0 / 2.0,
    6: 1.0 / 2.0,
}

KNN_EXPECTED: Dict[int, int] = {
    1: 8,
    2: 8,
    4: 1,
    5: 6,
    6: 6,
}

ELEM_CHARGE = 1.602176634e-19
ANG2M = 1e-10
V_CUBE = (4.0 ** 3) * (ANG2M ** 3)


@dataclass
class DumpFrame:
    step: int
    ids: np.ndarray
    types: np.ndarray
    pos: np.ndarray
    origin: np.ndarray
    boxlen: np.ndarray
    source: str


@dataclass
class RefSystem:
    boxlen: np.ndarray
    origin: np.ndarray
    ids: np.ndarray
    types: np.ndarray
    pos: np.ndarray
    index_by_id: np.ndarray


@dataclass
class LayerGrid:
    k: int
    z_mean: float
    ni: int
    nj: int
    actual_count: int
    node_local_grid: np.ndarray   # shape=(nj, ni), local Ti index in original Ti subset, -1 means blank
    mean_edge: float
    score: float
    theta: float
    ok: bool
    padded_nodes: int
    conflict_nodes: int
    truncated_nodes: int


@dataclass
class GridTopology:
    ti_ids_ordered: np.ndarray
    ref_wrapped_ordered: np.ndarray
    ref_unwrapped_ordered: np.ndarray
    layer_reports: List[LayerGrid]
    layer_node_grids: List[np.ndarray]  # each shape=(nj,ni), values are ordered Ti index, -1 blank
    n_components: int
    avg_ti_degree: float
    logical_i: np.ndarray
    logical_j: np.ndarray
    logical_k: np.ndarray
    max_ni: int
    max_nj: int
    nk: int


@dataclass
class SectionPlan:
    name: str
    idx: np.ndarray             # may contain -1 blank nodes
    I: int
    J: int
    K: int
    uv_mode: str
    title: str
    target_coord: Optional[float]
    actual_coord: float
    ordered: bool


@dataclass
class Config:
    pattern: str
    out_dir: str
    out_prefix: str
    write_all_layers: bool
    write_sections: bool
    write_stats: bool
    write_grid_report: bool
    write_padded_debug: bool
    max_knn_dist: float
    rebuild_freq: int
    ti_link_cutoff: float
    z_layer_tol: float
    grid_ni: Optional[int]
    grid_nj: Optional[int]
    grid_build_method: str
    lattice_a_A: float
    lattice_dist_tol_A: float
    layer_grid_mode: str
    allow_layer_padding: bool
    layer_mean_edge_max: float
    padded_grid_mode: str
    section_mode: str
    xy_z: Optional[float]
    xz_y: Optional[float]
    yz_x: Optional[float]
    slice_k: Optional[int]
    slice_j: Optional[int]
    slice_i: Optional[int]
    section_tol: float
    boundary_enable: bool
    cut_i_low: int
    cut_i_high: int
    cut_j_low: int
    cut_j_high: int
    cut_k_low: int
    cut_k_high: int
    cut_x_low_A: float
    cut_x_high_A: float
    cut_y_low_A: float
    cut_y_high_A: float
    cut_z_low_A: float
    cut_z_high_A: float
    apply_boundary_to_polarization: bool
    apply_boundary_to_stats: bool
    apply_boundary_to_output: bool
    min_neighbor_complete_for_stats: Optional[float]
    use_topology_cache: bool
    topology_cache_file: Optional[str]
    max_angle_candidates: int
    max_shape_candidates: int
    fast_grid_assignment: bool
    frame_stride: int
    max_frames: Optional[int]
    progress_enable: bool
    progress_mode: str
    progress_interval_frames: int


# ==============================================================================
# 通用工具
# ==============================================================================

def _fmt_hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"


def natural_sort(files: Sequence[str]) -> List[str]:
    return sorted(files, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])


def mic_delta(delta: np.ndarray, boxlen: np.ndarray) -> np.ndarray:
    return delta - boxlen * np.rint(delta / boxlen)


def wrap_pos(pos: np.ndarray, origin: np.ndarray, boxlen: np.ndarray) -> np.ndarray:
    return origin + np.mod(pos - origin, boxlen)


def safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x.min()), float(x.max()), 0.0, 0.0)
    return (float(x.min()), float(x.max()), float(x.std(ddof=1)), float(x.var(ddof=1)))


def build_id_to_row(ids: np.ndarray) -> np.ndarray:
    lut = np.full(int(ids.max()) + 1, -1, dtype=int)
    lut[ids] = np.arange(ids.size, dtype=int)
    return lut


def split_layers(vals: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    if vals.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    order = np.argsort(vals)
    sorted_vals = vals[order]
    gap = np.diff(sorted_vals) > tol
    labels_sorted = np.insert(np.cumsum(gap), 0, 0)
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    means = np.array([float(np.mean(vals[labels == k])) for k in range(labels_sorted.max() + 1)])
    return labels, means


def actual_indices(raw_idx: np.ndarray, *masks: Optional[np.ndarray]) -> np.ndarray:
    raw = np.asarray(raw_idx, dtype=int).reshape(-1)
    out = raw[raw >= 0]
    for m in masks:
        if m is not None and out.size:
            out = out[m[out]]
    return out


# ==============================================================================
# dump 读取
# ==============================================================================

def choose_coord_columns(hdr: List[str]) -> Tuple[str, str, str, str]:
    candidates = [
        ("xu", "yu", "zu", "cart"),
        ("x", "y", "z", "cart"),
        ("xs", "ys", "zs", "scaled"),
        ("xsu", "ysu", "zsu", "scaled"),
    ]
    for cx, cy, cz, mode in candidates:
        if cx in hdr and cy in hdr and cz in hdr:
            return cx, cy, cz, mode
    raise ValueError(f"dump 的 ATOMS header 中找不到坐标列。当前 header={hdr}")


def iter_dump_frames(path: str) -> Iterator[DumpFrame]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(float(f.readline().strip()))

            line = f.readline()
            if not line.startswith("ITEM: NUMBER"):
                raise ValueError(f"{path}: TIMESTEP {step} 后没有 ITEM: NUMBER OF ATOMS。")
            n = int(float(f.readline().strip()))

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{path}: TIMESTEP {step} 后没有 ITEM: BOX BOUNDS。")

            bounds = []
            for _ in range(3):
                sp = f.readline().split()
                if len(sp) < 2:
                    raise ValueError(f"{path}: box bounds 行格式错误。")
                bounds.append((float(sp[0]), float(sp[1])))

            origin = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=float)
            boxlen = np.array([bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0]], dtype=float)

            atom_header = f.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"{path}: TIMESTEP {step} 后没有 ITEM: ATOMS。")
            hdr = atom_header.split()[2:]
            if "id" not in hdr or "type" not in hdr:
                raise ValueError(f"{path}: dump 必须包含 id 和 type 列。当前 header={hdr}")

            cx, cy, cz, coord_mode = choose_coord_columns(hdr)
            idx_id, idx_type = hdr.index("id"), hdr.index("type")
            idx_x, idx_y, idx_z = hdr.index(cx), hdr.index(cy), hdr.index(cz)

            ids = np.empty(n, dtype=int)
            types = np.empty(n, dtype=int)
            pos = np.empty((n, 3), dtype=float)

            for i in range(n):
                sp = f.readline().split()
                ids[i] = int(float(sp[idx_id]))
                types[i] = int(float(sp[idx_type]))
                pos[i, 0] = float(sp[idx_x])
                pos[i, 1] = float(sp[idx_y])
                pos[i, 2] = float(sp[idx_z])

            if coord_mode == "scaled":
                pos = origin + pos * boxlen

            yield DumpFrame(step=step, ids=ids, types=types, pos=pos, origin=origin, boxlen=boxlen, source=path)


def first_frame_from_file(path: str) -> DumpFrame:
    for frame in iter_dump_frames(path):
        return frame
    raise ValueError(f"{path} 中没有读到任何 dump frame。")


def frame_to_ref(frame: DumpFrame) -> RefSystem:
    return RefSystem(
        boxlen=frame.boxlen.copy(),
        origin=frame.origin.copy(),
        ids=frame.ids.copy(),
        types=frame.types.copy(),
        pos=frame.pos.copy(),
        index_by_id=build_id_to_row(frame.ids),
    )


# ==============================================================================
# 扭角拓扑网格识别
# ==============================================================================

def unwrap_by_ti_neighbor_graph(ti_pos: np.ndarray, origin: np.ndarray, boxlen: np.ndarray, link_cutoff: float) -> Tuple[np.ndarray, np.ndarray, int, float]:
    n = ti_pos.shape[0]
    if n == 0:
        return ti_pos.copy(), ti_pos.copy(), 0, 0.0

    r_w = wrap_pos(ti_pos, origin, boxlen)
    shifted = np.mod(r_w - origin, boxlen)
    tree = cKDTree(shifted, boxsize=boxlen)
    neigh = tree.query_ball_point(shifted, r=link_cutoff)
    degrees = np.array([max(0, len(v) - 1) for v in neigh], dtype=float)

    r_uw = r_w.copy()
    visited = np.zeros(n, dtype=bool)
    n_components = 0

    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        visited[start] = True
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in neigh[cur]:
                if nxt == cur or visited[nxt]:
                    continue
                visited[nxt] = True
                r_uw[nxt] = r_uw[cur] + mic_delta(r_w[nxt] - r_w[cur], boxlen)
                queue.append(nxt)

    box_mid = origin + 0.5 * boxlen
    centroid = np.mean(r_uw, axis=0)
    r_uw -= np.round((centroid - box_mid) / boxlen) * boxlen
    return r_uw, r_w, n_components, float(degrees.mean()) if degrees.size else 0.0


def candidate_angles_from_neighbors(c2: np.ndarray, max_candidates: int = 10) -> List[float]:
    n = c2.shape[0]
    if n < 4:
        return [0.0]
    tree = cKDTree(c2)
    k = min(7, n)
    dists, idxs = tree.query(c2, k=k)
    nearest = dists[:, 1]
    a = float(np.median(nearest[nearest > 1e-8])) if np.any(nearest > 1e-8) else 4.0

    vecs = []
    for i in range(n):
        for kk in range(1, k):
            d = dists[i, kk]
            if 0.45 * a <= d <= 1.45 * a:
                vecs.append(c2[idxs[i, kk]] - c2[i])
    if not vecs:
        return [0.0]

    vecs = np.asarray(vecs)
    angles = np.mod(np.arctan2(vecs[:, 1], vecs[:, 0]), np.pi)
    hist, edges = np.histogram(angles, bins=180, range=(0.0, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    top = np.argsort(hist)[::-1]

    out = []
    for idx in top:
        if hist[idx] == 0:
            break
        theta = float(centers[idx])
        if all(abs(math.sin(theta - old)) > 0.20 for old in out):
            out.append(theta)
        theta90 = float(np.mod(theta + 0.5 * np.pi, np.pi))
        if all(abs(math.sin(theta90 - old)) > 0.20 for old in out):
            out.append(theta90)
        if len(out) >= max_candidates:
            break

    out.extend([0.0, 0.5 * np.pi])
    unique = []
    for theta in out:
        theta = float(np.mod(theta, np.pi))
        if all(abs(math.sin(theta - old)) > 0.05 for old in unique):
            unique.append(theta)
    return unique[:max_candidates]


def factor_candidates(n: int, ni_hint: Optional[int], nj_hint: Optional[int], c2: np.ndarray, max_candidates: int = 40) -> List[Tuple[int, int]]:
    if ni_hint and nj_hint:
        return [(ni_hint, nj_hint)]

    factors = [(i, n // i) for i in range(1, int(math.sqrt(n)) + 1) if n % i == 0]
    pairs = []
    span = np.ptp(c2, axis=0)
    aspect = max(float(span[0]), 1e-9) / max(float(span[1]), 1e-9)
    for a, b in factors:
        for ni, nj in [(a, b), (b, a)]:
            ratio = ni / max(nj, 1)
            if 0.25 * aspect <= ratio <= 4.0 * aspect:
                pairs.append((ni, nj))

    if not pairs:
        for a, b in factors:
            pairs.extend([(a, b), (b, a)])

    pairs = sorted(set(pairs), key=lambda p: abs(math.log((p[0] / max(p[1], 1)) / max(aspect, 1e-9))))
    return pairs[:max_candidates]


def _grid_edge_stats(c2: np.ndarray, node_grid: np.ndarray) -> float:
    """快速计算网格已有相邻节点的平均边长。"""
    parts = []
    nj, ni = node_grid.shape
    if ni > 1:
        a = node_grid[:, :-1]
        b = node_grid[:, 1:]
        m = (a >= 0) & (b >= 0)
        if np.any(m):
            d = c2[b[m]] - c2[a[m]]
            parts.append(np.linalg.norm(d, axis=1))
    if nj > 1:
        a = node_grid[:-1, :]
        b = node_grid[1:, :]
        m = (a >= 0) & (b >= 0)
        if np.any(m):
            d = c2[b[m]] - c2[a[m]]
            parts.append(np.linalg.norm(d, axis=1))
    if not parts:
        return float("inf")
    return float(np.mean(np.concatenate(parts)))


def assign_layer_to_grid(c2: np.ndarray, theta: float, ni: int, nj: int, fast: bool = True) -> Tuple[np.ndarray, float, int, int, int]:
    """把层内点按投影坐标分配到最近 (i,j) 网格节点。

    返回：
        node_grid: shape=(nj,ni)，值为 local index，-1 表示 blank。
        score:     平均边长 + penalty。
        padded:    空节点数量。
        conflicts: 多点落入同一节点时发生的冲突数量。
        truncated: 无法放入网格的点数。
    """
    n = c2.shape[0]
    node_grid = np.full((nj, ni), -1, dtype=int)
    if n == 0 or ni <= 0 or nj <= 0:
        return node_grid, float("inf"), ni * nj, 0, 0

    e1 = np.array([math.cos(theta), math.sin(theta)])
    e2 = np.array([-math.sin(theta), math.cos(theta)])
    uv = np.column_stack((c2 @ e1, c2 @ e2))

    umin, umax = float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))
    vmin, vmax = float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))
    du = max(umax - umin, 1e-12)
    dv = max(vmax - vmin, 1e-12)

    # 归一化到网格坐标。这样缺点时，缺口更可能留在真实空间对应位置，而不是简单填在最后。
    gi_float = (uv[:, 0] - umin) / du * max(ni - 1, 1)
    gj_float = (uv[:, 1] - vmin) / dv * max(nj - 1, 1)
    gi = np.clip(np.rint(gi_float).astype(int), 0, ni - 1)
    gj = np.clip(np.rint(gj_float).astype(int), 0, nj - 1)

    center_dist = (gi_float - gi) ** 2 + (gj_float - gj) ** 2
    order = np.argsort(center_dist)

    conflicts = 0
    truncated = 0
    for p in order:
        j, i = int(gj[p]), int(gi[p])
        if node_grid[j, i] < 0:
            node_grid[j, i] = int(p)
        else:
            conflicts += 1
            if fast:
                # 快速模式：不做逐半径寻找空位。冲突点直接不进入网格，但仍参与统计。
                truncated += 1
                continue
            # 慢速模式：尝试放入周围最近空节点，避免直接丢失。
            placed = False
            max_radius = max(ni, nj)
            for radius in range(1, max_radius + 1):
                candidates = []
                for dj in range(-radius, radius + 1):
                    for di in range(-radius, radius + 1):
                        jj, ii = j + dj, i + di
                        if 0 <= jj < nj and 0 <= ii < ni and node_grid[jj, ii] < 0:
                            candidates.append((dj * dj + di * di, jj, ii))
                if candidates:
                    _, jj, ii = min(candidates)
                    node_grid[jj, ii] = int(p)
                    placed = True
                    break
            if not placed:
                truncated += 1

    padded = int(np.sum(node_grid < 0))
    mean_edge = _grid_edge_stats(c2, node_grid)
    shape_penalty = 0.02 * abs(math.log(max(ni, nj) / max(1, min(ni, nj))))
    score = mean_edge + shape_penalty + 0.10 * conflicts
    return node_grid, score, padded, conflicts, truncated


def _dominant_lattice_angle(c2: np.ndarray, lattice_a: float, tol: float) -> float:
    """用层内 Ti-Ti 近邻向量估计一个晶格主方向角。

    只寻找距离接近 BTO 晶格常数的近邻向量，不再穷举很多角度。
    """
    n = c2.shape[0]
    if n < 4:
        return 0.0
    tree = cKDTree(c2)
    k = min(7, n)
    dists, idxs = tree.query(c2, k=k)
    vecs = []
    dmin = max(0.1, lattice_a - tol)
    dmax = lattice_a + tol
    for p in range(n):
        for kk in range(1, k):
            d = dists[p, kk]
            if dmin <= d <= dmax:
                vecs.append(c2[idxs[p, kk]] - c2[p])
    if not vecs:
        return 0.0
    vecs = np.asarray(vecs)
    angles = np.mod(np.arctan2(vecs[:, 1], vecs[:, 0]), np.pi)
    hist, edges = np.histogram(angles, bins=180, range=(0.0, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[int(np.argmax(hist))])


def _estimate_axis_spacing(c2: np.ndarray, theta: float, lattice_a: float, tol: float) -> Tuple[float, float]:
    """根据近邻向量投影估计两个晶格方向的有效间距。"""
    n = c2.shape[0]
    if n < 4:
        return lattice_a, lattice_a
    e1 = np.array([math.cos(theta), math.sin(theta)])
    e2 = np.array([-math.sin(theta), math.cos(theta)])
    tree = cKDTree(c2)
    k = min(7, n)
    dists, idxs = tree.query(c2, k=k)
    s1 = []
    s2 = []
    dmin = max(0.1, lattice_a - tol)
    dmax = lattice_a + tol
    for p in range(n):
        for kk in range(1, k):
            d = dists[p, kk]
            if not (dmin <= d <= dmax):
                continue
            v = c2[idxs[p, kk]] - c2[p]
            a1 = abs(float(v @ e1))
            a2 = abs(float(v @ e2))
            if a1 > 0.55 * lattice_a and a2 < 0.65 * lattice_a:
                s1.append(a1)
            if a2 > 0.55 * lattice_a and a1 < 0.65 * lattice_a:
                s2.append(a2)
    a1 = float(np.median(s1)) if s1 else lattice_a
    a2 = float(np.median(s2)) if s2 else lattice_a
    return max(a1, 1e-6), max(a2, 1e-6)


def assign_layer_to_lattice_grid(c2: np.ndarray, conf: Config) -> Tuple[np.ndarray, float, int, int, int, int, float, float, float]:
    """用 BTO 晶格常数直接把 Ti 层分配到 (i,j) 网格。

    返回 node_grid, mean_edge, padded, conflicts, truncated, ni, nj, theta, score。
    """
    n = c2.shape[0]
    if n == 0:
        return np.full((1, 1), -1, dtype=int), float("inf"), 1, 0, 0, 1, 1, 0.0, float("inf")

    c2_centered = c2 - np.mean(c2, axis=0)
    theta = _dominant_lattice_angle(c2_centered, conf.lattice_a_A, conf.lattice_dist_tol_A)
    e1 = np.array([math.cos(theta), math.sin(theta)])
    e2 = np.array([-math.sin(theta), math.cos(theta)])
    uv = np.column_stack((c2_centered @ e1, c2_centered @ e2))
    a1, a2 = _estimate_axis_spacing(c2_centered, theta, conf.lattice_a_A, conf.lattice_dist_tol_A)

    umin, vmin = float(np.min(uv[:, 0])), float(np.min(uv[:, 1]))
    gi_float = (uv[:, 0] - umin) / a1
    gj_float = (uv[:, 1] - vmin) / a2
    gi = np.rint(gi_float).astype(int)
    gj = np.rint(gj_float).astype(int)

    # 平移到非负索引。
    gi -= int(np.min(gi))
    gj -= int(np.min(gj))

    ni_auto = int(np.max(gi)) + 1
    nj_auto = int(np.max(gj)) + 1
    ni = int(conf.grid_ni) if conf.grid_ni else ni_auto
    nj = int(conf.grid_nj) if conf.grid_nj else nj_auto
    ni = max(1, ni)
    nj = max(1, nj)

    node_grid = np.full((nj, ni), -1, dtype=int)
    frac_err = (gi_float - np.rint(gi_float)) ** 2 + (gj_float - np.rint(gj_float)) ** 2
    order = np.argsort(frac_err)
    conflicts = 0
    truncated = 0
    for p in order:
        i = int(gi[p])
        j = int(gj[p])
        if not (0 <= i < ni and 0 <= j < nj):
            truncated += 1
            continue
        if node_grid[j, i] < 0:
            node_grid[j, i] = int(p)
        else:
            conflicts += 1
            truncated += 1

    padded = int(np.sum(node_grid < 0))
    mean_edge = _grid_edge_stats(c2_centered, node_grid)
    score = mean_edge + 0.10 * conflicts + 0.02 * padded
    return node_grid, mean_edge, padded, conflicts, truncated, ni, nj, theta, score


def infer_layer_grid_lattice(c2: np.ndarray, layer_k: int, z_mean: float, conf: Config) -> LayerGrid:
    n = c2.shape[0]
    node_grid, mean_edge, padded, conflicts, truncated, ni, nj, theta, score = assign_layer_to_lattice_grid(c2, conf)
    full = padded == 0 and truncated == 0
    ok = np.isfinite(mean_edge) and mean_edge < conf.layer_mean_edge_max and (full or conf.allow_layer_padding)
    return LayerGrid(layer_k, z_mean, ni, nj, n, node_grid, mean_edge, score, theta, ok, padded, conflicts, truncated)


def infer_layer_grid(c2: np.ndarray, layer_k: int, z_mean: float, conf: Config) -> LayerGrid:
    if conf.grid_build_method.lower().strip() == "lattice":
        return infer_layer_grid_lattice(c2, layer_k, z_mean, conf)

    n = c2.shape[0]
    if n == 0:
        return LayerGrid(layer_k, z_mean, 0, 0, 0, np.full((1, 1), -1, dtype=int), float("inf"), float("inf"), 0.0, False, 1, 0, 0)

    c2_centered = c2 - np.mean(c2, axis=0)
    angles = candidate_angles_from_neighbors(c2_centered, max_candidates=max(1, conf.max_angle_candidates))
    if conf.layer_grid_mode == "user_shape" and conf.grid_ni and conf.grid_nj:
        pairs = [(conf.grid_ni, conf.grid_nj)]
    else:
        pairs = factor_candidates(n, conf.grid_ni, conf.grid_nj, c2_centered, max_candidates=max(1, conf.max_shape_candidates))

    best: Optional[LayerGrid] = None
    for theta in angles:
        for ni, nj in pairs:
            node_grid, score, padded, conflicts, truncated = assign_layer_to_grid(c2_centered, theta, ni, nj, fast=conf.fast_grid_assignment)
            mean_edge = _grid_edge_stats(c2_centered, node_grid)
            full = padded == 0 and truncated == 0
            ok = np.isfinite(mean_edge) and mean_edge < conf.layer_mean_edge_max and (full or conf.allow_layer_padding)
            candidate = LayerGrid(layer_k, z_mean, ni, nj, n, node_grid, mean_edge, score, theta, ok, padded, conflicts, truncated)
            if best is None or candidate.score < best.score:
                best = candidate

    assert best is not None
    return best


def setup_grid_topology(ref: RefSystem, conf: Config) -> GridTopology:
    ti_ids_raw = ref.ids[ref.types == DEFAULT_CENTER_TYPE]
    if ti_ids_raw.size == 0:
        raise ValueError(f"reference dump 中没有 type={DEFAULT_CENTER_TYPE} 的 Ti center。")

    ti_rows = ref.index_by_id[ti_ids_raw]
    ti_pos = ref.pos[ti_rows]
    r_uw_raw, r_w_raw, n_components, avg_degree = unwrap_by_ti_neighbor_graph(ti_pos, ref.origin, ref.boxlen, conf.ti_link_cutoff)

    z_labels, z_means = split_layers(r_uw_raw[:, 2], conf.z_layer_tol)
    layer_reports: List[LayerGrid] = []
    ordered_raw_indices: List[int] = []
    layer_node_grids: List[np.ndarray] = []

    # 建立 raw Ti index -> ordered Ti index 映射。
    raw_to_ordered: Dict[int, int] = {}
    for k in range(len(z_means)):
        raw_idx = np.where(z_labels == k)[0]
        c2 = r_uw_raw[raw_idx][:, :2]
        lg = infer_layer_grid(c2, k, float(z_means[k]), conf)
        layer_reports.append(lg)

        node_grid_ordered = np.full(lg.node_local_grid.shape, -1, dtype=int)
        for jj in range(lg.node_local_grid.shape[0]):
            for ii in range(lg.node_local_grid.shape[1]):
                local = int(lg.node_local_grid[jj, ii])
                if local >= 0:
                    raw = int(raw_idx[local])
                    if raw not in raw_to_ordered:
                        raw_to_ordered[raw] = len(ordered_raw_indices)
                        ordered_raw_indices.append(raw)
                    node_grid_ordered[jj, ii] = raw_to_ordered[raw]
        layer_node_grids.append(node_grid_ordered)

    # 加入没有被网格表示的真实 Ti，保证统计不丢点。
    for raw in range(ti_ids_raw.size):
        if raw not in raw_to_ordered:
            raw_to_ordered[raw] = len(ordered_raw_indices)
            ordered_raw_indices.append(raw)

    ordered_raw = np.asarray(ordered_raw_indices, dtype=int)
    n_ordered = ordered_raw.size
    ti_ids_ordered = ti_ids_raw[ordered_raw]
    ref_wrapped_ordered = r_w_raw[ordered_raw]
    ref_unwrapped_ordered = r_uw_raw[ordered_raw]

    logical_i = np.full(n_ordered, -1, dtype=int)
    logical_j = np.full(n_ordered, -1, dtype=int)
    logical_k = np.full(n_ordered, -1, dtype=int)
    for k, grid in enumerate(layer_node_grids):
        nj, ni = grid.shape
        for j in range(nj):
            for i in range(ni):
                idx = int(grid[j, i])
                if idx >= 0:
                    logical_i[idx] = i
                    logical_j[idx] = j
                    logical_k[idx] = k

    max_ni = max((g.shape[1] for g in layer_node_grids), default=0)
    max_nj = max((g.shape[0] for g in layer_node_grids), default=0)

    return GridTopology(
        ti_ids_ordered=ti_ids_ordered,
        ref_wrapped_ordered=ref_wrapped_ordered,
        ref_unwrapped_ordered=ref_unwrapped_ordered,
        layer_reports=layer_reports,
        layer_node_grids=layer_node_grids,
        n_components=n_components,
        avg_ti_degree=avg_degree,
        logical_i=logical_i,
        logical_j=logical_j,
        logical_k=logical_k,
        max_ni=max_ni,
        max_nj=max_nj,
        nk=len(layer_reports),
    )


def current_unwrapped_ti_positions(cpos_current: np.ndarray, valid_ti: np.ndarray, topo: GridTopology, boxlen: np.ndarray) -> np.ndarray:
    out = topo.ref_unwrapped_ordered.copy()
    disp = np.zeros_like(out)
    disp[valid_ti] = mic_delta(cpos_current[valid_ti] - topo.ref_wrapped_ordered[valid_ti], boxlen)
    out[valid_ti] += disp[valid_ti]
    return out


# ==============================================================================
# padded 3D debug grid
# ==============================================================================

def _choose_padded_shape(topo: GridTopology, conf: Config) -> Tuple[int, int]:
    if conf.padded_grid_mode == "user_shape" and conf.grid_ni and conf.grid_nj:
        return conf.grid_ni, conf.grid_nj
    if conf.padded_grid_mode == "pad_to_mode":
        shapes = [(g.shape[1], g.shape[0]) for g in topo.layer_node_grids if g.size > 0]
        if shapes:
            (ni, nj), _ = Counter(shapes).most_common(1)[0]
            return ni, nj
    # default pad_to_max
    if topo.layer_node_grids:
        gmax = max(topo.layer_node_grids, key=lambda g: g.size)
        return gmax.shape[1], gmax.shape[0]
    return 0, 0


def build_padded_debug_grid(topo: GridTopology, conf: Config) -> np.ndarray:
    ni, nj = _choose_padded_shape(topo, conf)
    nk = topo.nk
    out = np.full((nk, nj, ni), -1, dtype=int)
    for k, grid in enumerate(topo.layer_node_grids):
        gj, gi = grid.shape
        copy_j = min(nj, gj)
        copy_i = min(ni, gi)
        out[k, :copy_j, :copy_i] = grid[:copy_j, :copy_i]
    return out


# ==============================================================================
# 边界裁剪 / 统计 mask
# ==============================================================================

def build_boundary_keep_mask(topo: GridTopology, conf: Config) -> np.ndarray:
    n = topo.ti_ids_ordered.size
    keep = np.ones(n, dtype=bool)
    if not conf.boundary_enable:
        return keep

    li, lj, lk = topo.logical_i, topo.logical_j, topo.logical_k
    represented = (li >= 0) & (lj >= 0) & (lk >= 0)
    keep &= represented

    if conf.cut_i_low > 0:
        keep &= li >= conf.cut_i_low
    if conf.cut_i_high > 0:
        keep &= li < topo.max_ni - conf.cut_i_high
    if conf.cut_j_low > 0:
        keep &= lj >= conf.cut_j_low
    if conf.cut_j_high > 0:
        keep &= lj < topo.max_nj - conf.cut_j_high
    if conf.cut_k_low > 0:
        keep &= lk >= conf.cut_k_low
    if conf.cut_k_high > 0:
        keep &= lk < topo.nk - conf.cut_k_high

    coords = topo.ref_unwrapped_ordered
    for axis, low_margin, high_margin in [
        (0, conf.cut_x_low_A, conf.cut_x_high_A),
        (1, conf.cut_y_low_A, conf.cut_y_high_A),
        (2, conf.cut_z_low_A, conf.cut_z_high_A),
    ]:
        if low_margin > 0 or high_margin > 0:
            cmin = float(np.min(coords[:, axis]))
            cmax = float(np.max(coords[:, axis]))
            if low_margin > 0:
                keep &= coords[:, axis] >= cmin + low_margin
            if high_margin > 0:
                keep &= coords[:, axis] <= cmax - high_margin

    return keep


# ==============================================================================
# 截面选择
# ==============================================================================

def _clip_index(v: Optional[int], n: int) -> int:
    if n <= 0:
        return 0
    if v is None:
        return n // 2
    return max(0, min(n - 1, int(v)))


def _nearest_layer_index(topo: GridTopology, target_z: Optional[float]) -> int:
    if topo.nk == 0:
        return 0
    if target_z is None:
        return topo.nk // 2
    zvals = np.array([lg.z_mean for lg in topo.layer_reports], dtype=float)
    return int(np.argmin(np.abs(zvals - target_z)))


def _nearest_logical_j_for_y(topo: GridTopology, target_y: Optional[float]) -> int:
    if target_y is None:
        return topo.max_nj // 2
    means = []
    for j in range(topo.max_nj):
        idx = np.where(topo.logical_j == j)[0]
        means.append(float(np.mean(topo.ref_unwrapped_ordered[idx, 1])) if idx.size else np.nan)
    arr = np.asarray(means, dtype=float)
    if np.all(~np.isfinite(arr)):
        return topo.max_nj // 2
    return int(np.nanargmin(np.abs(arr - target_y)))


def _nearest_logical_i_for_x(topo: GridTopology, target_x: Optional[float]) -> int:
    if target_x is None:
        return topo.max_ni // 2
    means = []
    for i in range(topo.max_ni):
        idx = np.where(topo.logical_i == i)[0]
        means.append(float(np.mean(topo.ref_unwrapped_ordered[idx, 0])) if idx.size else np.nan)
    arr = np.asarray(means, dtype=float)
    if np.all(~np.isfinite(arr)):
        return topo.max_ni // 2
    return int(np.nanargmin(np.abs(arr - target_x)))


def _nearest_scatter_band(topo: GridTopology, axis: int, target: Optional[float], tol: float) -> Tuple[np.ndarray, float]:
    coords = topo.ref_unwrapped_ordered[:, axis]
    if coords.size == 0:
        return np.array([], dtype=int), np.nan
    if target is None:
        target = float(np.median(coords))
    center = float(coords[np.argmin(np.abs(coords - target))])
    mask = np.abs(coords - center) <= tol
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(coords - target)))], dtype=int)
    return idx, center


def build_section_plans(topo: GridTopology, conf: Config) -> Dict[str, SectionPlan]:
    plans: Dict[str, SectionPlan] = {}
    mode = conf.section_mode.lower().strip()

    if mode == "logical":
        k = _clip_index(conf.slice_k, topo.nk)
        grid = topo.layer_node_grids[k]
        idx = grid.reshape(-1)
        idx_real = actual_indices(idx)
        actual = float(np.mean(topo.ref_unwrapped_ordered[idx_real, 2])) if idx_real.size else np.nan
        plans["XY"] = SectionPlan("XY", idx, grid.shape[1], grid.shape[0], 1, "XY", "XY_LOGICAL_GRID", None, actual, True)

        j = _clip_index(conf.slice_j, topo.max_nj)
        rows = []
        for grid in topo.layer_node_grids:
            if j < grid.shape[0]:
                rows.append(grid[j, :])
            else:
                rows.append(np.full(topo.max_ni, -1, dtype=int))
        idx = np.vstack([np.pad(r, (0, max(0, topo.max_ni - r.size)), constant_values=-1)[:topo.max_ni] for r in rows]).reshape(-1)
        idx_real = actual_indices(idx)
        actual = float(np.mean(topo.ref_unwrapped_ordered[idx_real, 1])) if idx_real.size else np.nan
        plans["XZ"] = SectionPlan("XZ", idx, topo.max_ni, topo.nk, 1, "XZ", "XZ_LOGICAL_GRID", None, actual, True)

        i = _clip_index(conf.slice_i, topo.max_ni)
        cols = []
        for grid in topo.layer_node_grids:
            if i < grid.shape[1]:
                cols.append(grid[:, i])
            else:
                cols.append(np.full(topo.max_nj, -1, dtype=int))
        idx = np.vstack([np.pad(c, (0, max(0, topo.max_nj - c.size)), constant_values=-1)[:topo.max_nj] for c in cols]).reshape(-1)
        idx_real = actual_indices(idx)
        actual = float(np.mean(topo.ref_unwrapped_ordered[idx_real, 0])) if idx_real.size else np.nan
        plans["YZ"] = SectionPlan("YZ", idx, topo.max_nj, topo.nk, 1, "YZ", "YZ_LOGICAL_GRID", None, actual, True)
        return plans

    # coordinate mode
    k = _nearest_layer_index(topo, conf.xy_z)
    grid = topo.layer_node_grids[k]
    idx = grid.reshape(-1)
    idx_real = actual_indices(idx)
    actual = float(np.mean(topo.ref_unwrapped_ordered[idx_real, 2])) if idx_real.size else np.nan
    plans["XY"] = SectionPlan("XY", idx, grid.shape[1], grid.shape[0], 1, "XY", "XY_GRID", conf.xy_z, actual, True)

    j = _nearest_logical_j_for_y(topo, conf.xz_y)
    rows = []
    for grid in topo.layer_node_grids:
        if j < grid.shape[0]:
            rows.append(grid[j, :])
        else:
            rows.append(np.full(topo.max_ni, -1, dtype=int))
    idx = np.vstack([np.pad(r, (0, max(0, topo.max_ni - r.size)), constant_values=-1)[:topo.max_ni] for r in rows]).reshape(-1)
    idx_real = actual_indices(idx)
    actual = float(np.mean(topo.ref_unwrapped_ordered[idx_real, 1])) if idx_real.size else np.nan
    if idx_real.size:
        plans["XZ"] = SectionPlan("XZ", idx, topo.max_ni, topo.nk, 1, "XZ", "XZ_LOGICAL_GRID", conf.xz_y, actual, True)
    else:
        idx2, actual2 = _nearest_scatter_band(topo, axis=1, target=conf.xz_y, tol=conf.section_tol)
        plans["XZ"] = SectionPlan("XZ", idx2, idx2.size, 1, 1, "XZ", "XZ_NEAREST_SCATTER", conf.xz_y, actual2, False)

    i = _nearest_logical_i_for_x(topo, conf.yz_x)
    cols = []
    for grid in topo.layer_node_grids:
        if i < grid.shape[1]:
            cols.append(grid[:, i])
        else:
            cols.append(np.full(topo.max_nj, -1, dtype=int))
    idx = np.vstack([np.pad(c, (0, max(0, topo.max_nj - c.size)), constant_values=-1)[:topo.max_nj] for c in cols]).reshape(-1)
    idx_real = actual_indices(idx)
    actual = float(np.mean(topo.ref_unwrapped_ordered[idx_real, 0])) if idx_real.size else np.nan
    if idx_real.size:
        plans["YZ"] = SectionPlan("YZ", idx, topo.max_nj, topo.nk, 1, "YZ", "YZ_LOGICAL_GRID", conf.yz_x, actual, True)
    else:
        idx2, actual2 = _nearest_scatter_band(topo, axis=0, target=conf.yz_x, tol=conf.section_tol)
        plans["YZ"] = SectionPlan("YZ", idx2, idx2.size, 1, 1, "YZ", "YZ_NEAREST_SCATTER", conf.yz_x, actual2, False)

    return plans


# ==============================================================================
# 极化计算
# ==============================================================================

def update_knn_cache(cpos: np.ndarray, types: np.ndarray, r: np.ndarray, ids: np.ndarray, origin: np.ndarray, boxlen: np.ndarray, max_knn_dist: float) -> Dict[int, np.ndarray]:
    cached_neighbor_ids: Dict[int, np.ndarray] = {}
    c_shift = np.mod(cpos - origin, boxlen)

    for t, k_val in KNN_EXPECTED.items():
        t_mask = types == t
        t_r = r[t_mask]
        t_ids = ids[t_mask]
        if t_r.size == 0:
            cached_neighbor_ids[t] = np.full((len(cpos), k_val), -1, dtype=int)
            continue
        tree = cKDTree(np.mod(t_r - origin, boxlen), boxsize=boxlen)
        _, idxs = tree.query(c_shift, k=k_val, distance_upper_bound=max_knn_dist)
        if k_val == 1:
            idxs = idxs.reshape(-1, 1)
        valid_query = idxs < len(t_ids)
        safe_idxs = np.where(valid_query, idxs, 0)
        ids_matrix = t_ids[safe_idxs]
        ids_matrix[~valid_query] = -1
        cached_neighbor_ids[t] = ids_matrix
    return cached_neighbor_ids


def compute_polarization_tensors(
    n_ti: int,
    cpos: np.ndarray,
    valid_ti: np.ndarray,
    r: np.ndarray,
    id_to_cur_row: np.ndarray,
    boxlen: np.ndarray,
    cached_neighbor_ids: Dict[int, np.ndarray],
    charge_lut: np.ndarray,
    share_lut: np.ndarray,
    max_knn_dist: float,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float, float]:
    wq_center = float(share_lut[DEFAULT_CENTER_TYPE] * charge_lut[DEFAULT_CENTER_TYPE])
    sum_rq_pos = np.zeros((n_ti, 3), dtype=float)
    sum_rq_neg = np.zeros((n_ti, 3), dtype=float)
    total_q_pos = np.full(n_ti, wq_center, dtype=float)
    total_q_neg = np.zeros(n_ti, dtype=float)
    c_pos_expand = cpos[:, None, :]

    per_cell_sum = np.zeros(n_ti, dtype=float)
    n_types_count = 0
    type_means = []

    for t, k_val in KNN_EXPECTED.items():
        n_ids = cached_neighbor_ids[t]
        valid_id_mask = n_ids != -1
        safe_ids = np.where(valid_id_mask, n_ids, 0)
        out_of_bounds = safe_ids >= len(id_to_cur_row)
        safe_ids[out_of_bounds] = 0
        rows = id_to_cur_row[safe_ids]
        valid_row_mask = (rows != -1) & valid_id_mask & (~out_of_bounds)

        target_pos = np.zeros((n_ti, k_val, 3), dtype=float)
        target_pos[valid_row_mask] = r[rows[valid_row_mask]]
        d_vec = mic_delta(target_pos - c_pos_expand, boxlen)
        final_valid_mask = valid_row_mask & (np.linalg.norm(d_vec, axis=-1) <= max_knn_dist) & valid_ti[:, None]

        comp = np.zeros(n_ti, dtype=float)
        comp[valid_ti] = np.sum(final_valid_mask[valid_ti], axis=1) / float(k_val)
        per_cell_sum += comp
        n_types_count += 1
        if np.any(valid_ti):
            type_means.append(float(np.mean(comp[valid_ti])))

        wq = float(share_lut[t] * charge_lut[t])
        if wq > 0:
            sum_rq_pos += np.sum(d_vec * final_valid_mask[:, :, None] * wq, axis=1)
            total_q_pos += np.sum(final_valid_mask * wq, axis=1)
        elif wq < 0:
            sum_rq_neg += np.sum(d_vec * final_valid_mask[:, :, None] * abs(wq), axis=1)
            total_q_neg += np.sum(final_valid_mask * abs(wq), axis=1)

    neighbor_complete_cell = per_cell_sum / max(n_types_count, 1)

    safe_pos_div = np.where(total_q_pos > 0, total_q_pos, 1.0)
    safe_neg_div = np.where(total_q_neg > 0, total_q_neg, 1.0)
    r_pos_local = np.where(total_q_pos[:, None] > 0, sum_rq_pos / safe_pos_div[:, None], 0.0)
    r_neg_local = np.where(total_q_neg[:, None] > 0, sum_rq_neg / safe_neg_div[:, None], 0.0)

    P_cells = np.zeros((n_ti, 3), dtype=float)
    P_cells[valid_ti] = (total_q_pos[valid_ti, None] * ELEM_CHARGE) * (r_pos_local[valid_ti] - r_neg_local[valid_ti]) * ANG2M / V_CUBE
    Pmag = np.linalg.norm(P_cells, axis=1)

    qdiff_absmax = float(np.max(np.abs((total_q_pos - total_q_neg)[valid_ti]))) if np.any(valid_ti) else 0.0
    neigh_type_mean_min = float(np.min(type_means)) if type_means else 0.0
    neigh_cell_mean = float(np.mean(neighbor_complete_cell[valid_ti])) if np.any(valid_ti) else 0.0
    neigh_cell_min = float(np.min(neighbor_complete_cell[valid_ti])) if np.any(valid_ti) else 0.0
    return P_cells, Pmag, qdiff_absmax, neighbor_complete_cell, neigh_type_mean_min, neigh_cell_mean, neigh_cell_min


# ==============================================================================
# 输出
# ==============================================================================

def section_stats_row(Psec: np.ndarray) -> List[float]:
    if Psec.size == 0:
        return [0.0] + [np.nan] * 22
    Px, Py, Pz = Psec[:, 0], Psec[:, 1], Psec[:, 2]
    Pmag = np.linalg.norm(Psec, axis=1)
    return [
        float(Psec.shape[0]),
        float(Px.mean()), float(Py.mean()), float(Pz.mean()),
        *safe_stats(Px), *safe_stats(Py), *safe_stats(Pz),
        float(np.abs(Px).mean()), float(np.abs(Py).mean()), float(np.abs(Pz).mean()), float(Pmag.mean()),
        float(np.mean(Px * Px)), float(np.mean(Py * Py)), float(np.mean(Pz * Pz)),
    ]


def stat_header_core(prefix: str = "") -> str:
    names = [
        "N", "Px_mean", "Py_mean", "Pz_mean",
        "Px_min", "Px_max", "Px_std", "Px_var",
        "Py_min", "Py_max", "Py_std", "Py_var",
        "Pz_min", "Pz_max", "Pz_std", "Pz_var",
        "absPx_mean", "absPy_mean", "absPz_mean", "Pmag_mean",
        "M2x", "M2y", "M2z",
    ]
    return " ".join(f'"{n}{prefix}"' for n in names)


def gather_output_arrays(raw_idx: np.ndarray, coords: np.ndarray, P: np.ndarray, Pmag: np.ndarray, ti_ids: np.ndarray, valid_ti: np.ndarray, output_keep: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_idx, dtype=int).reshape(-1)
    n = raw.size
    coords_out = np.zeros((n, 3), dtype=float)
    P_out = np.zeros((n, 3), dtype=float)
    Pmag_out = np.zeros(n, dtype=float)
    ids_out = np.zeros(n, dtype=int)

    ok_pos = np.where(raw >= 0)[0]
    if ok_pos.size == 0:
        return coords_out, P_out, Pmag_out, ids_out

    src = raw[ok_pos]
    keep = valid_ti[src]
    if output_keep is not None:
        keep &= output_keep[src]
    ok_pos = ok_pos[keep]
    src = src[keep]

    coords_out[ok_pos] = coords[src]
    P_out[ok_pos] = P[src]
    Pmag_out[ok_pos] = Pmag[src]
    ids_out[ok_pos] = ti_ids[src]
    return coords_out, P_out, Pmag_out, ids_out


def write_zone(fh, title: str, step: int, coords: np.ndarray, P: np.ndarray, Pmag: np.ndarray, ti_ids: np.ndarray, I: int, J: int = 1, K: int = 1, uv_mode: str = "XYZ") -> None:
    """Write a Tecplot ordered/scatter zone.

    注意：这里不用字符串里的反斜杠换行，而用 chr(10)，避免复制/画布更新时把换行符错误展开，造成 EOL string literal。
    """
    n = coords.shape[0]
    if n <= 0:
        return
    if I * J * K != n:
        I, J, K = n, 1, 1

    header = 'ZONE T="{}_TS{}", I={}, J={}, K={}, DATAPACKING=POINT, SOLUTIONTIME={:.6f}'.format(
        title, step, I, J, K, float(step)
    )
    fh.write(header + chr(10))

    if uv_mode == "XY":
        U = P[:, 0]
        V = P[:, 1]
        W = P[:, 2]
    elif uv_mode == "XZ":
        U = P[:, 0]
        V = P[:, 2]
        W = P[:, 1]
    elif uv_mode == "YZ":
        U = P[:, 1]
        V = P[:, 2]
        W = P[:, 0]
    else:
        U = P[:, 0]
        V = P[:, 1]
        W = P[:, 2]

    data = np.column_stack((
        coords[:, 0], coords[:, 1], coords[:, 2],
        P[:, 0], P[:, 1], P[:, 2],
        U, V, W, Pmag, ti_ids.astype(int),
    ))
    np.savetxt(
        fh,
        data,
        fmt=["%.8f", "%.8f", "%.8f", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%d"],
        delimiter=" ",
    )


def write_fe_quad_zone(
    fh,
    title: str,
    step: int,
    raw_idx_grid: np.ndarray,
    coords: np.ndarray,
    P: np.ndarray,
    Pmag: np.ndarray,
    ti_ids: np.ndarray,
    valid_ti: np.ndarray,
    output_keep: Optional[np.ndarray],
    uv_mode: str = "XYZ",
) -> None:
    """写 Tecplot FEQUADRILATERAL zone，只保留真实节点和真实四边形单元。

    这是为了解决 ordered grid + blank node 时 Tecplot 把 0 坐标空节点连成巨大三角/金字塔的问题。
    """
    grid = np.asarray(raw_idx_grid, dtype=int)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    nj, ni = grid.shape

    node_id_grid = np.full((nj, ni), -1, dtype=int)
    node_raw: List[int] = []
    raw_to_node: Dict[int, int] = {}

    for j in range(nj):
        for i in range(ni):
            raw = int(grid[j, i])
            if raw < 0:
                continue
            if not valid_ti[raw]:
                continue
            if output_keep is not None and not output_keep[raw]:
                continue
            if raw not in raw_to_node:
                raw_to_node[raw] = len(node_raw)
                node_raw.append(raw)
            node_id_grid[j, i] = raw_to_node[raw]

    elems = []
    for j in range(nj - 1):
        for i in range(ni - 1):
            n1 = node_id_grid[j, i]
            n2 = node_id_grid[j, i + 1]
            n3 = node_id_grid[j + 1, i + 1]
            n4 = node_id_grid[j + 1, i]
            if n1 >= 0 and n2 >= 0 and n3 >= 0 and n4 >= 0:
                elems.append((n1 + 1, n2 + 1, n3 + 1, n4 + 1))

    # 若没有真实节点，不写空 zone，避免 Tecplot 读到 I=0 或空 FE zone。
    if len(node_raw) == 0:
        return
    if len(elems) == 0:
        raw = np.asarray(node_raw, dtype=int)
        write_zone(fh, title + "_SCATTER", step, coords[raw], P[raw], Pmag[raw], ti_ids[raw], raw.size, 1, 1, uv_mode)
        return

    raw = np.asarray(node_raw, dtype=int)
    coords_o = coords[raw]
    P_o = P[raw]
    Pmag_o = Pmag[raw]
    ids_o = ti_ids[raw]

    if uv_mode == "XY":
        U, V, W = P_o[:, 0], P_o[:, 1], P_o[:, 2]
    elif uv_mode == "XZ":
        U, V, W = P_o[:, 0], P_o[:, 2], P_o[:, 1]
    elif uv_mode == "YZ":
        U, V, W = P_o[:, 1], P_o[:, 2], P_o[:, 0]
    else:
        U, V, W = P_o[:, 0], P_o[:, 1], P_o[:, 2]

    header = 'ZONE T="{}_TS{}", NODES={}, ELEMENTS={}, ZONETYPE=FEQUADRILATERAL, DATAPACKING=POINT, SOLUTIONTIME={:.6f}'.format(
        title, step, raw.size, len(elems), float(step)
    )
    fh.write(header + chr(10))
    data = np.column_stack((coords_o[:, 0], coords_o[:, 1], coords_o[:, 2], P_o[:, 0], P_o[:, 1], P_o[:, 2], U, V, W, Pmag_o, ids_o.astype(int)))
    np.savetxt(
        fh,
        data,
        fmt=["%.8f", "%.8f", "%.8f", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%d"],
        delimiter=" ",
    )
    np.savetxt(fh, np.asarray(elems, dtype=int), fmt="%d", delimiter=" ")


def write_grid_report(path: str, topo: GridTopology) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "z_mean", "ni", "nj", "actual_count", "blank_nodes", "conflict_nodes", "truncated_nodes", "mean_edge", "theta", "ok"])
        for lg in topo.layer_reports:
            writer.writerow([lg.k, lg.z_mean, lg.ni, lg.nj, lg.actual_count, lg.padded_nodes, lg.conflict_nodes, lg.truncated_nodes, lg.mean_edge, lg.theta, int(lg.ok)])


def topology_cache_path(conf: Config) -> str:
    if conf.topology_cache_file:
        return conf.topology_cache_file
    return os.path.join(conf.out_dir, f"{conf.out_prefix}_topology_cache.pkl")


def topology_cache_signature(files: List[str], conf: Config) -> str:
    first = natural_sort(files)[0]
    try:
        st = os.stat(first)
        file_meta = {
            "path": os.path.abspath(first),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except OSError:
        file_meta = {"path": os.path.abspath(first), "size": -1, "mtime_ns": -1}
    cfg = {
        "version": 5,
        "center_type": DEFAULT_CENTER_TYPE,
        "ti_link_cutoff": conf.ti_link_cutoff,
        "z_layer_tol": conf.z_layer_tol,
        "grid_ni": conf.grid_ni,
        "grid_nj": conf.grid_nj,
        "grid_build_method": conf.grid_build_method,
        "lattice_a_A": conf.lattice_a_A,
        "lattice_dist_tol_A": conf.lattice_dist_tol_A,
        "layer_grid_mode": conf.layer_grid_mode,
        "allow_layer_padding": conf.allow_layer_padding,
        "layer_mean_edge_max": conf.layer_mean_edge_max,
        "max_angle_candidates": conf.max_angle_candidates,
        "max_shape_candidates": conf.max_shape_candidates,
        "fast_grid_assignment": conf.fast_grid_assignment,
        "file": file_meta,
    }
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _topology_cache_is_usable(topo: object) -> bool:
    """防止旧版本 pickle 缓存字段不完整但仍被误读。"""
    required = [
        "ti_ids_ordered", "ref_wrapped_ordered", "ref_unwrapped_ordered",
        "layer_reports", "layer_node_grids", "logical_i", "logical_j",
        "logical_k", "max_ni", "max_nj", "nk",
    ]
    if not isinstance(topo, GridTopology):
        return False
    return all(hasattr(topo, name) for name in required)


def load_topology_cache(files: List[str], conf: Config) -> Optional[GridTopology]:
    if not conf.use_topology_cache:
        return None
    path = topology_cache_path(conf)
    if not os.path.exists(path):
        return None
    sig = topology_cache_signature(files, conf)
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        topo = payload.get("topology")
        if payload.get("signature") == sig and _topology_cache_is_usable(topo):
            print(f"[cache] 已读取拓扑网格缓存: {path}")
            return topo
        if payload.get("signature") == sig:
            print("[cache] 拓扑缓存来自旧代码结构，重新构建。")
        print("[cache] 拓扑缓存参数或文件已变化，重新构建。")
    except Exception as exc:
        print(f"[cache] 读取拓扑缓存失败，重新构建: {exc}")
    return None


def save_topology_cache(files: List[str], conf: Config, topo: GridTopology) -> None:
    if not conf.use_topology_cache:
        return
    path = topology_cache_path(conf)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"signature": topology_cache_signature(files, conf), "topology": topo}
    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] 已保存拓扑网格缓存: {path}")
    except Exception as exc:
        print(f"[cache] 保存拓扑缓存失败: {exc}")


def build_or_load_topology(files: List[str], conf: Config) -> Tuple[GridTopology, RefSystem]:
    ref_frame = first_frame_from_file(files[0])
    ref = frame_to_ref(ref_frame)
    topo = load_topology_cache(files, conf)
    if topo is not None:
        return topo, ref
    t0 = time.time()
    topo = setup_grid_topology(ref, conf)
    print(f"[time] 拓扑网格构建耗时: {_fmt_hms(time.time() - t0)}")
    save_topology_cache(files, conf, topo)
    return topo, ref


def process_trajectory(files: List[str], conf: Config) -> None:
    files = natural_sort(files)
    os.makedirs(conf.out_dir, exist_ok=True)

    topo, ref = build_or_load_topology(files, conf)
    n_ti = topo.ti_ids_ordered.size
    sections = build_section_plans(topo, conf)
    boundary_keep = build_boundary_keep_mask(topo, conf)

    compute_keep = boundary_keep if (conf.boundary_enable and conf.apply_boundary_to_polarization) else np.ones(n_ti, dtype=bool)
    boundary_stats_keep = boundary_keep if (conf.boundary_enable and conf.apply_boundary_to_stats) else np.ones(n_ti, dtype=bool)
    output_keep = boundary_keep if (conf.boundary_enable and conf.apply_boundary_to_output) else None

    paths = {
        "ALL_LAYERS": os.path.join(conf.out_dir, f"{conf.out_prefix}_ALL_layers.plt"),
        "PADDED": os.path.join(conf.out_dir, f"{conf.out_prefix}_ALL_padded_debug.plt"),
        "XY": os.path.join(conf.out_dir, f"{conf.out_prefix}_XY_section.plt"),
        "XZ": os.path.join(conf.out_dir, f"{conf.out_prefix}_XZ_section.plt"),
        "YZ": os.path.join(conf.out_dir, f"{conf.out_prefix}_YZ_section.plt"),
        "STAT_ALL": os.path.join(conf.out_dir, f"{conf.out_prefix}_stats_overall.dat"),
        "STAT_SEC": os.path.join(conf.out_dir, f"{conf.out_prefix}_stats_sections.dat"),
        "GRID_REPORT": os.path.join(conf.out_dir, f"{conf.out_prefix}_grid_report.csv"),
    }

    if conf.write_grid_report:
        write_grid_report(paths["GRID_REPORT"], topo)

    print("\n================ [ 扭角拓扑网格报告 ] ================")
    print(f"reference dump: {files[0]}")
    print(f"Ti center 数量: {n_ti}")
    print(f"Ti 邻接展开 cutoff: {conf.ti_link_cutoff:.3f} Å, 连通分量: {topo.n_components}, 平均邻接度: {topo.avg_ti_degree:.2f}")
    print(f"z 层数: {topo.nk}, z-layer tol: {conf.z_layer_tol:.3f} Å")
    print(f"ALL 主输出: layer-by-layer FEQUADRILATERAL surface zones; max I={topo.max_ni}, max J={topo.max_nj}")
    for lg in topo.layer_reports[:12]:
        print(
            f" layer {lg.k:>3d}: z={lg.z_mean:>10.4f}, I={lg.ni:>5d}, J={lg.nj:>5d}, "
            f"N={lg.actual_count:>6d}, blank={lg.padded_nodes:>5d}, conflict={lg.conflict_nodes:>5d}, "
            f"mean_edge={lg.mean_edge:>7.3f}, theta={lg.theta:>6.3f}, ok={lg.ok}"
        )
    if len(topo.layer_reports) > 12:
        print(f" ... 其余 {len(topo.layer_reports)-12} 层省略")
    print("---------------- [ 截面输出计划 ] ----------------")
    for key in ["XY", "XZ", "YZ"]:
        sp = sections[key]
        mode = "ordered" if sp.ordered else "scatter"
        n_real = int(np.sum(np.asarray(sp.idx) >= 0))
        print(f" {key}: {mode:>7s}, target={sp.target_coord}, actual≈{sp.actual_coord:.4f}, N_real={n_real}, I={sp.I}, J={sp.J}, K={sp.K}, title={sp.title}")
    print("---------------- [ 边界/完整性过滤 ] ----------------")
    print(f" boundary_enable={conf.boundary_enable}, boundary_kept={int(np.sum(boundary_keep))}/{n_ti}")
    print(f" apply: polarization={conf.apply_boundary_to_polarization}, stats={conf.apply_boundary_to_stats}, output={conf.apply_boundary_to_output}")
    print(f" min_neighbor_complete_for_stats={conf.min_neighbor_complete_for_stats}")
    print("======================================================\n")

    charge_lut = np.zeros(max(DEFAULT_CHARGE.keys()) + 1, dtype=float)
    share_lut = np.zeros(max(DEFAULT_SHARE.keys()) + 1, dtype=float)
    for k, v in DEFAULT_CHARGE.items():
        charge_lut[k] = v
    for k, v in DEFAULT_SHARE.items():
        share_lut[k] = v

    open_files = []
    try:
        f_all_layers = None
        f_padded = None
        fxy = fxz = fyz = None
        fs_all = fs_sec = None

        if conf.write_all_layers:
            f_all_layers = open(paths["ALL_LAYERS"], "w", encoding="utf-8")
            open_files.append(f_all_layers)
            f_all_layers.write('VARIABLES= "X" "Y" "Z" "Px" "Py" "Pz" "U" "V" "W" "Pmag" "Ti_id"\n')

        if conf.write_padded_debug:
            f_padded = open(paths["PADDED"], "w", encoding="utf-8")
            open_files.append(f_padded)
            f_padded.write('VARIABLES= "X" "Y" "Z" "Px" "Py" "Pz" "U" "V" "W" "Pmag" "Ti_id"\n')
            padded_debug_grid = build_padded_debug_grid(topo, conf)
        else:
            padded_debug_grid = None

        if conf.write_sections:
            fxy = open(paths["XY"], "w", encoding="utf-8")
            fxz = open(paths["XZ"], "w", encoding="utf-8")
            fyz = open(paths["YZ"], "w", encoding="utf-8")
            open_files.extend([fxy, fxz, fyz])
            for fh in [fxy, fxz, fyz]:
                fh.write('VARIABLES= "X" "Y" "Z" "Px" "Py" "Pz" "U" "V" "W" "Pmag" "Ti_id"\n')

        if conf.write_stats:
            fs_all = open(paths["STAT_ALL"], "w", encoding="utf-8")
            fs_sec = open(paths["STAT_SEC"], "w", encoding="utf-8")
            open_files.extend([fs_all, fs_sec])
            fs_all.write('VARIABLES= "Step" "FrameIndex" ' + stat_header_core() + ' "qdiff_absmax" "neighbor_complete_type_mean_min" "neighbor_complete_cell_mean" "neighbor_complete_cell_min" "stats_bad_neighbor_fraction"\n')
            fs_sec.write('VARIABLES= "Step" "FrameIndex" ' + stat_header_core("_XY") + ' ' + stat_header_core("_XZ") + ' ' + stat_header_core("_YZ") + '\n')

        cached_neighbor_ids: Dict[int, np.ndarray] = {}
        frame_index = 0
        processed_frames = 0
        stop_processing = False
        t0 = time.time()
        n_files = len(files)

        for file_idx, fp in enumerate(files, start=1):
            if stop_processing:
                break
            for frame in iter_dump_frames(fp):
                frame_index += 1
                if conf.frame_stride > 1 and ((frame_index - 1) % conf.frame_stride != 0):
                    continue
                if conf.max_frames is not None and processed_frames >= conf.max_frames:
                    stop_processing = True
                    break
                processed_frames += 1
                id_to_cur_row = build_id_to_row(frame.ids)
                ti_rows = id_to_cur_row[topo.ti_ids_ordered]
                valid_ti_raw = ti_rows != -1
                valid_ti_compute = valid_ti_raw & compute_keep
                cpos = np.zeros((n_ti, 3), dtype=float)
                cpos[valid_ti_raw] = frame.pos[ti_rows[valid_ti_raw]]

                if (frame_index - 1) % conf.rebuild_freq == 0 or not cached_neighbor_ids:
                    cached_neighbor_ids = update_knn_cache(cpos, frame.types, frame.pos, frame.ids, frame.origin, frame.boxlen, conf.max_knn_dist)

                P_cells, Pmag, qdiff, neigh_cell, neigh_type_min, neigh_cell_mean, neigh_cell_min = compute_polarization_tensors(
                    n_ti, cpos, valid_ti_compute, frame.pos, id_to_cur_row, frame.boxlen,
                    cached_neighbor_ids, charge_lut, share_lut, conf.max_knn_dist,
                )

                coords_cur = current_unwrapped_ti_positions(cpos, valid_ti_raw, topo, frame.boxlen)
                valid_for_output = valid_ti_raw

                if conf.write_all_layers and f_all_layers is not None:
                    for lg, grid in zip(topo.layer_reports, topo.layer_node_grids):
                        title = f"ALL_LAYER_{lg.k:03d}"
                        write_fe_quad_zone(
                            f_all_layers, title, frame.step, grid,
                            coords_cur, P_cells, Pmag, topo.ti_ids_ordered,
                            valid_for_output, output_keep, "XYZ",
                        )

                if conf.write_padded_debug and f_padded is not None and padded_debug_grid is not None:
                    raw_idx = padded_debug_grid.reshape(-1)
                    coords_o, P_o, Pmag_o, ids_o = gather_output_arrays(raw_idx, coords_cur, P_cells, Pmag, topo.ti_ids_ordered, valid_for_output, output_keep)
                    write_zone(f_padded, "ALL_PADDED_DEBUG", frame.step, coords_o, P_o, Pmag_o, ids_o, padded_debug_grid.shape[2], padded_debug_grid.shape[1], padded_debug_grid.shape[0], "XYZ")

                if conf.write_sections and fxy is not None and fxz is not None and fyz is not None:
                    for key, fh in [("XY", fxy), ("XZ", fxz), ("YZ", fyz)]:
                        sp = sections[key]
                        if sp.ordered:
                            grid2d = np.asarray(sp.idx, dtype=int).reshape(sp.J, sp.I)
                            write_fe_quad_zone(
                                fh, sp.title, frame.step, grid2d,
                                coords_cur, P_cells, Pmag, topo.ti_ids_ordered,
                                valid_for_output, output_keep, sp.uv_mode,
                            )
                        else:
                            idx = actual_indices(sp.idx, valid_for_output, output_keep)
                            write_zone(fh, sp.title, frame.step, coords_cur[idx], P_cells[idx], Pmag[idx], topo.ti_ids_ordered[idx], idx.size, 1, 1, sp.uv_mode)

                if conf.write_stats and fs_all is not None and fs_sec is not None:
                    neighbor_stats_keep = np.ones(n_ti, dtype=bool)
                    if conf.min_neighbor_complete_for_stats is not None:
                        neighbor_stats_keep &= neigh_cell >= float(conf.min_neighbor_complete_for_stats)
                    stats_base_mask = valid_ti_raw & boundary_stats_keep
                    stats_mask = stats_base_mask & neighbor_stats_keep
                    bad_neighbor_fraction = (
                        float(np.sum(stats_base_mask & (~neighbor_stats_keep)) / max(1, np.sum(stats_base_mask)))
                        if np.any(stats_base_mask) else 0.0
                    )
                    row_all = [float(frame.step), float(frame_index)] + section_stats_row(P_cells[stats_mask]) + [qdiff, neigh_type_min, neigh_cell_mean, neigh_cell_min, bad_neighbor_fraction]
                    fs_all.write(" ".join(f"{v:.8e}" for v in row_all) + "\n")

                    row_sec = [float(frame.step), float(frame_index)]
                    for key in ["XY", "XZ", "YZ"]:
                        idx = actual_indices(sections[key].idx, valid_ti_raw, boundary_stats_keep, neighbor_stats_keep)
                        row_sec += section_stats_row(P_cells[idx])
                    fs_sec.write(" ".join(f"{v:.8e}" for v in row_sec) + "\n")

                if conf.progress_enable:
                    should_report = (
                        processed_frames == 1
                        or processed_frames % conf.progress_interval_frames == 0
                    )
                    if should_report:
                        prog = file_idx / max(1, n_files)
                        bar = "#" * int(PROGRESS_BAR_LEN * prog) + "-" * (PROGRESS_BAR_LEN - int(PROGRESS_BAR_LEN * prog))
                        elapsed = time.time() - t0
                        msg = (
                            f"[{bar}] file={file_idx}/{n_files} processed={processed_frames} frame={frame_index} "
                            f"TS={frame.step} qdiff={qdiff:.2e} neigh_mean={neigh_cell_mean:.3f} elapsed={_fmt_hms(elapsed)}"
                        )
                        if conf.progress_mode.lower().strip() == "overwrite":
                            print("\r" + msg, end="", flush=True)
                        elif conf.progress_mode.lower().strip() == "none":
                            pass
                        else:
                            print(msg, flush=True)
    finally:
        for fh in open_files:
            fh.close()

    print(f"\n[{time.strftime('%H:%M:%S')}] 完成。输出目录: {conf.out_dir}/")
    print("主可视化文件为 *_ALL_layers.plt；padded 3D debug 输出默认关闭，可在 USER_CONFIG 中开启。")


# ==============================================================================
# 参数构建
# ==============================================================================

def config_from_user_dict(d: Dict[str, object]) -> Config:
    min_comp = d.get("min_neighbor_complete_for_stats", None)
    return Config(
        pattern=str(d["pattern"]),
        out_dir=str(d["out_dir"]),
        out_prefix=str(d["out_prefix"]),
        write_all_layers=bool(d["write_all_layers"]),
        write_sections=bool(d["write_sections"]),
        write_stats=bool(d["write_stats"]),
        write_grid_report=bool(d["write_grid_report"]),
        write_padded_debug=bool(d["write_padded_debug"]),
        max_knn_dist=float(d["max_knn_dist"]),
        rebuild_freq=max(1, int(d["rebuild_freq"])),
        ti_link_cutoff=float(d["ti_link_cutoff"]),
        z_layer_tol=float(d["z_layer_tol"]),
        grid_ni=None if d["grid_ni"] is None else int(d["grid_ni"]),
        grid_nj=None if d["grid_nj"] is None else int(d["grid_nj"]),
        grid_build_method=str(d["grid_build_method"]),
        lattice_a_A=float(d["lattice_a_A"]),
        lattice_dist_tol_A=float(d["lattice_dist_tol_A"]),
        layer_grid_mode=str(d["layer_grid_mode"]),
        allow_layer_padding=bool(d["allow_layer_padding"]),
        layer_mean_edge_max=float(d["layer_mean_edge_max"]),
        padded_grid_mode=str(d["padded_grid_mode"]),
        section_mode=str(d["section_mode"]),
        xy_z=None if d["xy_z"] is None else float(d["xy_z"]),
        xz_y=None if d["xz_y"] is None else float(d["xz_y"]),
        yz_x=None if d["yz_x"] is None else float(d["yz_x"]),
        slice_k=None if d["slice_k"] is None else int(d["slice_k"]),
        slice_j=None if d["slice_j"] is None else int(d["slice_j"]),
        slice_i=None if d["slice_i"] is None else int(d["slice_i"]),
        section_tol=float(d["section_tol"]),
        boundary_enable=bool(d["boundary_enable"]),
        cut_i_low=int(d["cut_i_low"]),
        cut_i_high=int(d["cut_i_high"]),
        cut_j_low=int(d["cut_j_low"]),
        cut_j_high=int(d["cut_j_high"]),
        cut_k_low=int(d["cut_k_low"]),
        cut_k_high=int(d["cut_k_high"]),
        cut_x_low_A=float(d["cut_x_low_A"]),
        cut_x_high_A=float(d["cut_x_high_A"]),
        cut_y_low_A=float(d["cut_y_low_A"]),
        cut_y_high_A=float(d["cut_y_high_A"]),
        cut_z_low_A=float(d["cut_z_low_A"]),
        cut_z_high_A=float(d["cut_z_high_A"]),
        apply_boundary_to_polarization=bool(d["apply_boundary_to_polarization"]),
        apply_boundary_to_stats=bool(d["apply_boundary_to_stats"]),
        apply_boundary_to_output=bool(d["apply_boundary_to_output"]),
        min_neighbor_complete_for_stats=None if min_comp is None else float(min_comp),
        use_topology_cache=bool(d["use_topology_cache"]),
        topology_cache_file=None if d["topology_cache_file"] is None else str(d["topology_cache_file"]),
        max_angle_candidates=max(1, int(d["max_angle_candidates"])),
        max_shape_candidates=max(1, int(d["max_shape_candidates"])),
        fast_grid_assignment=bool(d["fast_grid_assignment"]),
        frame_stride=max(1, int(d["frame_stride"])),
        max_frames=None if d["max_frames"] is None else int(d["max_frames"]),
        progress_enable=bool(d["progress_enable"]),
        progress_mode=str(d["progress_mode"]),
        progress_interval_frames=max(1, int(d["progress_interval_frames"])),
    )


def parse_cli_overrides(base: Config) -> Config:
    parser = argparse.ArgumentParser(description="Dump-only polarization post-processing with layer-wise topology grid output.")
    parser.add_argument("--pattern", default=base.pattern)
    parser.add_argument("--out-dir", default=base.out_dir)
    parser.add_argument("--out-prefix", default=base.out_prefix)
    parser.add_argument("--grid-ni", type=int, default=base.grid_ni)
    parser.add_argument("--grid-nj", type=int, default=base.grid_nj)
    parser.add_argument("--section-mode", default=base.section_mode)
    parser.add_argument("--xy-z", type=float, default=base.xy_z)
    parser.add_argument("--xz-y", type=float, default=base.xz_y)
    parser.add_argument("--yz-x", type=float, default=base.yz_x)
    parser.add_argument("--boundary-enable", action="store_true", default=base.boundary_enable)
    args = parser.parse_args()

    base.pattern = args.pattern
    base.out_dir = args.out_dir
    base.out_prefix = args.out_prefix
    base.grid_ni = args.grid_ni
    base.grid_nj = args.grid_nj
    base.section_mode = args.section_mode
    base.xy_z = args.xy_z
    base.xz_y = args.xz_y
    base.yz_x = args.yz_x
    base.boundary_enable = args.boundary_enable
    return base


def main() -> None:
    conf = config_from_user_dict(USER_CONFIG)
    if bool(USER_CONFIG.get("allow_cli_override", False)):
        conf = parse_cli_overrides(conf)

    files = glob.glob(conf.pattern)
    if not files:
        print(f"未找到匹配的 dump 文件: {conf.pattern}")
        return
    process_trajectory(files, conf)


if __name__ == "__main__":
    main()
