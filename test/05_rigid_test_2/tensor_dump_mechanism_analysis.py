#!/usr/bin/env python3
"""
tensor_dump_mechanism_analysis.py

用于 twisted BaTiO3 bilayer 新版 LAMMPS tensor_dump 输出的单个算例分析脚本。

这个脚本不是多 case 的 Git/PLT 批处理流程，而是针对当前一个计算目录进行深入分析。
把本 .py 文件放在某一个计算文件夹中，例如：

    ./in.by
    ./tensor_dump/twist_bto_C_com_distance_Tensor.0.dump
    ./tensor_dump/twist_bto_C_com_distance_Tensor.5000.dump
    ...

直接运行：
    python tensor_dump_mechanism_analysis.py

正常使用时，所有运行参数都在下方 USER SETTINGS 区域中修改。
不需要在命令行输入 --root、--out、--max-frames、--skip-plots 等参数。

读取内容：
- tensor_dump/ 中的 LAMMPS custom dump 文件。
- 可选的 in.by，仅用于读取加载阶段步数和阶段标签。
- 可选的 tensor_dump/*global_tensor_timeseries*.dat，目前只作为后续对照文件预留。

主要功能：
1. 读取原子级 dump 字段：xu/yu/zu、c_dis[1:3]、c_stress[1:6]、q、type、id。
2. 基于参考帧建立 Ti-core centered layer grid。
3. 重构 Ti-grid displacement、小应变 proxy、rotation/vorticity proxy。
4. 依据 Ti-centered perovskite 邻居规则重新计算局域 core-shell polarization。
5. 将 LAMMPS 输出的 raw per-atom virial c_stress[1:6] 按显式 cell volume 约定转为 local stress proxy。
6. 输出 cell-level / layer-level CSV、关键帧图像和 markdown 报告。

重要物理边界：
- c_stress[1:6] 是真实 MD 输出的 raw per-atom virial，但转为 local Cauchy stress 需要体积约定。
- 这里的 strain tensor 是 Ti-grid displacement-gradient proxy，不是完整有限变形应变张量。
- polarization 是从 dump 中重新按 Ti-centered core-shell 邻居规则计算的，应与旧 polarization_20260514.py 输出对照检查，不能默认完全一致。
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# 兼容性补丁：老版本 pandas 与新版本 NumPy 可能不兼容。
# 典型报错：AttributeError: module 'numpy' has no attribute 'bool'
# 原因是 np.bool / np.int / np.float 等旧别名在 NumPy 1.24+ 中被移除，
# 但集群上的旧 pandas 仍可能调用这些别名。
# 这个补丁必须放在 import pandas 之前。
# -----------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for _np_name, _py_type in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }.items():
        if not hasattr(np, _np_name):
            setattr(np, _np_name, _py_type)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

# LAMMPS units metal 中，compute stress/atom 的输出单位是 pressure*volume。
# pressure 单位为 bar，volume 单位为 Angstrom^3。
# 因此：c_stress[1:6] / V_cell -> bar；再乘 1e-4 -> GPa。
# 之前若使用 160.21766208，会把 stress 放大约 1.6e6 倍。
BAR_TO_GPA = 1.0e-4
E_PER_A2_TO_CM2 = 16.021766208

# 当前 BTO core-shell 模型中的 atom type 约定。
TYPE_BA_CORE = 1
TYPE_BA_SHELL = 2
TYPE_TI_CORE = 3
TYPE_TI_SHELL = 4
TYPE_O_CORE = 5
TYPE_O_SHELL = 6

STRESS_COLS = ["c_stress[1]", "c_stress[2]", "c_stress[3]", "c_stress[4]", "c_stress[5]", "c_stress[6]"]
STRESS_NAMES = ["xx", "yy", "zz", "xy", "xz", "yz"]
DIS_COLS = ["c_dis[1]", "c_dis[2]", "c_dis[3]"]

MAIN_LAYERS = [4, 5, 6, 7]
# layer 1/10 是表面/加载异常层。按照当前研究约定，它们不参与机制判断、趋势表、空间重合和矢量方向统计。
EXCLUDE_LAYERS = [1, 10]
ANALYSIS_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
CLEAN_LAYERS = [4, 6, 7]
BACKGROUND_LAYER = 5

# =============================================================================
# 用户设置区 USER SETTINGS
# =============================================================================
# 正常工作流：只需要修改本区域，然后直接运行：
#     python tensor_dump_mechanism_analysis.py
#
# 默认目录结构：
#     ./tensor_dump_mechanism_analysis.py
#     ./in.by                       可选，用于读取阶段步数
#     ./tensor_dump/                 必需，新版 LAMMPS tensor dump 文件夹
#     ./Pol_out_twist_grid/          可选；当前这个旧画布版本暂不直接读取
#
# 建议把本脚本放在当前 case 文件夹中。ROOT_DIR 默认就是脚本所在目录。
ROOT_DIR = Path(__file__).resolve().parent
DUMP_DIR = "tensor_dump"
OUT_DIR = "mechanism_tensor_out"

# 快速测试开关。
# TEST_MODE = True  ：只读取少量 frame，用来检查 dump 解析和字段是否正确。
# TEST_MODE = False ：正式分析。
TEST_MODE = False
TEST_MAX_FRAMES = 3
TEST_SKIP_PLOTS = True

# 正式运行设置。
FORMAL_MAX_FRAMES = None      # None 表示读取全部 frame
FORMAL_SKIP_PLOTS = True      # True 表示不画图，只输出 CSV 和报告；当前自动图仅作诊断，默认关闭
FORMAL_SAVE_CELL_FIELDS = False  # True 表示保存每一个 frame 的 cell-level CSV；文件会很多
FORMAL_STRIDE = 1             # 每隔几个 frame 读取一次；1 表示全部读取

# 分析参数。
N_LAYERS = 10                 # Ti 层数，当前模型默认为 10 层
KNN_DERIVATIVE = 12           # local least-squares derivative 使用的近邻数

# 定性机制判断设置。
# 当前主线不是寻找单点线性公式，而是判断阶段趋势、层间大小关系、空间共定位和矢量取向。
COMPUTE_POINTWISE_METRICS = False   # 默认关闭逐点 Pearson/R2；需要诊断时可改 True
TOP_FRACTION_FOR_PATTERN = 0.20     # 空间重合：取 |dPxy| 或 |dPz| 等强响应区前 20%
HIGH_RESPONSE_FRACTION = 0.30       # 矢量方向：在 |dPxy| 前 30% 高响应区统计方向一致性
STRONG_RATIO_THRESHOLD = 2.0        # 大小关系：某层/某量超过参照的 2 倍，视为明显更强
MODERATE_RATIO_THRESHOLD = 1.3      # 大小关系：超过 1.3 倍，视为偏强

# 输出文件精简设置。
# 默认只输出最重要的少数文件；需要 debug / 深挖时再打开详细输出。
WRITE_DETAILED_TABLES = False       # False：只写核心表；True：写出所有诊断表
WRITE_SELECTED_CELL_FIELDS = False  # True：写 selected_cell_fields.csv，文件可能较大
WRITE_LAYER_SUMMARY = True          # layer_summary.csv 仍建议保留，便于追查逐层数据

# 迭代分析模式。
# summary    ：只输出核心 CSV 和报告。
# validation ：在 summary 基础上额外输出 time_path / geometry / 机制图，用于验证 P0_hold 增强是否真实。
# figure     ：重点生成干净机制图，适合检查直观性。
# full       ：输出全部诊断表和机制图。
ANALYSIS_MODE = "validation"
GENERATE_MECHANISM_FIGURES = False   # 默认关闭绘图以加快速度；需要图时改 True 或 ANALYSIS_MODE='figure'
FIGURE_DIR = "mechanism_figures"
FIGURE_DPI = 300
SPATIAL_CLEAN_LAYER_FOR_FIGURE = 6   # 空间机制图中与 layer 5 对比的 clean layer

# 物理证据验证设置。
# 目标不是建立定量本构式，而是用真实计算验证三个物理猜想：
# 1) Pz 是否跟 vertical gap / corrugation 同阶段变化；
# 2) Pxy 是否跟 shear / rotation / vorticity mechanical template 有空间共定位；
# 3) layer 5/6 若有相似模板但极化幅值不同，是否说明 layer-dependent susceptibility / background amplification。
COMPUTE_LOCAL_GAP_TEMPLATE = True
INTERP_K_FOR_GAP = 6
EVIDENCE_TOP_FRACTION = 0.20
WRITE_EVIDENCE_TABLES = True

# 加速/聚焦开关：stress proxy 目前不是主机制证据，默认不算；需要比较 sigma_shear 时再打开。
COMPUTE_STRESS_PROXY = False
COMPUTE_SIGMA_SHEAR_EVIDENCE = False

# Nature-style Pxy 证据：不只看 |dPxy| 位置，还看分量符号、方向模板和 curl/chirality 关系。
COMPUTE_PXY_NATURE_STYLE_EVIDENCE = True

# 进一步聚焦机制证据：默认关闭旧的泛化 pattern/vector 诊断，只保留与当前物理问题直接相关的证据表。
COMPUTE_LEGACY_PATTERN_VECTOR = False
COMPUTE_PXY_DIRECTION_EVIDENCE = True

# 进度打印设置。
PRINT_EVERY_FRAME = True      # 是否打印每个 frame 的处理进度
PRINT_LAYER_GRID = True       # 是否打印每一层的 Ti 数、平均 z、cell volume


@dataclass
class DumpSnapshot:
    step: int
    box: np.ndarray  # 形状为 (3, 2)，两列分别为 lo / hi
    columns: List[str]
    data: pd.DataFrame
    path: Path

    @property
    def lx(self) -> float:
        return float(self.box[0, 1] - self.box[0, 0])

    @property
    def ly(self) -> float:
        return float(self.box[1, 1] - self.box[1, 0])

    @property
    def lz(self) -> float:
        return float(self.box[2, 1] - self.box[2, 0])

    @property
    def area_xy(self) -> float:
        return self.lx * self.ly


@dataclass
class ReferenceGrid:
    ti_ids: np.ndarray
    ref_pos: np.ndarray
    layer: np.ndarray
    layer_means_z: Dict[int, float]
    layer_thickness: Dict[int, float]
    layer_cell_volume: Dict[int, float]
    ti_index_by_id: Dict[int, int]
    atom_to_ti_index_ref: Dict[int, int]


@dataclass
class StageParams:
    ramp_load: int = 30000
    hold_comp: int = 50000
    ramp_unload: int = 30000
    hold_p0: int = 50000
    sep_ramp: int = 20000
    sep_hold: int = 30000

    @property
    def load_end(self) -> int:
        return self.ramp_load

    @property
    def comp_hold_end(self) -> int:
        return self.ramp_load + self.hold_comp

    @property
    def unload_end(self) -> int:
        return self.comp_hold_end + self.ramp_unload

    @property
    def p0_hold_end(self) -> int:
        return self.unload_end + self.hold_p0

    @property
    def opening_ramp_end(self) -> int:
        return self.p0_hold_end + self.sep_ramp

    @property
    def opening_hold_end(self) -> int:
        return self.opening_ramp_end + self.sep_hold


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    norm = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"Missing required column among {candidates}. Available columns: {list(df.columns)}")
    return None


def parse_lammps_dump_file(path: Path) -> List[DumpSnapshot]:
    snaps: List[DumpSnapshot] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step_line = fh.readline()
            if not step_line:
                break
            step = int(float(step_line.strip()))

            line = fh.readline()
            if not line.startswith("ITEM: NUMBER"):
                raise RuntimeError(f"{path}: expected ITEM: NUMBER OF ATOMS after timestep {step}")
            n_atoms = int(float(fh.readline().strip()))

            line = fh.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"{path}: expected ITEM: BOX BOUNDS after atom count")
            box = []
            for _ in range(3):
                vals = fh.readline().split()
                box.append([float(vals[0]), float(vals[1])])
            box_arr = np.array(box, dtype=float)

            line = fh.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError(f"{path}: expected ITEM: ATOMS header")
            columns = line.strip().split()[2:]
            rows = []
            for _ in range(n_atoms):
                vals = fh.readline().split()
                if len(vals) != len(columns):
                    raise RuntimeError(f"{path}: atom row has {len(vals)} values but expected {len(columns)} at step {step}")
                rows.append(vals)
            arr = np.asarray(rows, dtype=float)
            df = pd.DataFrame(arr, columns=columns)
            for c in ["id", "mol", "type", "ix", "iy", "iz"]:
                if c in df.columns:
                    df[c] = df[c].astype(np.int64)
            snaps.append(DumpSnapshot(step=step, box=box_arr, columns=columns, data=df, path=path))
    return snaps


def discover_dump_files(root: Path, dump_dir: str = "tensor_dump") -> List[Path]:
    candidates = []
    d = root / dump_dir
    if d.is_dir():
        candidates.extend(sorted(d.glob("*.dump")))
    candidates.extend(sorted(root.glob("*.dump")))
    # 优先读取 Tensor dump；只有在找不到 tensor dump 时，才退回读取当前目录下其他 dump。
    tensor = [p for p in candidates if "Tensor" in p.name or "tensor" in str(p.parent).lower()]
    files = tensor if tensor else candidates
    if not files:
        raise FileNotFoundError(f"No dump files found in {root}/{dump_dir} or {root}")
    return sorted(set(files), key=lambda p: natural_step_key(p.name))


def natural_step_key(name: str) -> Tuple[int, str]:
    nums = re.findall(r"(\d+)", name)
    return (int(nums[-1]) if nums else -1, name)


def read_all_snapshots(root: Path, dump_dir: str = "tensor_dump", max_frames: Optional[int] = None, stride: int = 1) -> List[DumpSnapshot]:
    files = discover_dump_files(root, dump_dir=dump_dir)
    snaps: List[DumpSnapshot] = []
    for p in files:
        snaps.extend(parse_lammps_dump_file(p))
    snaps.sort(key=lambda s: s.step)
    # 如果多个文件中包含相同步数，只保留第一次读到的快照。
    unique: Dict[int, DumpSnapshot] = {}
    for s in snaps:
        unique.setdefault(s.step, s)
    snaps = [unique[k] for k in sorted(unique)]
    if stride > 1:
        snaps = snaps[::stride]
    if max_frames is not None and max_frames > 0:
        snaps = snaps[:max_frames]
    return snaps


def get_positions(df: pd.DataFrame, prefer_unwrapped: bool = True) -> np.ndarray:
    if prefer_unwrapped and all(c in df.columns for c in ["xu", "yu", "zu"]):
        return df[["xu", "yu", "zu"]].to_numpy(float)
    return df[["x", "y", "z"]].to_numpy(float)


def get_displacement(df_cur: pd.DataFrame, df_ref_by_id: pd.DataFrame) -> np.ndarray:
    if all(c in df_cur.columns for c in DIS_COLS):
        return df_cur[DIS_COLS].to_numpy(float)
    pos_cur = get_positions(df_cur)
    pos_ref = get_positions(df_ref_by_id)
    return pos_cur - pos_ref


def parse_inby_stage_params(root: Path) -> StageParams:
    params = StageParams()
    inby = root / "in.by"
    if not inby.exists():
        return params
    raw: Dict[str, float] = {}
    pat = re.compile(r"^\s*variable\s+(\w+)\s+equal\s+(.+?)\s*(?:#.*)?$")
    for line in inby.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if not m:
            continue
        key, expr = m.groups()
        expr = expr.strip()
        if re.search(r"[A-Za-z_]", expr):
            continue
        try:
            raw[key] = float(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            continue
    params.ramp_load = int(raw.get("RAMP_LOAD_STEPS", params.ramp_load))
    params.hold_comp = int(raw.get("HOLD_COMP_STEPS", params.hold_comp))
    params.ramp_unload = int(raw.get("RAMP_UNLOAD_STEPS", params.ramp_unload))
    params.hold_p0 = int(raw.get("HOLD_P0_STEPS", params.hold_p0))
    params.sep_ramp = int(raw.get("SEP_RAMP_STEPS", params.sep_ramp))
    params.sep_hold = int(raw.get("SEP_HOLD_STEPS", params.sep_hold))
    return params


def stage_label(step: int, p: StageParams) -> str:
    if step == 0:
        return "initial"
    if step <= p.load_end:
        return "compression_ramp"
    if step <= p.comp_hold_end:
        return "compressed_hold"
    if step <= p.unload_end:
        return "unload"
    if step <= p.p0_hold_end:
        return "P0_hold"
    if step <= p.opening_ramp_end:
        return "first_opening_ramp"
    if step <= p.opening_hold_end:
        return "first_opening_hold"
    return "post_opening"


def nearest_existing_steps(steps: Sequence[int], targets: Sequence[int]) -> Dict[str, Optional[int]]:
    """寻找最接近目标阶段的实际输出 step。

    如果目标阶段超过当前 dump 的最大 step，则返回 None，避免把 P0_hold 误标成 first_opening。
    """
    labels = ["initial", "load_end", "compressed_hold_end", "unload_end", "P0_hold_end", "first_opening_ramp_end", "first_opening_hold_end"]
    arr = np.asarray(list(steps), dtype=int)
    out: Dict[str, Optional[int]] = {}
    if arr.size == 0:
        return {lab: None for lab in labels}
    min_step = int(np.min(arr))
    max_step = int(np.max(arr))
    for lab, t in zip(labels, targets):
        tt = int(t)
        if tt < min_step or tt > max_step:
            out[lab] = None
        else:
            out[lab] = int(arr[np.argmin(np.abs(arr - tt))])
    return out


def build_reference_grid(ref: DumpSnapshot, n_layers: int = 10) -> ReferenceGrid:
    df = ref.data.copy()
    ti = df[df["type"] == TYPE_TI_CORE].copy().sort_values("id")
    if ti.empty:
        raise RuntimeError("No Ti core atoms found; cannot build Ti-centered grid.")
    ti_ids = ti["id"].to_numpy(int)
    ref_pos = get_positions(ti)

    order_z = np.argsort(ref_pos[:, 2])
    layer = np.zeros(len(ti_ids), dtype=int)
    chunks = np.array_split(order_z, n_layers)
    for li, idx in enumerate(chunks, start=1):
        layer[idx] = li
    layer_means_z = {li: float(np.mean(ref_pos[layer == li, 2])) for li in range(1, n_layers + 1)}

    means = np.array([layer_means_z[i] for i in range(1, n_layers + 1)], dtype=float)
    bounds = np.zeros(n_layers + 1)
    bounds[1:-1] = 0.5 * (means[:-1] + means[1:])
    bounds[0] = means[0] - 0.5 * (means[1] - means[0])
    bounds[-1] = means[-1] + 0.5 * (means[-1] - means[-2])
    layer_thickness = {li: float(max(bounds[li] - bounds[li - 1], 1e-6)) for li in range(1, n_layers + 1)}

    layer_cell_volume = {}
    for li in range(1, n_layers + 1):
        n_ti_layer = int(np.sum(layer == li))
        layer_cell_volume[li] = ref.area_xy * layer_thickness[li] / max(n_ti_layer, 1)

    ti_index_by_id = {int(i): k for k, i in enumerate(ti_ids)}

    atom_pos_ref = get_positions(df)
    atom_ids = df["id"].to_numpy(int)
    ti_tree = make_periodic_tree(ref_pos, np.arange(len(ti_ids)), ref.box)
    _, nearest_ti = periodic_query(ti_tree, atom_pos_ref, ref.box, k=1)
    atom_to_ti_index_ref = {int(aid): int(tidx) for aid, tidx in zip(atom_ids, nearest_ti)}

    return ReferenceGrid(
        ti_ids=ti_ids,
        ref_pos=ref_pos,
        layer=layer,
        layer_means_z=layer_means_z,
        layer_thickness=layer_thickness,
        layer_cell_volume=layer_cell_volume,
        ti_index_by_id=ti_index_by_id,
        atom_to_ti_index_ref=atom_to_ti_index_ref,
    )


@dataclass
class PeriodicTree:
    tree: cKDTree
    rep_pos: np.ndarray
    base_index: np.ndarray


def make_periodic_tree(pos: np.ndarray, base_index: np.ndarray, box: np.ndarray) -> PeriodicTree:
    lx = box[0, 1] - box[0, 0]
    ly = box[1, 1] - box[1, 0]
    shifts = []
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            shifts.append(np.array([ix * lx, iy * ly, 0.0]))
    rep = []
    idx = []
    for s in shifts:
        rep.append(pos + s[None, :])
        idx.append(base_index)
    rep_pos = np.vstack(rep)
    base = np.concatenate(idx)
    return PeriodicTree(tree=cKDTree(rep_pos), rep_pos=rep_pos, base_index=base)


def periodic_query(ptree: PeriodicTree, query_pos: np.ndarray, box: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, len(ptree.base_index))
    d, ii = ptree.tree.query(query_pos, k=k)
    if k == 1:
        return np.asarray(d), ptree.base_index[np.asarray(ii)]
    return np.asarray(d), ptree.base_index[np.asarray(ii)]


def local_lsq_gradients(x: np.ndarray, y: np.ndarray, values: np.ndarray, k: int = 12, min_neighbors: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    values = np.asarray(values, float)
    n = len(x)
    gx = np.full(n, np.nan)
    gy = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    cond = np.full(n, np.nan)
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    kk = min(max(k, min_neighbors), n)
    dists, idxs = tree.query(pts, k=kk)
    if kk == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    for i in range(n):
        idx = idxs[i]
        vals = values[idx]
        if idx.size < min_neighbors or not np.all(np.isfinite(vals)):
            continue
        dx = x[idx] - x[i]
        dy = y[idx] - y[i]
        A = np.column_stack([np.ones(idx.size), dx, dy])
        scale = np.median(dists[i][1:]) if idx.size > 1 else 1.0
        scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        w = 1.0 / (dists[i] + 0.25 * scale + 1e-12)
        Aw = A * np.sqrt(w)[:, None]
        bw = vals * np.sqrt(w)
        try:
            beta, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            pred = A @ beta
            gx[i] = beta[1]
            gy[i] = beta[2]
            resid[i] = float(np.sqrt(np.mean((pred - vals) ** 2)))
            cond[i] = float(np.linalg.cond(Aw.T @ Aw))
        except Exception:
            pass
    return gx, gy, resid, cond


def aligned_current_ti(snapshot: DumpSnapshot, ref_grid: ReferenceGrid) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = snapshot.data.set_index("id", drop=False)
    ti_cur = df.loc[ref_grid.ti_ids].copy()
    cur_pos = get_positions(ti_cur)
    ref_pos = ref_grid.ref_pos
    if all(c in ti_cur.columns for c in DIS_COLS):
        u = ti_cur[DIS_COLS].to_numpy(float)
    else:
        u = cur_pos - ref_pos
    return ti_cur.reset_index(drop=True), cur_pos, u


def compute_strain_fields(ref_grid: ReferenceGrid, u: np.ndarray, k: int = 12) -> Dict[str, np.ndarray]:
    x = ref_grid.ref_pos[:, 0]
    y = ref_grid.ref_pos[:, 1]
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    dux_dx, dux_dy, rux, cux = local_lsq_gradients(x, y, ux, k=k)
    duy_dx, duy_dy, ruy, cuy = local_lsq_gradients(x, y, uy, k=k)
    duz_dx, duz_dy, ruz, cuz = local_lsq_gradients(x, y, uz, k=k)
    eps_xx = dux_dx
    eps_yy = duy_dy
    eps_xy = 0.5 * (duy_dx + dux_dy)
    omega_z = 0.5 * (duy_dx - dux_dy)
    curl_u_z = duy_dx - dux_dy
    depsxy_dx, depsxy_dy, rexy, cexy = local_lsq_gradients(x, y, eps_xy, k=k)
    return {
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_xy": eps_xy,
        "omega_z": omega_z,
        "curl_u_z": curl_u_z,
        "duz_dx": duz_dx,
        "duz_dy": duz_dy,
        "grad_uz_mag": np.sqrt(duz_dx ** 2 + duz_dy ** 2),
        "depsxy_dx": depsxy_dx,
        "depsxy_dy": depsxy_dy,
        "grad_resid_median": np.full_like(eps_xy, np.nanmedian(np.concatenate([rux[np.isfinite(rux)], ruy[np.isfinite(ruy)], ruz[np.isfinite(ruz)]]))),
        "grad_cond_median": np.full_like(eps_xy, np.nanmedian(np.concatenate([cux[np.isfinite(cux)], cuy[np.isfinite(cuy)], cuz[np.isfinite(cuz)]]))),
    }


def compute_local_stress(snapshot: DumpSnapshot, ref_grid: ReferenceGrid) -> Dict[str, np.ndarray]:
    df = snapshot.data
    n_ti = len(ref_grid.ti_ids)
    virial_sum = np.zeros((n_ti, 6), dtype=float)
    if not all(c in df.columns for c in STRESS_COLS):
        return {f"sigma_{name}_GPa": np.full(n_ti, np.nan) for name in STRESS_NAMES}
    atom_ids = df["id"].to_numpy(int)
    stress = df[STRESS_COLS].to_numpy(float)
    for row_i, aid in enumerate(atom_ids):
        ti_idx = ref_grid.atom_to_ti_index_ref.get(int(aid), None)
        if ti_idx is not None:
            virial_sum[ti_idx] += stress[row_i]
    sigma = np.zeros_like(virial_sum)
    for i in range(n_ti):
        li = int(ref_grid.layer[i])
        vol = ref_grid.layer_cell_volume[li]
        sigma[i] = -virial_sum[i] / vol * BAR_TO_GPA
    return {f"sigma_{name}_GPa": sigma[:, j] for j, name in enumerate(STRESS_NAMES)}


def species_periodic_tree(snapshot: DumpSnapshot, atom_type: int) -> Optional[PeriodicTree]:
    df = snapshot.data
    sub = df[df["type"] == atom_type]
    if sub.empty:
        return None
    pos = get_positions(sub)
    base = sub.index.to_numpy(int)
    return make_periodic_tree(pos, base, snapshot.box)


def compute_polarization_perovskite(snapshot: DumpSnapshot, ref_grid: ReferenceGrid, ti_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ti-centered core-shell polarization using a perovskite-neighbor weighting rule:
    - 1 nearest Ti core + 1 nearest Ti shell, weight 1
    - 8 nearest Ba core + 8 nearest Ba shell, weight 1/8
    - 6 nearest O core + 6 nearest O shell, weight 1/2

    这是对旧 Ti-centered local-cell 极化计算思路的实用重构。
    后续仍应与旧 polarization_20260514.py 输出进行对照，不能直接认为完全一致。
    """
    df = snapshot.data.reset_index(drop=True)
    pos_all = get_positions(df)
    q_all = df["q"].to_numpy(float)
    n_ti = len(ti_pos)
    dip = np.zeros((n_ti, 3), dtype=float)
    qsum = np.zeros(n_ti, dtype=float)

    # atom_type、邻居数量、权重
    rules = [
        (TYPE_TI_CORE, 1, 1.0),
        (TYPE_TI_SHELL, 1, 1.0),
        (TYPE_BA_CORE, 8, 1.0 / 8.0),
        (TYPE_BA_SHELL, 8, 1.0 / 8.0),
        (TYPE_O_CORE, 6, 1.0 / 2.0),
        (TYPE_O_SHELL, 6, 1.0 / 2.0),
    ]

    for atom_type, k, weight in rules:
        ptree = species_periodic_tree(snapshot, atom_type)
        if ptree is None:
            warnings.warn(f"No atoms of type {atom_type} found for polarization reconstruction.")
            continue
        kk = min(k, len(ptree.base_index))
        d, idx_rep = ptree.tree.query(ti_pos, k=kk)
        if kk == 1:
            idx_rep = idx_rep[:, None]
        for i in range(n_ti):
            for rep_i in np.ravel(idx_rep[i]):
                base_i = int(ptree.base_index[rep_i])
                dr = ptree.rep_pos[rep_i] - ti_pos[i]
                q = q_all[base_i]
                dip[i] += weight * q * dr
                qsum[i] += weight * q

    P_eA_A3 = np.zeros_like(dip)
    for i in range(n_ti):
        vol = ref_grid.layer_cell_volume[int(ref_grid.layer[i])]
        P_eA_A3[i] = dip[i] / vol
    P_C_m2 = P_eA_A3 * E_PER_A2_TO_CM2
    return {
        "Px_C_m2": P_C_m2[:, 0],
        "Py_C_m2": P_C_m2[:, 1],
        "Pz_C_m2": P_C_m2[:, 2],
        "Pmag_C_m2": np.linalg.norm(P_C_m2, axis=1),
        "dipole_x_eA": dip[:, 0],
        "dipole_y_eA": dip[:, 1],
        "dipole_z_eA": dip[:, 2],
        "cell_qsum_e": qsum,
    }


def curl_div_on_layer(ref_grid: ReferenceGrid, px: np.ndarray, py: np.ndarray, k: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    x = ref_grid.ref_pos[:, 0]
    y = ref_grid.ref_pos[:, 1]
    dpx_dx, dpx_dy, _, _ = local_lsq_gradients(x, y, px, k=k)
    dpy_dx, dpy_dy, _, _ = local_lsq_gradients(x, y, py, k=k)
    curl = dpy_dx - dpx_dy
    div = dpx_dx + dpy_dy
    return curl, div


def idw_interpolate_xy(src_x: np.ndarray, src_y: np.ndarray, src_val: np.ndarray, qx: np.ndarray, qy: np.ndarray, k: int = 6) -> np.ndarray:
    """在 x-y 平面上做简单 IDW 插值，用于构造 layer-to-layer local gap proxy。"""
    src_x = np.asarray(src_x, float)
    src_y = np.asarray(src_y, float)
    src_val = np.asarray(src_val, float)
    qx = np.asarray(qx, float)
    qy = np.asarray(qy, float)
    out = np.full(qx.shape, np.nan, dtype=float)
    mask = np.isfinite(src_x) & np.isfinite(src_y) & np.isfinite(src_val)
    if np.count_nonzero(mask) < 3:
        return out
    pts = np.column_stack([src_x[mask], src_y[mask]])
    vals = src_val[mask]
    tree = cKDTree(pts)
    kk = min(max(1, k), len(vals))
    d, idx = tree.query(np.column_stack([qx, qy]), k=kk)
    if kk == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(d, 1e-10) ** 2
    exact = d[:, 0] < 1e-10
    interp = np.sum(w * vals[idx], axis=1) / np.sum(w, axis=1)
    if np.any(exact):
        interp[exact] = vals[idx[exact, 0]]
    out[:] = interp
    return out


def add_local_gap_template(cell_df: pd.DataFrame, k_interp: int = 6, k_grad: int = 12) -> pd.DataFrame:
    """为每个 Ti cell 增加 h_upper / h_lower 及其梯度 proxy。

    h_upper = z_upper_interpolated(x,y) - z_current(x,y)
    h_lower = z_current(x,y) - z_lower_interpolated(x,y)

    这里的 h 不是最终严格 buried-interface local gap，但比 layer-averaged gap_to_upper 更接近局域 vertical geometry template。
    """
    cur = cell_df.copy()
    n = len(cur)
    for col in ["h_upper", "h_lower", "dh_upper", "dh_lower", "grad_h_upper_x", "grad_h_upper_y", "grad_h_upper_mag", "grad_h_lower_x", "grad_h_lower_y", "grad_h_lower_mag"]:
        if col not in cur.columns:
            cur[col] = np.nan
    for li in sorted(cur["layer"].unique()):
        li = int(li)
        idx = cur.index[cur["layer"] == li].to_numpy()
        if idx.size < 4:
            continue
        g = cur.loc[idx]
        qx = g["x_ref"].to_numpy(float)
        qy = g["y_ref"].to_numpy(float)
        zc = g["z"].to_numpy(float)
        upper = cur[cur["layer"] == li + 1]
        lower = cur[cur["layer"] == li - 1]
        if not upper.empty:
            zu = idw_interpolate_xy(upper["x_ref"].to_numpy(float), upper["y_ref"].to_numpy(float), upper["z"].to_numpy(float), qx, qy, k=k_interp)
            h = zu - zc
            cur.loc[idx, "h_upper"] = h
            ghx, ghy, _, _ = local_lsq_gradients(qx, qy, h, k=k_grad)
            cur.loc[idx, "grad_h_upper_x"] = ghx
            cur.loc[idx, "grad_h_upper_y"] = ghy
            cur.loc[idx, "grad_h_upper_mag"] = np.sqrt(ghx ** 2 + ghy ** 2)
        if not lower.empty:
            zl = idw_interpolate_xy(lower["x_ref"].to_numpy(float), lower["y_ref"].to_numpy(float), lower["z"].to_numpy(float), qx, qy, k=k_interp)
            h = zc - zl
            cur.loc[idx, "h_lower"] = h
            ghx, ghy, _, _ = local_lsq_gradients(qx, qy, h, k=k_grad)
            cur.loc[idx, "grad_h_lower_x"] = ghx
            cur.loc[idx, "grad_h_lower_y"] = ghy
            cur.loc[idx, "grad_h_lower_mag"] = np.sqrt(ghx ** 2 + ghy ** 2)
    return cur


def add_delta_gap_template(cell_df: pd.DataFrame, ref_cell_df: pd.DataFrame) -> pd.DataFrame:
    """相对参考帧计算 delta h，用于判断 Pz 是否响应 vertical geometry reorganization。"""
    cur = cell_df.copy()
    ref_cols = [c for c in ["h_upper", "h_lower", "grad_h_upper_mag", "grad_h_lower_mag"] if c in ref_cell_df.columns]
    if not ref_cols:
        return cur
    ref = ref_cell_df.set_index("Ti_id")[ref_cols]
    aligned = ref.loc[cur["Ti_id"].to_numpy(int)]
    if "h_upper" in ref_cols:
        cur["dh_upper"] = cur["h_upper"].to_numpy(float) - aligned["h_upper"].to_numpy(float)
    if "h_lower" in ref_cols:
        cur["dh_lower"] = cur["h_lower"].to_numpy(float) - aligned["h_lower"].to_numpy(float)
    if "grad_h_upper_mag" in ref_cols:
        cur["dgrad_h_upper_mag"] = cur["grad_h_upper_mag"].to_numpy(float) - aligned["grad_h_upper_mag"].to_numpy(float)
    if "grad_h_lower_mag" in ref_cols:
        cur["dgrad_h_lower_mag"] = cur["grad_h_lower_mag"].to_numpy(float) - aligned["grad_h_lower_mag"].to_numpy(float)
    cur["abs_dh_upper"] = np.abs(cur["dh_upper"].to_numpy(float)) if "dh_upper" in cur.columns else np.nan
    cur["abs_dh_lower"] = np.abs(cur["dh_lower"].to_numpy(float)) if "dh_lower" in cur.columns else np.nan
    if "abs_dh_upper" in cur.columns and "abs_dh_lower" in cur.columns:
        cur["abs_dh_best"] = np.nanmax(np.column_stack([cur["abs_dh_upper"].to_numpy(float), cur["abs_dh_lower"].to_numpy(float)]), axis=1)
    if "grad_h_upper_mag" in cur.columns and "grad_h_lower_mag" in cur.columns:
        cur["grad_h_best_mag"] = np.nanmax(np.column_stack([cur["grad_h_upper_mag"].to_numpy(float), cur["grad_h_lower_mag"].to_numpy(float)]), axis=1)
    # nearest-interface 约定：下半部分更看 upper，上半部分更看 lower；layer 5/6 附近尤其用于检查上下不对称。
    layer_arr = cur["layer"].to_numpy(int)
    cur["abs_dh_nearest"] = np.where(layer_arr <= BACKGROUND_LAYER, cur.get("abs_dh_upper", np.nan), cur.get("abs_dh_lower", np.nan))
    cur["grad_h_nearest_mag"] = np.where(layer_arr <= BACKGROUND_LAYER, cur.get("grad_h_upper_mag", np.nan), cur.get("grad_h_lower_mag", np.nan))
    if "grad_h_upper_x" in cur.columns and "grad_h_lower_x" in cur.columns:
        cur["grad_h_nearest_x"] = np.where(layer_arr <= BACKGROUND_LAYER, cur["grad_h_upper_x"].to_numpy(float), cur["grad_h_lower_x"].to_numpy(float))
        cur["grad_h_nearest_y"] = np.where(layer_arr <= BACKGROUND_LAYER, cur["grad_h_upper_y"].to_numpy(float), cur["grad_h_lower_y"].to_numpy(float))
    return cur


def build_cell_dataframe(snapshot: DumpSnapshot, ref: DumpSnapshot, ref_grid: ReferenceGrid, k: int = 12) -> pd.DataFrame:
    ti_cur, ti_pos, u = aligned_current_ti(snapshot, ref_grid)
    fields = compute_strain_fields(ref_grid, u, k=k)
    if COMPUTE_STRESS_PROXY:
        stress = compute_local_stress(snapshot, ref_grid)
    else:
        n_ti = len(ref_grid.ti_ids)
        stress = {f"sigma_{name}_GPa": np.full(n_ti, np.nan) for name in STRESS_NAMES}
    pol = compute_polarization_perovskite(snapshot, ref_grid, ti_pos)

    # 这里只写入绝对极化 P；相对参考帧的 deltaP 会在后续函数中统一计算。
    out = pd.DataFrame({
        "step": snapshot.step,
        "Ti_id": ref_grid.ti_ids,
        "layer": ref_grid.layer,
        "x_ref": ref_grid.ref_pos[:, 0],
        "y_ref": ref_grid.ref_pos[:, 1],
        "z_ref": ref_grid.ref_pos[:, 2],
        "x": ti_pos[:, 0],
        "y": ti_pos[:, 1],
        "z": ti_pos[:, 2],
        "ux": u[:, 0],
        "uy": u[:, 1],
        "uz": u[:, 2],
    })
    for d in [fields, stress, pol]:
        for key, val in d.items():
            out[key] = val
    out["cell_volume_A3"] = [ref_grid.layer_cell_volume[int(li)] for li in ref_grid.layer]
    return out


def add_delta_p_and_topology(cell_df: pd.DataFrame, ref_cell_df: pd.DataFrame, ref_grid: ReferenceGrid, k: int = 12) -> pd.DataFrame:
    ref_p = ref_cell_df.set_index("Ti_id")[["Px_C_m2", "Py_C_m2", "Pz_C_m2"]]
    cur = cell_df.copy()
    aligned = ref_p.loc[cur["Ti_id"].to_numpy(int)]
    cur["dPx_C_m2"] = cur["Px_C_m2"].to_numpy(float) - aligned["Px_C_m2"].to_numpy(float)
    cur["dPy_C_m2"] = cur["Py_C_m2"].to_numpy(float) - aligned["Py_C_m2"].to_numpy(float)
    cur["dPz_C_m2"] = cur["Pz_C_m2"].to_numpy(float) - aligned["Pz_C_m2"].to_numpy(float)
    cur["Pxy_mag_C_m2"] = np.sqrt(cur["Px_C_m2"] ** 2 + cur["Py_C_m2"] ** 2)
    cur["dPxy_mag_C_m2"] = np.sqrt(cur["dPx_C_m2"] ** 2 + cur["dPy_C_m2"] ** 2)
    curl_p, div_p = curl_div_on_layer(ref_grid, cur["Px_C_m2"].to_numpy(float), cur["Py_C_m2"].to_numpy(float), k=k)
    curl_dp, div_dp = curl_div_on_layer(ref_grid, cur["dPx_C_m2"].to_numpy(float), cur["dPy_C_m2"].to_numpy(float), k=k)
    cur["curl_Pxy"] = curl_p
    cur["div_Pxy"] = div_p
    cur["curl_dPxy"] = curl_dp
    cur["div_dPxy"] = div_dp
    return cur


def rms_array(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x ** 2)))


def summarize_layer(cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (step, layer), g in cell_df.groupby(["step", "layer"]):
        if int(layer) in EXCLUDE_LAYERS:
            continue
        row = {"step": int(step), "layer": int(layer), "n_Ti": len(g)}
        # 几何信息：z_mean / z_std 用于后续检查 vertical geometry、layer spacing 和 corrugation。
        row["z_mean"] = float(np.nanmean(g["z"]))
        row["z_std"] = float(np.nanstd(g["z"]))
        row["z_ref_mean"] = float(np.nanmean(g["z_ref"]))
        for col in ["ux", "uy", "uz", "eps_xx", "eps_yy", "eps_xy", "omega_z", "curl_u_z", "duz_dx", "duz_dy", "grad_uz_mag", "depsxy_dx", "depsxy_dy"]:
            row[f"{col}_mean"] = float(np.nanmean(g[col]))
            row[f"{col}_rms"] = rms_array(g[col].to_numpy(float))
        for col in ["Px_C_m2", "Py_C_m2", "Pz_C_m2", "Pxy_mag_C_m2", "dPx_C_m2", "dPy_C_m2", "dPz_C_m2", "dPxy_mag_C_m2", "curl_Pxy", "curl_dPxy", "div_Pxy", "div_dPxy", "cell_qsum_e"]:
            row[f"{col}_mean"] = float(np.nanmean(g[col]))
            row[f"{col}_rms"] = rms_array(g[col].to_numpy(float))
        for col in ["h_upper", "h_lower", "dh_upper", "dh_lower", "abs_dh_upper", "abs_dh_lower", "abs_dh_best", "abs_dh_nearest", "grad_h_upper_mag", "grad_h_lower_mag", "grad_h_best_mag", "grad_h_nearest_mag", "dgrad_h_upper_mag", "dgrad_h_lower_mag"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(np.nanmean(g[col]))
                row[f"{col}_rms"] = rms_array(g[col].to_numpy(float))
        for col in [f"sigma_{name}_GPa" for name in STRESS_NAMES]:
            row[f"{col}_mean"] = float(np.nanmean(g[col]))
            row[f"{col}_rms"] = rms_array(g[col].to_numpy(float))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["step", "layer"])


def summarize_frame(layer_df: pd.DataFrame, params: StageParams) -> pd.DataFrame:
    rows = []
    for step, g in layer_df.groupby("step"):
        row = {"step": int(step), "stage": stage_label(int(step), params)}
        # layer_df 已经排除了 layer 1/10，因此 analysis = layers 2-9。
        analysis = g[g["layer"].isin(ANALYSIS_LAYERS)]
        clean = g[g["layer"].isin(CLEAN_LAYERS)]
        layer5 = g[g["layer"] == BACKGROUND_LAYER]
        medium = g[g["layer"].isin([3, 8])]
        weak = g[g["layer"].isin([2, 9])]
        for prefix, sub in [("analysis", analysis), ("clean", clean), ("layer5", layer5), ("medium", medium), ("weak", weak)]:
            if sub.empty:
                row[f"{prefix}_dPxy_rms_mean"] = np.nan
                row[f"{prefix}_dPz_rms_mean"] = np.nan
                row[f"{prefix}_Pxy_rms_mean"] = np.nan
                row[f"{prefix}_Pz_rms_mean"] = np.nan
                row[f"{prefix}_grad_uz_rms_mean"] = np.nan
                row[f"{prefix}_grad_h_upper_rms_mean"] = np.nan
                row[f"{prefix}_grad_h_lower_rms_mean"] = np.nan
                row[f"{prefix}_grad_h_best_rms_mean"] = np.nan
                row[f"{prefix}_grad_h_nearest_rms_mean"] = np.nan
                row[f"{prefix}_abs_dh_upper_rms_mean"] = np.nan
                row[f"{prefix}_abs_dh_lower_rms_mean"] = np.nan
                row[f"{prefix}_abs_dh_best_rms_mean"] = np.nan
                row[f"{prefix}_abs_dh_nearest_rms_mean"] = np.nan
                row[f"{prefix}_uz_rms_mean"] = np.nan
                row[f"{prefix}_epsxy_rms_mean"] = np.nan
                row[f"{prefix}_omega_rms_mean"] = np.nan
                row[f"{prefix}_szz_mean"] = np.nan
                continue
            row[f"{prefix}_dPxy_rms_mean"] = float(np.nanmean(sub["dPxy_mag_C_m2_rms"]))
            row[f"{prefix}_dPz_rms_mean"] = float(np.nanmean(sub["dPz_C_m2_rms"]))
            row[f"{prefix}_Pxy_rms_mean"] = float(np.nanmean(sub["Pxy_mag_C_m2_rms"]))
            row[f"{prefix}_Pz_rms_mean"] = float(np.nanmean(sub["Pz_C_m2_rms"]))
            row[f"{prefix}_grad_uz_rms_mean"] = float(np.nanmean(sub["grad_uz_mag_rms"]))
            row[f"{prefix}_grad_h_upper_rms_mean"] = float(np.nanmean(sub["grad_h_upper_mag_rms"])) if "grad_h_upper_mag_rms" in sub.columns else np.nan
            row[f"{prefix}_grad_h_lower_rms_mean"] = float(np.nanmean(sub["grad_h_lower_mag_rms"])) if "grad_h_lower_mag_rms" in sub.columns else np.nan
            row[f"{prefix}_grad_h_best_rms_mean"] = float(np.nanmean(sub["grad_h_best_mag_rms"])) if "grad_h_best_mag_rms" in sub.columns else np.nan
            row[f"{prefix}_grad_h_nearest_rms_mean"] = float(np.nanmean(sub["grad_h_nearest_mag_rms"])) if "grad_h_nearest_mag_rms" in sub.columns else np.nan
            row[f"{prefix}_abs_dh_upper_rms_mean"] = float(np.nanmean(sub["abs_dh_upper_rms"])) if "abs_dh_upper_rms" in sub.columns else np.nan
            row[f"{prefix}_abs_dh_lower_rms_mean"] = float(np.nanmean(sub["abs_dh_lower_rms"])) if "abs_dh_lower_rms" in sub.columns else np.nan
            row[f"{prefix}_abs_dh_best_rms_mean"] = float(np.nanmean(sub["abs_dh_best_rms"])) if "abs_dh_best_rms" in sub.columns else np.nan
            row[f"{prefix}_abs_dh_nearest_rms_mean"] = float(np.nanmean(sub["abs_dh_nearest_rms"])) if "abs_dh_nearest_rms" in sub.columns else np.nan
            row[f"{prefix}_uz_rms_mean"] = float(np.nanmean(sub["uz_rms"]))
            row[f"{prefix}_epsxy_rms_mean"] = float(np.nanmean(sub["eps_xy_rms"]))
            row[f"{prefix}_omega_rms_mean"] = float(np.nanmean(sub["omega_z_rms"]))
            row[f"{prefix}_szz_mean"] = float(np.nanmean(sub["sigma_zz_GPa_mean"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step")


def fit_corr(x: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 4:
        return int(np.count_nonzero(m)), float("nan"), float("nan")
    xx, yy = x[m], y[m]
    corr = float(np.corrcoef(xx, yy)[0, 1])
    A = np.column_stack([xx, np.ones_like(xx)])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    pred = A @ beta
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return int(len(xx)), corr, r2


def model_metrics(cell_df: pd.DataFrame, selected_steps: Sequence[int]) -> pd.DataFrame:
    rows = []
    if not COMPUTE_POINTWISE_METRICS:
        return pd.DataFrame(columns=["step", "layer", "model", "n", "corr", "r2"])
    sel = cell_df[cell_df["step"].isin(selected_steps)]
    for (step, layer), g in sel.groupby(["step", "layer"]):
        if int(layer) in EXCLUDE_LAYERS:
            continue
        # 这里只做 proxy 相关性检查，不要把这些结果当作本构拟合或严格证明。
        metrics = [
            ("grad_uz_to_dPxy", g["grad_uz_mag"], g["dPxy_mag_C_m2"]),
            ("abs_dPz_to_dPxy", np.abs(g["dPz_C_m2"]), g["dPxy_mag_C_m2"]),
            ("omega_to_curl_dPxy", g["omega_z"], g["curl_dPxy"]),
            ("curl_u_to_curl_dPxy", g["curl_u_z"], g["curl_dPxy"]),
            ("strict_dPy_from_depsxy_dx", g["depsxy_dx"], g["dPy_C_m2"]),
            ("strict_dPx_from_depsxy_dy", g["depsxy_dy"], g["dPx_C_m2"]),
        ]
        for name, x, y in metrics:
            n, corr, r2 = fit_corr(x.to_numpy(float), y.to_numpy(float))
            rows.append({"step": int(step), "layer": int(layer), "model": name, "n": n, "corr": corr, "r2": r2})
    return pd.DataFrame(rows).sort_values(["step", "layer", "model"])


def qualitative_trend_label(loading_change: float, residual_change: float, scale: float) -> str:
    """把数值变化转成趋势标签，避免把弱相关/小差值过度解释为机制。"""
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    if not np.isfinite(loading_change):
        return "无法判断"
    if abs(loading_change) < 0.05 * scale:
        return "加载响应不明显"
    if not np.isfinite(residual_change):
        return "有加载响应，但残余未知"
    ratio = abs(residual_change) / (abs(loading_change) + 1e-12)
    if ratio < 0.2:
        return "加载响应明显，卸载后基本恢复"
    if ratio < 0.7:
        return "加载响应明显，卸载后有部分残余"
    return "加载响应明显，卸载后残余较强"


def build_stage_response_summary(layer_df: pd.DataFrame, selected: Dict[str, int]) -> pd.DataFrame:
    """生成以阶段趋势为核心的逐层 summary。

    这个表不追求逐点线性相关，而是回答更物理的问题：
    initial → compressed hold → unload → P0 hold 过程中，每一层的量是增强、恢复还是残余。
    """
    if layer_df.empty:
        return pd.DataFrame()

    stage_steps = {
        "initial": selected.get("initial"),
        "compressed_hold": selected.get("compressed_hold_end"),
        "unload_end": selected.get("unload_end"),
        "P0_hold": selected.get("P0_hold_end"),
        "first_opening": selected.get("first_opening_ramp_end"),
    }
    variables = [
        "dPxy_mag_C_m2_rms",
        "dPz_C_m2_rms",
        "Pxy_mag_C_m2_rms",
        "Pz_C_m2_rms",
        "uz_rms",
        "z_std",
        "abs_dh_upper_rms",
        "abs_dh_lower_rms",
        "abs_dh_best_rms",
        "abs_dh_nearest_rms",
        "grad_h_upper_mag_rms",
        "grad_h_lower_mag_rms",
        "grad_h_best_mag_rms",
        "grad_h_nearest_mag_rms",
        "eps_xy_rms",
        "omega_z_rms",
        "curl_u_z_rms",
        "grad_uz_mag_rms",
        "curl_dPxy_rms",
        "sigma_zz_GPa_mean",
        "sigma_xy_GPa_mean",
        "sigma_xz_GPa_mean",
        "sigma_yz_GPa_mean",
    ]
    variables = [v for v in variables if v in layer_df.columns]

    rows = []
    for layer in sorted(layer_df["layer"].unique()):
        sub = layer_df[layer_df["layer"] == layer].set_index("step")
        for var in variables:
            vals = {}
            for label, step in stage_steps.items():
                if step is not None and step in sub.index:
                    vals[label] = float(sub.loc[step, var])
                else:
                    vals[label] = np.nan
            initial = vals.get("initial", np.nan)
            comp = vals.get("compressed_hold", np.nan)
            unload = vals.get("unload_end", np.nan)
            p0 = vals.get("P0_hold", np.nan)
            loading_change = comp - initial if np.isfinite(comp) and np.isfinite(initial) else np.nan
            unload_change = unload - comp if np.isfinite(unload) and np.isfinite(comp) else np.nan
            residual_change = p0 - initial if np.isfinite(p0) and np.isfinite(initial) else np.nan
            recovery_fraction = np.nan
            if np.isfinite(loading_change) and abs(loading_change) > 1e-12 and np.isfinite(p0):
                recovery_fraction = 1.0 - abs(p0 - initial) / (abs(loading_change) + 1e-12)
            scale = np.nanmax(np.abs([initial, comp, unload, p0])) if np.any(np.isfinite([initial, comp, unload, p0])) else 1.0
            rows.append({
                "layer": int(layer),
                "variable": var,
                "initial": initial,
                "compressed_hold": comp,
                "unload_end": unload,
                "P0_hold": p0,
                "first_opening": vals.get("first_opening", np.nan),
                "loading_change": loading_change,
                "unload_change": unload_change,
                "residual_change": residual_change,
                "recovery_fraction_rough": recovery_fraction,
                "trend_label": qualitative_trend_label(loading_change, residual_change, scale),
            })
    return pd.DataFrame(rows).sort_values(["layer", "variable"])


def build_qualitative_mechanism_summary(stage_df: pd.DataFrame) -> pd.DataFrame:
    """把 stage_response_summary 进一步压缩成面向机制判断的中文表。

    该表只给趋势判断，不把单个相关系数当作结论。
    """
    if stage_df.empty:
        return pd.DataFrame()
    rows = []
    for layer in sorted(stage_df["layer"].unique()):
        sub = stage_df[stage_df["layer"] == layer].set_index("variable")
        def get_label(var: str) -> str:
            return str(sub.loc[var, "trend_label"]) if var in sub.index else "未计算"
        def get_res(var: str) -> float:
            return float(sub.loc[var, "residual_change"]) if var in sub.index else np.nan
        if layer in EXCLUDE_LAYERS:
            role = "表面/加载异常层，排除主机制"
        elif layer in CLEAN_LAYERS:
            role = "clean response/readout layer"
        elif layer == BACKGROUND_LAYER:
            role = "background + redistribution layer"
        elif layer in [3, 8]:
            role = "中等继承层"
        else:
            role = "弱继承/边界层"
        rows.append({
            "layer": int(layer),
            "role": role,
            "dPxy_trend": get_label("dPxy_mag_C_m2_rms"),
            "dPz_trend": get_label("dPz_C_m2_rms"),
            "epsxy_trend": get_label("eps_xy_rms"),
            "omega_trend": get_label("omega_z_rms"),
            "stress_zz_trend": get_label("sigma_zz_GPa_mean"),
            "dPxy_residual_change": get_res("dPxy_mag_C_m2_rms"),
            "interpretation_note": (
                "主要看 loading-induced response 和 P0 residual，不要求与单一 scalar proxy 逐点线性相关"
                if layer in CLEAN_LAYERS else
                "该层可能混有强 background / surface / boundary 效应，避免作为 clean 定量层"
            ),
        })
    return pd.DataFrame(rows).sort_values("layer")


def top_fraction_mask(values: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    """返回绝对值最高 fraction 区域的布尔 mask。"""
    v = np.asarray(values, float)
    mask = np.isfinite(v)
    out = np.zeros(v.shape, dtype=bool)
    if np.count_nonzero(mask) < 5:
        return out
    abs_v = np.abs(v[mask])
    threshold = np.nanpercentile(abs_v, 100.0 * (1.0 - fraction))
    out[mask] = np.abs(v[mask]) >= threshold
    return out


def build_pattern_overlap_metrics(cell_df: pd.DataFrame, selected_steps: Sequence[int], fraction: float = 0.20) -> pd.DataFrame:
    """计算空间重合趋势，而不是逐点线性相关。

    物理含义：极化重分布强的区域，是否也落在 rotation / shear / vertical gradient / stress proxy 强的区域。
    这是面向纹理图案的定性判据，比单一 Pearson correlation 更适合 moiré texture。
    """
    if cell_df.empty:
        return pd.DataFrame()
    df = cell_df[cell_df["step"].isin(selected_steps)].copy()
    if df.empty:
        return pd.DataFrame()
    if all(c in df.columns for c in ["sigma_xy_GPa", "sigma_xz_GPa", "sigma_yz_GPa"]):
        df["sigma_shear_mag_GPa"] = np.sqrt(df["sigma_xy_GPa"] ** 2 + df["sigma_xz_GPa"] ** 2 + df["sigma_yz_GPa"] ** 2)
    comparators = [
        ("abs_eps_xy", "eps_xy"),
        ("abs_omega_z", "omega_z"),
        ("abs_curl_u_z", "curl_u_z"),
        ("grad_uz_mag", "grad_uz_mag"),
        ("abs_curl_dPxy", "curl_dPxy"),
        ("abs_sigma_zz", "sigma_zz_GPa"),
        ("sigma_shear_mag", "sigma_shear_mag_GPa"),
    ]
    rows = []
    for (step, layer), g in df.groupby(["step", "layer"]):
        if int(layer) in EXCLUDE_LAYERS:
            continue
        base = top_fraction_mask(g["dPxy_mag_C_m2"].to_numpy(float), fraction=fraction)
        base_count = int(np.count_nonzero(base))
        if base_count == 0:
            continue
        for name, col in comparators:
            if col not in g.columns:
                continue
            comp = top_fraction_mask(g[col].to_numpy(float), fraction=fraction)
            comp_count = int(np.count_nonzero(comp))
            inter = int(np.count_nonzero(base & comp))
            overlap_on_dPxy = inter / base_count if base_count else np.nan
            overlap_on_comp = inter / comp_count if comp_count else np.nan
            enrichment = overlap_on_dPxy / fraction if fraction > 0 else np.nan
            if enrichment >= 1.5:
                trend = "空间重合偏强"
            elif enrichment >= 0.8:
                trend = "空间重合一般"
            else:
                trend = "空间重合偏弱"
            rows.append({
                "step": int(step),
                "layer": int(layer),
                "base": "top_abs_dPxy",
                "comparator": name,
                "top_fraction": fraction,
                "base_count": base_count,
                "comparator_count": comp_count,
                "intersection_count": inter,
                "overlap_on_dPxy": overlap_on_dPxy,
                "overlap_on_comparator": overlap_on_comp,
                "enrichment_vs_random": enrichment,
                "trend_label": trend,
            })
    return pd.DataFrame(rows).sort_values(["step", "layer", "comparator"]) if rows else pd.DataFrame()


def vector_alignment_stats(px: np.ndarray, py: np.ndarray, ax: np.ndarray, ay: np.ndarray, high_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    pnorm = np.sqrt(px ** 2 + py ** 2)
    anorm = np.sqrt(ax ** 2 + ay ** 2)
    mask = np.isfinite(px) & np.isfinite(py) & np.isfinite(ax) & np.isfinite(ay) & (pnorm > 1e-12) & (anorm > 1e-12)
    if high_mask is not None:
        mask = mask & high_mask
    if np.count_nonzero(mask) < 4:
        return {"n": int(np.count_nonzero(mask)), "signed_mean_cos": np.nan, "abs_mean_cos": np.nan, "median_abs_cos": np.nan}
    cosv = (px[mask] * ax[mask] + py[mask] * ay[mask]) / (pnorm[mask] * anorm[mask])
    return {
        "n": int(cosv.size),
        "signed_mean_cos": float(np.nanmean(cosv)),
        "abs_mean_cos": float(np.nanmean(np.abs(cosv))),
        "median_abs_cos": float(np.nanmedian(np.abs(cosv))),
    }


def build_vector_alignment_metrics(cell_df: pd.DataFrame, selected_steps: Sequence[int]) -> pd.DataFrame:
    """计算矢量方向一致性，避免只看 x/y 单分量相关。

    如果方向一致但幅值非线性，Pearson correlation 可能很低；vector alignment 更适合定性判断。
    """
    if cell_df.empty:
        return pd.DataFrame()
    df = cell_df[cell_df["step"].isin(selected_steps)].copy()
    if df.empty:
        return pd.DataFrame()
    vector_fields = [
        ("grad_uz", "duz_dx", "duz_dy"),
        ("rotated_grad_uz", "duz_dy", "duz_dx"),
        ("strict_shear_gradient_vector", "depsxy_dy", "depsxy_dx"),
    ]
    rows = []
    for (step, layer), g in df.groupby(["step", "layer"]):
        if int(layer) in EXCLUDE_LAYERS:
            continue
        dpx = g["dPx_C_m2"].to_numpy(float)
        dpy = g["dPy_C_m2"].to_numpy(float)
        high = top_fraction_mask(g["dPxy_mag_C_m2"].to_numpy(float), fraction=HIGH_RESPONSE_FRACTION)
        for name, ax_col, ay_col in vector_fields:
            if ax_col not in g.columns or ay_col not in g.columns:
                continue
            ax = g[ax_col].to_numpy(float)
            ay = g[ay_col].to_numpy(float)
            if name == "rotated_grad_uz":
                ax = -ax
            all_stats = vector_alignment_stats(dpx, dpy, ax, ay, None)
            high_stats = vector_alignment_stats(dpx, dpy, ax, ay, high)
            abs_cos = high_stats.get("abs_mean_cos", np.nan)
            if np.isfinite(abs_cos) and abs_cos >= 0.65:
                trend = "高响应区方向较一致"
            elif np.isfinite(abs_cos) and abs_cos >= 0.45:
                trend = "高响应区方向有一定一致性"
            else:
                trend = "高响应区方向一致性不明显"
            rows.append({
                "step": int(step),
                "layer": int(layer),
                "vector_proxy": name,
                "n_all": all_stats["n"],
                "signed_mean_cos_all": all_stats["signed_mean_cos"],
                "abs_mean_cos_all": all_stats["abs_mean_cos"],
                "median_abs_cos_all": all_stats["median_abs_cos"],
                "n_high_dPxy": high_stats["n"],
                "signed_mean_cos_high_dPxy": high_stats["signed_mean_cos"],
                "abs_mean_cos_high_dPxy": high_stats["abs_mean_cos"],
                "median_abs_cos_high_dPxy": high_stats["median_abs_cos"],
                "trend_label": trend,
            })
    return pd.DataFrame(rows).sort_values(["step", "layer", "vector_proxy"]) if rows else pd.DataFrame()


def build_vertical_channel_summary(stage_df: pd.DataFrame, magnitude_df: pd.DataFrame) -> pd.DataFrame:
    """专门总结面外极化 Pz / dPz 的 vertical channel。

    Pxy 主要用于 in-plane redistribution；Pz 更接近 normal/vertical geometry response。
    这里不追求严格函数式，而是看 dPz 的阶段趋势、layer 5/clean 大小关系、以及与 uz/grad_uz 的协同趋势。
    """
    if stage_df.empty:
        return pd.DataFrame()

    rows = []
    dPz = stage_df[stage_df["variable"] == "dPz_C_m2_rms"].copy()
    grad = stage_df[stage_df["variable"] == "grad_uz_mag_rms"].copy()
    uz = stage_df[stage_df["variable"] == "uz_rms"].copy() if "uz_rms" in stage_df["variable"].unique() else pd.DataFrame()

    def group_mean(df: pd.DataFrame, layers: List[int], col: str) -> float:
        if df.empty or col not in df.columns:
            return np.nan
        vals = df[df["layer"].isin(layers)][col].to_numpy(float)
        return float(np.nanmean(vals)) if vals.size else np.nan

    groups = [
        ("clean", CLEAN_LAYERS, "clean response/readout layers"),
        ("layer5", [BACKGROUND_LAYER], "background + redistribution layer"),
        ("medium", [3, 8], "medium inheritance layers"),
        ("weak", [2, 9], "weak inheritance/boundary layers"),
    ]
    for group_name, layers, note in groups:
        comp_dPz = group_mean(dPz, layers, "compressed_hold")
        p0_dPz = group_mean(dPz, layers, "P0_hold")
        comp_grad = group_mean(grad, layers, "compressed_hold")
        p0_grad = group_mean(grad, layers, "P0_hold")
        comp_uz = group_mean(uz, layers, "compressed_hold")
        p0_uz = group_mean(uz, layers, "P0_hold")
        residual_ratio = safe_ratio(p0_dPz, comp_dPz)
        grad_ratio = safe_ratio(p0_grad, comp_grad)
        if np.isfinite(residual_ratio):
            if residual_ratio < 0.7:
                trend = "Pz 卸载后部分恢复"
            elif residual_ratio < 1.3:
                trend = "Pz 卸载后保留较强残余"
            else:
                trend = "Pz 在 P0_hold 后继续增强/重排"
        else:
            trend = "无法判断"
        rows.append({
            "group": group_name,
            "layers": ",".join(str(x) for x in layers),
            "role": note,
            "dPz_compressed_hold": comp_dPz,
            "dPz_P0_hold": p0_dPz,
            "dPz_P0_over_compressed": residual_ratio,
            "grad_uz_compressed_hold": comp_grad,
            "grad_uz_P0_hold": p0_grad,
            "grad_uz_P0_over_compressed": grad_ratio,
            "uz_compressed_hold": comp_uz,
            "uz_P0_hold": p0_uz,
            "trend_label": trend,
            "interpretation_note": "用于判断 Pz vertical/normal channel；只做趋势与大小关系，不作为严格本构式",
        })

    if not magnitude_df.empty:
        mag = magnitude_df[magnitude_df["variable"] == "dPz_C_m2_rms"]
        for stage in ["compressed_hold", "P0_hold"]:
            l5 = mag[(mag["stage"] == stage) & (mag["comparison"] == "layer5_vs_clean")]
            if not l5.empty:
                rows.append({
                    "group": f"layer5_vs_clean_{stage}",
                    "layers": "5 vs 4,6,7",
                    "role": "vertical-channel size relation",
                    "dPz_compressed_hold": np.nan,
                    "dPz_P0_hold": np.nan,
                    "dPz_P0_over_compressed": float(l5.iloc[0]["ratio_to_clean"]),
                    "grad_uz_compressed_hold": np.nan,
                    "grad_uz_P0_hold": np.nan,
                    "grad_uz_P0_over_compressed": np.nan,
                    "uz_compressed_hold": np.nan,
                    "uz_P0_hold": np.nan,
                    "trend_label": f"layer 5 dPz 相对 clean layers：{str(l5.iloc[0]['trend_label'])}",
                    "interpretation_note": "该行 ratio 字段表示 layer5/clean 的 dPz 大小关系，而非 P0/compressed 比值",
                })

    return pd.DataFrame(rows)


def build_core_mechanism_summary(
    frame_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    magnitude_df: pd.DataFrame,
    vertical_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    vector_df: pd.DataFrame,
) -> pd.DataFrame:
    """生成精简总表：默认输出中最应该先看的表。"""
    rows = []
    def add(topic: str, score_hint: str, evidence: str, interpretation: str, caution: str = "") -> None:
        rows.append({
            "topic": topic,
            "support_level": score_hint,
            "key_evidence": evidence,
            "interpretation": interpretation,
            "caution": caution,
        })

    if not frame_df.empty:
        final = frame_df.iloc[-1]
        add(
            "Pxy redistribution in clean layers",
            "中等偏强",
            f"P0_hold clean |dPxy|={final.get('clean_dPxy_rms_mean', np.nan):.4g} C/m^2; analysis layers={final.get('analysis_dPxy_rms_mean', np.nan):.4g}",
            "clean layers 存在可测的 Pxy 重分布，适合作为 readout 响应层。",
            "不能简化为单点线性 scalar relation。",
        )
        add(
            "Pz vertical channel",
            "需要重点加强",
            f"P0_hold clean |dPz|={final.get('clean_dPz_rms_mean', np.nan):.4g} C/m^2; layer5 |dPz|={final.get('layer5_dPz_rms_mean', np.nan):.4g}",
            "Pz 反映 normal/vertical geometry channel，应与 Pxy 并列分析。",
            "后续应检查 dPz 与 uz/grad_uz/local gap 的阶段协同。",
        )

    if not magnitude_df.empty:
        dpxy = magnitude_df[magnitude_df["variable"] == "dPxy_mag_C_m2_rms"]
        dPz = magnitude_df[magnitude_df["variable"] == "dPz_C_m2_rms"]
        l5_pxy = dpxy[(dpxy["comparison"] == "layer5_vs_clean") & (dpxy["stage"] == "P0_hold")]
        l5_pz = dPz[(dPz["comparison"] == "layer5_vs_clean") & (dPz["stage"] == "P0_hold")]
        if not l5_pxy.empty:
            add(
                "Layer 5 Pxy background + redistribution",
                "强",
                f"P0_hold layer5/clean |dPxy| ratio={float(l5_pxy.iloc[0]['ratio_to_clean']):.3g}",
                "layer 5 与 clean layers 有数量级差异，应作为 background + redistribution 层。",
                "不要把 layer 5 当作 clean readout layer。",
            )
        if not l5_pz.empty:
            add(
                "Layer 5 coupled Pxy-Pz response",
                "中等偏强",
                f"P0_hold layer5/clean |dPz| ratio={float(l5_pz.iloc[0]['ratio_to_clean']):.3g}",
                "layer 5 不只是面内 Pxy 特殊，也表现出强 Pz/vertical-channel 重排。",
                "需要排除局域 cell reconstruction 或 hold relaxation artifact。",
            )

    if not vertical_df.empty:
        clean_v = vertical_df[vertical_df["group"] == "clean"]
        if not clean_v.empty:
            add(
                "Clean-layer Pz residual",
                "中等",
                f"clean dPz P0/compressed={float(clean_v.iloc[0]['dPz_P0_over_compressed']):.3g}; grad_uz P0/compressed={float(clean_v.iloc[0]['grad_uz_P0_over_compressed']):.3g}",
                "clean layers 的 Pz 是否在 P0_hold 后继续增强，需要与 vertical geometry 同步检查。",
                "这是下一轮提高机理支撑度的关键。",
            )

    if not pattern_df.empty:
        clean_pat = pattern_df[pattern_df["layer"].isin(CLEAN_LAYERS)]
        if not clean_pat.empty:
            med = clean_pat.groupby("comparator")["enrichment_vs_random"].median(numeric_only=True).sort_values(ascending=False)
            if not med.empty:
                add(
                    "Spatial co-localization",
                    "弱到中等",
                    f"best comparator={med.index[0]}, enrichment={float(med.iloc[0]):.3g}",
                    "强响应区有一定空间共定位趋势，可作为辅助证据。",
                    "enrichment 不高时不要写成决定关系。",
                )

    if not vector_df.empty:
        clean_vec = vector_df[vector_df["layer"].isin(CLEAN_LAYERS)]
        if not clean_vec.empty:
            med = clean_vec.groupby("vector_proxy")["abs_mean_cos_high_dPxy"].median(numeric_only=True).sort_values(ascending=False)
            if not med.empty:
                add(
                    "Pxy vector direction template",
                    "中等偏强",
                    f"best vector={med.index[0]}, median |cosθ|={float(med.iloc[0]):.3g}",
                    "Pxy 的方向组织可能受 vertical geometry / displacement template 调制。",
                    "|cosθ| 是方向/轴向一致性，不代表幅值线性关系。",
                )

    return pd.DataFrame(rows)


def safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.nan
    return float(numer / denom)


def ratio_trend_label(ratio: float) -> str:
    """把大小比值转成定性判断。"""
    if not np.isfinite(ratio):
        return "无法判断"
    ar = abs(ratio)
    if ar >= STRONG_RATIO_THRESHOLD:
        return "明显更强"
    if ar >= MODERATE_RATIO_THRESHOLD:
        return "偏强"
    if ar <= 1.0 / STRONG_RATIO_THRESHOLD:
        return "明显更弱"
    if ar <= 1.0 / MODERATE_RATIO_THRESHOLD:
        return "偏弱"
    return "相近"


def build_magnitude_relation_summary(stage_df: pd.DataFrame) -> pd.DataFrame:
    """生成层间大小关系与阶段残余关系。

    这里不是做复杂数值拟合，而是把“谁更强、强多少、是否残余”说清楚。
    这些大小关系是定性机制判断的一部分，并不等于过度数值化。
    """
    if stage_df.empty:
        return pd.DataFrame()
    rows = []
    variables = [
        "dPxy_mag_C_m2_rms",
        "dPz_C_m2_rms",
        "Pxy_mag_C_m2_rms",
        "Pz_C_m2_rms",
        "uz_rms",
        "abs_dh_upper_rms",
        "abs_dh_lower_rms",
        "abs_dh_best_rms",
        "abs_dh_nearest_rms",
        "grad_h_upper_mag_rms",
        "grad_h_lower_mag_rms",
        "grad_h_best_mag_rms",
        "grad_h_nearest_mag_rms",
        "eps_xy_rms",
        "omega_z_rms",
        "grad_uz_mag_rms",
        "curl_dPxy_rms",
        "sigma_zz_GPa_mean",
        "sigma_xy_GPa_mean",
        "sigma_xz_GPa_mean",
        "sigma_yz_GPa_mean",
    ]
    stages = ["compressed_hold", "P0_hold"]
    for var in variables:
        sub = stage_df[stage_df["variable"] == var]
        if sub.empty:
            continue
        for stage in stages:
            clean_vals = sub[sub["layer"].isin(CLEAN_LAYERS)][stage].to_numpy(float) if stage in sub.columns else np.array([])
            layer5_vals = sub[sub["layer"] == BACKGROUND_LAYER][stage].to_numpy(float) if stage in sub.columns else np.array([])
            medium_vals = sub[sub["layer"].isin([3, 8])][stage].to_numpy(float) if stage in sub.columns else np.array([])
            weak_vals = sub[sub["layer"].isin([2, 9])][stage].to_numpy(float) if stage in sub.columns else np.array([])
            clean_mean = float(np.nanmean(clean_vals)) if clean_vals.size else np.nan
            layer5_val = float(np.nanmean(layer5_vals)) if layer5_vals.size else np.nan
            medium_mean = float(np.nanmean(medium_vals)) if medium_vals.size else np.nan
            weak_mean = float(np.nanmean(weak_vals)) if weak_vals.size else np.nan
            for compare_name, numer in [
                ("layer5_vs_clean", layer5_val),
                ("medium_vs_clean", medium_mean),
                ("weak_vs_clean", weak_mean),
            ]:
                r = safe_ratio(numer, clean_mean)
                rows.append({
                    "variable": var,
                    "stage": stage,
                    "comparison": compare_name,
                    "numerator_value": numer,
                    "clean_reference_value": clean_mean,
                    "ratio_to_clean": r,
                    "trend_label": ratio_trend_label(r),
                    "interpretation_note": "大小关系用于定性机制判断；不代表本构比例系数",
                })

        # 残余关系：P0_hold / compressed_hold。用于判断卸载后是否保留记忆。
        for group_name, layers in [("clean", CLEAN_LAYERS), ("layer5", [BACKGROUND_LAYER]), ("medium", [3, 8]), ("weak", [2, 9])]:
            vals_comp = sub[sub["layer"].isin(layers)]["compressed_hold"].to_numpy(float) if "compressed_hold" in sub.columns else np.array([])
            vals_p0 = sub[sub["layer"].isin(layers)]["P0_hold"].to_numpy(float) if "P0_hold" in sub.columns else np.array([])
            comp_mean = float(np.nanmean(vals_comp)) if vals_comp.size else np.nan
            p0_mean = float(np.nanmean(vals_p0)) if vals_p0.size else np.nan
            r = safe_ratio(p0_mean, comp_mean)
            if np.isfinite(r):
                if abs(r) < 0.2:
                    label = "卸载后基本消失"
                elif abs(r) < 0.7:
                    label = "卸载后部分残余"
                elif abs(r) < 1.3:
                    label = "卸载后保留较强残余"
                else:
                    label = "P0_hold 后继续增强/重排"
            else:
                label = "无法判断"
            rows.append({
                "variable": var,
                "stage": "P0_vs_compressed",
                "comparison": f"{group_name}_residual_ratio",
                "numerator_value": p0_mean,
                "clean_reference_value": comp_mean,
                "ratio_to_clean": r,
                "trend_label": label,
                "interpretation_note": "P0_hold/compressed_hold 比值用于判断恢复、残余或后续重排",
            })

    return pd.DataFrame(rows).sort_values(["variable", "stage", "comparison"]) if rows else pd.DataFrame()


def layer_group_mean_from_stage(stage_df: pd.DataFrame, variable: str, layers: List[int], stage: str) -> float:
    sub = stage_df[(stage_df["variable"] == variable) & (stage_df["layer"].isin(layers))]
    if sub.empty or stage not in sub.columns:
        return np.nan
    vals = sub[stage].to_numpy(float)
    return float(np.nanmean(vals)) if vals.size else np.nan


def build_time_continuity_evidence(frame_df: pd.DataFrame, params: StageParams) -> pd.DataFrame:
    """检查 P0_hold 增强是连续演化还是单帧跳变。

    该表用于验证“delayed redistribution / residual reorganization”猜想。
    jump_fraction 越小，越不像单帧跳变；但这只是证据，不是严格判据。
    """
    if frame_df.empty:
        return pd.DataFrame()
    rows = []
    series = [
        ("clean_dPxy", "clean_dPxy_rms_mean"),
        ("clean_dPz", "clean_dPz_rms_mean"),
        ("layer5_dPxy", "layer5_dPxy_rms_mean"),
        ("layer5_dPz", "layer5_dPz_rms_mean"),
        ("clean_grad_uz", "clean_grad_uz_rms_mean"),
        ("layer5_grad_uz", "layer5_grad_uz_rms_mean"),
        ("clean_abs_dh_upper", "clean_abs_dh_upper_rms_mean"),
        ("layer5_abs_dh_upper", "layer5_abs_dh_upper_rms_mean"),
    ]
    p0 = frame_df[(frame_df["step"] >= params.unload_end) & (frame_df["step"] <= params.p0_hold_end)].sort_values("step")
    for name, col in series:
        if col not in frame_df.columns:
            continue
        vals = p0[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 3:
            continue
        total_change = float(vals[-1] - vals[0])
        jumps = np.abs(np.diff(vals))
        max_jump = float(np.nanmax(jumps)) if jumps.size else np.nan
        jump_fraction = abs(max_jump / total_change) if abs(total_change) > 1e-12 else np.nan
        n_increase = int(np.sum(np.diff(vals) > 0))
        trend = "连续增强倾向" if np.isfinite(jump_fraction) and jump_fraction < 0.75 and total_change > 0 else "可能有跳变或非单调"
        rows.append({
            "series": name,
            "p0_start_value": float(vals[0]),
            "p0_end_value": float(vals[-1]),
            "p0_total_change": total_change,
            "max_adjacent_jump": max_jump,
            "jump_fraction": jump_fraction,
            "n_increase_steps": n_increase,
            "n_segments": int(max(vals.size - 1, 0)),
            "trend_label": trend,
            "interpretation_note": "用于检查 P0_hold 后增强是否像连续重排，而不是单帧后处理跳变",
        })
    return pd.DataFrame(rows)


def spatial_match_metrics(g: pd.DataFrame, response_values: np.ndarray, template_values: np.ndarray, fraction: float = 0.20) -> Dict[str, float]:
    """计算两个场的强响应区是否在同一位置或接近位置。

    enrichment > 1 表示比随机重合更强；
    median_nn_distance_norm 越小，说明 response 强区离 template 强区越近。
    """
    x = g["x_ref"].to_numpy(float)
    y = g["y_ref"].to_numpy(float)
    pts = np.column_stack([x, y])
    rmask = top_fraction_mask(response_values, fraction=fraction)
    tmask = top_fraction_mask(template_values, fraction=fraction)
    nr = int(np.count_nonzero(rmask))
    nt = int(np.count_nonzero(tmask))
    if nr == 0 or nt == 0:
        return {
            "n_response_top": nr,
            "n_template_top": nt,
            "overlap_on_response": np.nan,
            "jaccard": np.nan,
            "enrichment_vs_random": np.nan,
            "centroid_distance_norm": np.nan,
            "median_nn_distance_norm": np.nan,
        }
    inter = int(np.count_nonzero(rmask & tmask))
    union = int(np.count_nonzero(rmask | tmask))
    overlap = inter / nr
    jaccard = inter / union if union else np.nan
    enrichment = overlap / fraction if fraction > 0 else np.nan
    lx = float(np.nanmax(x) - np.nanmin(x)) if len(x) else np.nan
    ly = float(np.nanmax(y) - np.nanmin(y)) if len(y) else np.nan
    norm_len = math.sqrt(lx * lx + ly * ly) if np.isfinite(lx) and np.isfinite(ly) and lx > 0 and ly > 0 else 1.0
    cr = np.nanmean(pts[rmask], axis=0)
    ct = np.nanmean(pts[tmask], axis=0)
    centroid_distance_norm = float(np.linalg.norm(cr - ct) / norm_len)
    tree = cKDTree(pts[tmask])
    d, _ = tree.query(pts[rmask], k=1)
    median_nn_distance_norm = float(np.nanmedian(d) / norm_len)
    return {
        "n_response_top": nr,
        "n_template_top": nt,
        "overlap_on_response": float(overlap),
        "jaccard": float(jaccard),
        "enrichment_vs_random": float(enrichment),
        "centroid_distance_norm": centroid_distance_norm,
        "median_nn_distance_norm": median_nn_distance_norm,
    }


def build_spatial_evidence_summary(selected_cell_df: pd.DataFrame, selected: Dict[str, Optional[int]], fraction: float = 0.20) -> pd.DataFrame:
    """输出位置证据：响应强区和模板强区是否在同一位置或邻近位置。

    这是为了避免只看全局趋势。尤其用于验证：
    Pz 是否与 h/dh 同位置；Pxy 是否与 shear-gradient / rotation / vorticity 同位置。
    """
    if selected_cell_df.empty:
        return pd.DataFrame()
    steps = [selected.get("compressed_hold_end"), selected.get("P0_hold_end")]
    steps = [int(s) for s in steps if s is not None]
    df = selected_cell_df[selected_cell_df["step"].isin(steps)].copy()
    if df.empty:
        return pd.DataFrame()
    df["abs_dPz"] = np.abs(df["dPz_C_m2"].to_numpy(float))
    df["abs_omega_z"] = np.abs(df["omega_z"].to_numpy(float))
    df["abs_curl_u_z"] = np.abs(df["curl_u_z"].to_numpy(float))
    df["shear_gradient_mag"] = np.sqrt(df["depsxy_dx"].to_numpy(float) ** 2 + df["depsxy_dy"].to_numpy(float) ** 2)
    if all(c in df.columns for c in ["sigma_xy_GPa", "sigma_xz_GPa", "sigma_yz_GPa"]):
        df["sigma_shear_mag_GPa"] = np.sqrt(df["sigma_xy_GPa"] ** 2 + df["sigma_xz_GPa"] ** 2 + df["sigma_yz_GPa"] ** 2)
    pairs = [
        ("Pz_vs_abs_dh_upper", "abs_dPz", "abs_dh_upper"),
        ("Pz_vs_abs_dh_lower", "abs_dPz", "abs_dh_lower"),
        ("Pz_vs_abs_dh_best", "abs_dPz", "abs_dh_best"),
        ("Pz_vs_abs_dh_nearest", "abs_dPz", "abs_dh_nearest"),
        ("Pz_vs_grad_h_upper", "abs_dPz", "grad_h_upper_mag"),
        ("Pz_vs_grad_h_lower", "abs_dPz", "grad_h_lower_mag"),
        ("Pz_vs_grad_h_best", "abs_dPz", "grad_h_best_mag"),
        ("Pz_vs_grad_h_nearest", "abs_dPz", "grad_h_nearest_mag"),
        ("Pz_vs_grad_uz", "abs_dPz", "grad_uz_mag"),
        ("Pxy_vs_shear_gradient", "dPxy_mag_C_m2", "shear_gradient_mag"),
        ("Pxy_vs_omega", "dPxy_mag_C_m2", "abs_omega_z"),
        ("Pxy_vs_curl_u", "dPxy_mag_C_m2", "abs_curl_u_z"),
        ("Pxy_vs_grad_h_upper", "dPxy_mag_C_m2", "grad_h_upper_mag"),
        ("Pxy_vs_grad_h_lower", "dPxy_mag_C_m2", "grad_h_lower_mag"),
        ("Pxy_vs_grad_h_best", "dPxy_mag_C_m2", "grad_h_best_mag"),
        ("Pxy_vs_grad_h_nearest", "dPxy_mag_C_m2", "grad_h_nearest_mag"),
    ]
    if COMPUTE_SIGMA_SHEAR_EVIDENCE:
        pairs.append(("Pxy_vs_sigma_shear", "dPxy_mag_C_m2", "sigma_shear_mag_GPa"))
    rows = []
    for (step, layer), g in df.groupby(["step", "layer"]):
        li = int(layer)
        if li in EXCLUDE_LAYERS:
            continue
        for pair_name, response_col, template_col in pairs:
            if response_col not in g.columns or template_col not in g.columns:
                continue
            stats = spatial_match_metrics(g, g[response_col].to_numpy(float), g[template_col].to_numpy(float), fraction=fraction)
            if np.isfinite(stats["enrichment_vs_random"]):
                if stats["enrichment_vs_random"] >= 1.5:
                    verdict = "位置重合较强"
                elif stats["enrichment_vs_random"] >= 1.2:
                    verdict = "位置重合中等"
                else:
                    verdict = "位置重合较弱"
            else:
                verdict = "无法判断"
            row = {
                "step": int(step),
                "stage": stage_label(int(step), StageParams()),
                "layer": li,
                "pair": pair_name,
                "response": response_col,
                "template": template_col,
                "top_fraction": fraction,
                "verdict": verdict,
            }
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["step", "layer", "pair"]) if rows else pd.DataFrame()


def build_pxy_template_ranking(spatial_df: pd.DataFrame) -> pd.DataFrame:
    """比较 Pxy 与不同 mechanical template 的位置匹配强弱。"""
    if spatial_df.empty:
        return pd.DataFrame()
    df = spatial_df[spatial_df["pair"].str.startswith("Pxy_vs_")].copy()
    if df.empty:
        return pd.DataFrame()
    groups = []
    for group_name, layers in [("clean", CLEAN_LAYERS), ("layer5", [BACKGROUND_LAYER]), ("analysis", ANALYSIS_LAYERS)]:
        sub = df[df["layer"].isin(layers)]
        if sub.empty:
            continue
        rank = sub.groupby("pair")["enrichment_vs_random"].median(numeric_only=True).sort_values(ascending=False)
        for i, (pair, val) in enumerate(rank.items(), start=1):
            groups.append({
                "group": group_name,
                "rank": i,
                "pair": str(pair),
                "median_enrichment": float(val),
                "interpretation": "这是位置共定位排名，不是因果证明；若 shear-gradient 排名不高，说明当前数据不支持把 Pxy 简化为剪切应变梯度单机制。",
            })
    return pd.DataFrame(groups)


def build_pxy_direction_evidence(selected_cell_df: pd.DataFrame, selected: Dict[str, Optional[int]]) -> pd.DataFrame:
    """检查 Pxy 的方向是否与候选 mechanical vector template 对齐。

    这补充了大小/位置重合检验：Pxy 不一定和 shear-gradient 的幅值同位，
    也可能表现为方向或轴向对应。
    """
    if selected_cell_df.empty:
        return pd.DataFrame()
    steps = [selected.get("compressed_hold_end"), selected.get("P0_hold_end")]
    steps = [int(s) for s in steps if s is not None]
    df = selected_cell_df[selected_cell_df["step"].isin(steps)].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (step, layer), g in df.groupby(["step", "layer"]):
        li = int(layer)
        if li in EXCLUDE_LAYERS:
            continue
        dpx = g["dPx_C_m2"].to_numpy(float)
        dpy = g["dPy_C_m2"].to_numpy(float)
        high = top_fraction_mask(g["dPxy_mag_C_m2"].to_numpy(float), fraction=HIGH_RESPONSE_FRACTION)

        candidates: List[Tuple[str, np.ndarray, np.ndarray]] = []
        # 文献式候选：delta Px ~ d epsxy / dy, delta Py ~ d epsxy / dx
        candidates.append(("shear_gradient_literature", g["depsxy_dy"].to_numpy(float), g["depsxy_dx"].to_numpy(float)))
        candidates.append(("shear_gradient_direct", g["depsxy_dx"].to_numpy(float), g["depsxy_dy"].to_numpy(float)))
        candidates.append(("grad_uz", g["duz_dx"].to_numpy(float), g["duz_dy"].to_numpy(float)))
        candidates.append(("rotated_grad_uz", -g["duz_dy"].to_numpy(float), g["duz_dx"].to_numpy(float)))
        for prefix in ["upper", "lower", "nearest"]:
            xcol = f"grad_h_{prefix}_x"
            ycol = f"grad_h_{prefix}_y"
            if xcol in g.columns and ycol in g.columns:
                hx = g[xcol].to_numpy(float)
                hy = g[ycol].to_numpy(float)
                candidates.append((f"grad_h_{prefix}", hx, hy))
                candidates.append((f"rotated_grad_h_{prefix}", -hy, hx))
        if all(c in g.columns for c in ["grad_h_upper_x", "grad_h_upper_y", "grad_h_lower_x", "grad_h_lower_y"]):
            gux = g["grad_h_upper_x"].to_numpy(float)
            guy = g["grad_h_upper_y"].to_numpy(float)
            glx = g["grad_h_lower_x"].to_numpy(float)
            gly = g["grad_h_lower_y"].to_numpy(float)
            gum = np.sqrt(gux ** 2 + guy ** 2)
            glm = np.sqrt(glx ** 2 + gly ** 2)
            use_upper = gum >= glm
            bx = np.where(use_upper, gux, glx)
            by = np.where(use_upper, guy, gly)
            candidates.append(("grad_h_best", bx, by))
            candidates.append(("rotated_grad_h_best", -by, bx))
        if COMPUTE_SIGMA_SHEAR_EVIDENCE and all(c in g.columns for c in ["sigma_xz_GPa", "sigma_yz_GPa"]):
            candidates.append(("shear_stress_xz_yz", g["sigma_xz_GPa"].to_numpy(float), g["sigma_yz_GPa"].to_numpy(float)))

        for name, ax, ay in candidates:
            all_stats = vector_alignment_stats(dpx, dpy, ax, ay, None)
            high_stats = vector_alignment_stats(dpx, dpy, ax, ay, high)
            val = high_stats.get("median_abs_cos", np.nan)
            if np.isfinite(val) and val >= 0.70:
                verdict = "方向对应较强"
            elif np.isfinite(val) and val >= 0.55:
                verdict = "方向对应中等"
            else:
                verdict = "方向对应较弱"
            rows.append({
                "step": int(step),
                "stage": stage_label(int(step), StageParams()),
                "layer": li,
                "vector_template": name,
                "n_high_dPxy": high_stats["n"],
                "signed_mean_cos_high": high_stats["signed_mean_cos"],
                "abs_mean_cos_high": high_stats["abs_mean_cos"],
                "median_abs_cos_high": high_stats["median_abs_cos"],
                "abs_mean_cos_all": all_stats["abs_mean_cos"],
                "median_abs_cos_all": all_stats["median_abs_cos"],
                "verdict": verdict,
            })
    return pd.DataFrame(rows).sort_values(["step", "layer", "vector_template"]) if rows else pd.DataFrame()


def build_pxy_direction_ranking(direction_df: pd.DataFrame) -> pd.DataFrame:
    if direction_df.empty:
        return pd.DataFrame()
    rows = []
    for group_name, layers in [("clean", CLEAN_LAYERS), ("layer5", [BACKGROUND_LAYER]), ("analysis", ANALYSIS_LAYERS)]:
        sub = direction_df[direction_df["layer"].isin(layers)]
        if sub.empty:
            continue
        rank = sub.groupby("vector_template")["median_abs_cos_high"].median(numeric_only=True).sort_values(ascending=False)
        for i, (name, val) in enumerate(rank.items(), start=1):
            rows.append({
                "group": group_name,
                "rank": i,
                "vector_template": str(name),
                "median_abs_cos_high": float(val),
                "interpretation": "这是 Pxy 方向/轴向对应排名，不是幅值对应。shear-gradient 若方向排名高，说明可作为方向模板；若幅值排名低但方向排名高，也仍有物理意义。",
            })
    return pd.DataFrame(rows)


def sign_alignment_stats(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """符号/手性一致性统计。

    返回 same_fraction、opposite_fraction 和 best_fraction。
    best_fraction = max(same, opposite)，用于允许整体符号相反但图案仍有对应关系。
    """
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    m = np.isfinite(aa) & np.isfinite(bb) & (np.abs(aa) > 1e-12) & (np.abs(bb) > 1e-12)
    if mask is not None:
        m = m & mask
    n = int(np.count_nonzero(m))
    if n < 4:
        return {"n": n, "same_fraction": np.nan, "opposite_fraction": np.nan, "best_fraction": np.nan, "preferred_sign": "unknown"}
    same = np.sign(aa[m]) == np.sign(bb[m])
    same_fraction = float(np.mean(same))
    opposite_fraction = 1.0 - same_fraction
    if same_fraction >= opposite_fraction:
        preferred = "same"
        best = same_fraction
    else:
        preferred = "opposite"
        best = opposite_fraction
    return {"n": n, "same_fraction": same_fraction, "opposite_fraction": opposite_fraction, "best_fraction": float(best), "preferred_sign": preferred}


def build_pxy_nature_style_evidence(selected_cell_df: pd.DataFrame, selected: Dict[str, Optional[int]]) -> pd.DataFrame:
    """Nature-style Pxy 机制证据。

    不只检查 |dPxy| 与某个 template 的大小/位置是否重合，还检查：
    1) 文献式分量关系：dPx 对 depsxy_dy，dPy 对 depsxy_dx；
    2) Pxy 方向与 shear-gradient vector 的轴向一致性；
    3) curl/chirality：curl(dPxy) 与 rotation / displacement-vorticity 的符号或强区对应。
    """
    if selected_cell_df.empty:
        return pd.DataFrame()
    steps = [selected.get("compressed_hold_end"), selected.get("P0_hold_end")]
    steps = [int(s) for s in steps if s is not None]
    df = selected_cell_df[selected_cell_df["step"].isin(steps)].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (step, layer), g in df.groupby(["step", "layer"]):
        li = int(layer)
        if li in EXCLUDE_LAYERS:
            continue
        high_pxy = top_fraction_mask(g["dPxy_mag_C_m2"].to_numpy(float), fraction=HIGH_RESPONSE_FRACTION)
        high_curl = top_fraction_mask(g["curl_dPxy"].to_numpy(float), fraction=HIGH_RESPONSE_FRACTION)
        dpx = g["dPx_C_m2"].to_numpy(float)
        dpy = g["dPy_C_m2"].to_numpy(float)
        depsx = g["depsxy_dx"].to_numpy(float)
        depsy = g["depsxy_dy"].to_numpy(float)

        # 文献式 component 关系：不要求幅值线性，只看符号/趋势是否一致或整体相反。
        sx = sign_alignment_stats(dpx, depsy, high_pxy)
        sy = sign_alignment_stats(dpy, depsx, high_pxy)
        comp_best = np.nanmean([sx["best_fraction"], sy["best_fraction"]])
        rows.append({
            "step": int(step),
            "stage": stage_label(int(step), StageParams()),
            "layer": li,
            "test": "component_sign_literature",
            "template": "dPx~depsxy_dy and dPy~depsxy_dx",
            "metric": "mean_best_sign_fraction_high_dPxy",
            "value": float(comp_best) if np.isfinite(comp_best) else np.nan,
            "n": int(min(sx["n"], sy["n"])),
            "extra": f"x_pref={sx['preferred_sign']}; y_pref={sy['preferred_sign']}",
            "verdict": "component 符号对应较强" if np.isfinite(comp_best) and comp_best >= 0.70 else ("component 符号对应中等" if np.isfinite(comp_best) and comp_best >= 0.60 else "component 符号对应较弱"),
        })

        # 矢量方向：用 literature vector 与 direct vector 两种写法。
        for name, ax, ay in [
            ("vector_shear_gradient_literature", depsy, depsx),
            ("vector_shear_gradient_direct", depsx, depsy),
        ]:
            st = vector_alignment_stats(dpx, dpy, ax, ay, high_pxy)
            val = st["median_abs_cos"]
            rows.append({
                "step": int(step),
                "stage": stage_label(int(step), StageParams()),
                "layer": li,
                "test": name,
                "template": name.replace("vector_", ""),
                "metric": "median_abs_cos_high_dPxy",
                "value": val,
                "n": st["n"],
                "extra": f"signed_mean={st['signed_mean_cos']:.4g}",
                "verdict": "方向对应较强" if np.isfinite(val) and val >= 0.70 else ("方向对应中等" if np.isfinite(val) and val >= 0.55 else "方向对应较弱"),
            })

        # curl/chirality：Pxy 涡旋性是否跟 rotation/vorticity 相关。
        curl_dp = g["curl_dPxy"].to_numpy(float)
        for name, tpl in [
            ("curl_with_omega", g["omega_z"].to_numpy(float)),
            ("curl_with_curl_u", g["curl_u_z"].to_numpy(float)),
        ]:
            sg = sign_alignment_stats(curl_dp, tpl, high_curl)
            match = spatial_match_metrics(g, np.abs(curl_dp), np.abs(tpl), fraction=EVIDENCE_TOP_FRACTION)
            val = sg["best_fraction"]
            rows.append({
                "step": int(step),
                "stage": stage_label(int(step), StageParams()),
                "layer": li,
                "test": name,
                "template": name.replace("curl_with_", ""),
                "metric": "chirality_best_sign_fraction_and_top_overlap",
                "value": val,
                "n": sg["n"],
                "extra": f"preferred={sg['preferred_sign']}; curl_top_enrichment={match['enrichment_vs_random']:.4g}",
                "verdict": "curl/chirality 对应较强" if np.isfinite(val) and val >= 0.70 and np.isfinite(match["enrichment_vs_random"]) and match["enrichment_vs_random"] >= 1.2 else ("curl/chirality 部分对应" if np.isfinite(val) and val >= 0.60 else "curl/chirality 对应较弱"),
            })
    return pd.DataFrame(rows).sort_values(["step", "layer", "test"]) if rows else pd.DataFrame()


def build_pxy_nature_style_ranking(nature_df: pd.DataFrame) -> pd.DataFrame:
    if nature_df.empty:
        return pd.DataFrame()
    rows = []
    for group_name, layers in [("clean", CLEAN_LAYERS), ("layer5", [BACKGROUND_LAYER]), ("analysis", ANALYSIS_LAYERS)]:
        sub = nature_df[nature_df["layer"].isin(layers)]
        if sub.empty:
            continue
        rank = sub.groupby("test")["value"].median(numeric_only=True).sort_values(ascending=False)
        for i, (test, val) in enumerate(rank.items(), start=1):
            rows.append({
                "group": group_name,
                "rank": i,
                "test": str(test),
                "median_value": float(val),
                "interpretation": "Nature-style 证据排名：component sign、vector direction、curl/chirality 都可作为 Pxy 与 mechanical texture 的定性对应，不等同于幅值相关。",
            })
    return pd.DataFrame(rows)


def build_mechanism_evidence_summary(
    stage_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    selected_cell_df: pd.DataFrame,
    time_evidence_df: pd.DataFrame,
    selected: Dict[str, Optional[int]],
    spatial_evidence_df: Optional[pd.DataFrame] = None,
    pxy_ranking_df: Optional[pd.DataFrame] = None,
    pxy_direction_ranking_df: Optional[pd.DataFrame] = None,
    pxy_nature_ranking_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """把真实计算证据压缩为面向猜想验证的表。

    这里不是声称公式成立，而是逐条检查现有数据是否支持物理图像：
    Pz-h vertical channel、Pxy-shear/vorticity template、layer-dependent amplification、P0 delayed redistribution。
    """
    rows = []
    def add(hypothesis: str, evidence: str, verdict: str, caution: str) -> None:
        rows.append({
            "hypothesis": hypothesis,
            "computed_evidence": evidence,
            "verdict": verdict,
            "caution": caution,
        })

    clean_pz_ratio = safe_ratio(
        layer_group_mean_from_stage(stage_df, "dPz_C_m2_rms", CLEAN_LAYERS, "P0_hold"),
        layer_group_mean_from_stage(stage_df, "dPz_C_m2_rms", CLEAN_LAYERS, "compressed_hold"),
    )
    clean_h_ratio = safe_ratio(
        layer_group_mean_from_stage(stage_df, "abs_dh_upper_rms", CLEAN_LAYERS, "P0_hold"),
        layer_group_mean_from_stage(stage_df, "abs_dh_upper_rms", CLEAN_LAYERS, "compressed_hold"),
    )
    clean_gradh_ratio = safe_ratio(
        layer_group_mean_from_stage(stage_df, "grad_h_upper_mag_rms", CLEAN_LAYERS, "P0_hold"),
        layer_group_mean_from_stage(stage_df, "grad_h_upper_mag_rms", CLEAN_LAYERS, "compressed_hold"),
    )
    add(
        "Pz vertical-gap channel",
        f"clean P0/compressed |dPz|={clean_pz_ratio:.3g}, |dh_upper|={clean_h_ratio:.3g}, |grad_h_upper|={clean_gradh_ratio:.3g}",
        "支持" if np.isfinite(clean_pz_ratio) and clean_pz_ratio > 1.3 and ((np.isfinite(clean_h_ratio) and clean_h_ratio > 1.1) or (np.isfinite(clean_gradh_ratio) and clean_gradh_ratio > 1.1)) else "证据不足",
        "h_upper 是 IDW 局域 gap proxy，不是最终严格 buried-interface gap；需要后续 opposite-layer interface 插值版本确认。",
    )

    l5_pxy_ratio = safe_ratio(
        layer_group_mean_from_stage(stage_df, "dPxy_mag_C_m2_rms", [BACKGROUND_LAYER], "P0_hold"),
        layer_group_mean_from_stage(stage_df, "dPxy_mag_C_m2_rms", CLEAN_LAYERS, "P0_hold"),
    )
    l5_pz_ratio = safe_ratio(
        layer_group_mean_from_stage(stage_df, "dPz_C_m2_rms", [BACKGROUND_LAYER], "P0_hold"),
        layer_group_mean_from_stage(stage_df, "dPz_C_m2_rms", CLEAN_LAYERS, "P0_hold"),
    )
    l5_graduz_ratio = safe_ratio(
        layer_group_mean_from_stage(stage_df, "grad_uz_mag_rms", [BACKGROUND_LAYER], "P0_hold"),
        layer_group_mean_from_stage(stage_df, "grad_uz_mag_rms", CLEAN_LAYERS, "P0_hold"),
    )
    add(
        "Layer-dependent amplification instead of strain-copy",
        f"P0 layer5/clean ratios: |dPxy|={l5_pxy_ratio:.3g}, |dPz|={l5_pz_ratio:.3g}, |grad_uz|={l5_graduz_ratio:.3g}",
        "支持" if np.isfinite(l5_pxy_ratio) and np.isfinite(l5_graduz_ratio) and l5_pxy_ratio > 2.0 * max(l5_graduz_ratio, 1e-12) else "部分支持",
        "若几何模板相近但极化放大很多，说明需要 layer-dependent susceptibility / initial background；不是应变直接复制极化。",
    )

    p0_good = False
    if not time_evidence_df.empty:
        key = time_evidence_df[time_evidence_df["series"].isin(["clean_dPz", "layer5_dPz", "clean_dPxy", "layer5_dPxy"])]
        if not key.empty:
            p0_good = int((key["trend_label"] == "连续增强倾向").sum()) >= 2
            evidence = "; ".join([f"{r['series']}: jump_fraction={float(r['jump_fraction']):.3g}, {r['trend_label']}" for _, r in key.iterrows()])
        else:
            evidence = "time evidence unavailable"
    else:
        evidence = "time evidence unavailable"
    add(
        "P0_hold delayed redistribution",
        evidence,
        "支持" if p0_good else "需要检查",
        "该项用于排除单帧跳变；若 jump_fraction 大，应回看单帧/neighbor assignment。",
    )

    if spatial_evidence_df is not None and not spatial_evidence_df.empty:
        clean_or_l5 = spatial_evidence_df[spatial_evidence_df["layer"].isin(CLEAN_LAYERS + [BACKGROUND_LAYER])]
        pz_h = clean_or_l5[clean_or_l5["pair"] == "Pz_vs_abs_dh_upper"]
        if not pz_h.empty:
            med_enrich = float(np.nanmedian(pz_h["enrichment_vs_random"].to_numpy(float)))
            med_nn = float(np.nanmedian(pz_h["median_nn_distance_norm"].to_numpy(float)))
            add(
                "Spatial Pz-h positional match",
                f"top |dPz| with top |dh_upper|: median enrichment={med_enrich:.3g}, median nearest-distance/norm={med_nn:.3g}",
                "支持" if med_enrich > 1.2 else "较弱",
                "这是位置级证据：强 Pz 区和强 h-change 区是否同位或邻近；仍不是因果证明。",
            )
        if pxy_ranking_df is not None and not pxy_ranking_df.empty:
            clean_rank = pxy_ranking_df[pxy_ranking_df["group"] == "clean"].sort_values("rank")
            if not clean_rank.empty:
                top = clean_rank.iloc[0]
                shear = clean_rank[clean_rank["pair"] == "Pxy_vs_shear_gradient"]
                shear_rank = int(shear.iloc[0]["rank"]) if not shear.empty else -1
                shear_val = float(shear.iloc[0]["median_enrichment"]) if not shear.empty else np.nan
                add(
                    "Pxy mechanical-template ranking",
                    f"clean layers best={str(top['pair'])} enrichment={float(top['median_enrichment']):.3g}; shear-gradient rank={shear_rank}, enrichment={shear_val:.3g}",
                    "支持 shear-gradient" if shear_rank == 1 and np.isfinite(shear_val) and shear_val > 1.2 else "不支持单一 shear-gradient 主导",
                    "该项回答 Pxy 强区位置是否与剪切应变梯度强区同位；若 shear-gradient 不领先，不能用幅值/位置来证明它。",
                )
        if pxy_direction_ranking_df is not None and not pxy_direction_ranking_df.empty:
            clean_dir = pxy_direction_ranking_df[pxy_direction_ranking_df["group"] == "clean"].sort_values("rank")
            if not clean_dir.empty:
                topd = clean_dir.iloc[0]
                shear_dir = clean_dir[clean_dir["vector_template"] == "shear_gradient_literature"]
                shear_dir_rank = int(shear_dir.iloc[0]["rank"]) if not shear_dir.empty else -1
                shear_dir_val = float(shear_dir.iloc[0]["median_abs_cos_high"]) if not shear_dir.empty else np.nan
                add(
                    "Pxy direction-template ranking",
                    f"clean layers best={str(topd['vector_template'])} median |cos|={float(topd['median_abs_cos_high']):.3g}; literature shear-gradient direction rank={shear_dir_rank}, median |cos|={shear_dir_val:.3g}",
                    "支持 shear-gradient 方向模板" if shear_dir_rank == 1 and np.isfinite(shear_dir_val) and shear_dir_val > 0.55 else "不支持单一 shear-gradient 方向主导",
                    "该项回答 Pxy 方向是否与候选 vector template 对齐；它和位置/幅值证据互补。",
                )
        if pxy_nature_ranking_df is not None and not pxy_nature_ranking_df.empty:
            clean_nat = pxy_nature_ranking_df[pxy_nature_ranking_df["group"] == "clean"].sort_values("rank")
            if not clean_nat.empty:
                topn = clean_nat.iloc[0]
                comp = clean_nat[clean_nat["test"] == "component_sign_literature"]
                comp_rank = int(comp.iloc[0]["rank"]) if not comp.empty else -1
                comp_val = float(comp.iloc[0]["median_value"]) if not comp.empty else np.nan
                add(
                    "Pxy Nature-style evidence ranking",
                    f"clean layers best={str(topn['test'])} value={float(topn['median_value']):.3g}; component-sign rank={comp_rank}, value={comp_val:.3g}",
                    "支持文献式 Pxy-template 对应" if comp_rank > 0 and comp_rank <= 2 and np.isfinite(comp_val) and comp_val >= 0.60 else "文献式 Pxy-template 证据不足",
                    "该项不看幅值线性，而看分量符号、方向、curl/chirality 等更接近文献图像的对应方式。",
                )

    return pd.DataFrame(rows)


def build_time_path_summary(frame_df: pd.DataFrame) -> pd.DataFrame:
    """生成最直观的时间路径表，用来检查 P0_hold 增强是否连续、是否真实。

    这个表只保留核心通道：Pxy、Pz、uz/grad_uz 和 stress proxy。
    """
    if frame_df.empty:
        return pd.DataFrame()
    keep = [
        "step", "stage",
        "clean_dPxy_rms_mean", "clean_dPz_rms_mean", "clean_grad_uz_rms_mean", "clean_abs_dh_upper_rms_mean", "clean_grad_h_upper_rms_mean", "clean_uz_rms_mean", "clean_szz_mean",
        "layer5_dPxy_rms_mean", "layer5_dPz_rms_mean", "layer5_grad_uz_rms_mean", "layer5_abs_dh_upper_rms_mean", "layer5_grad_h_upper_rms_mean", "layer5_uz_rms_mean", "layer5_szz_mean",
        "analysis_dPxy_rms_mean", "analysis_dPz_rms_mean",
    ]
    keep = [c for c in keep if c in frame_df.columns]
    return frame_df[keep].copy()


def build_geometry_channel_summary(layer_df: pd.DataFrame) -> pd.DataFrame:
    """基于 Ti-layer 的 z_mean / z_std 构造简化 vertical geometry summary。

    这里先不声称它就是 buried-interface local gap，只作为 layer spacing / corrugation proxy。
    真正的 opposite-layer interpolated local gap 可作为下一轮更严格版本。
    """
    if layer_df.empty or "z_mean" not in layer_df.columns:
        return pd.DataFrame()
    rows = []
    for step, g in layer_df.groupby("step"):
        gg = g.sort_values("layer").copy()
        z0_by_layer = {}
        ref = layer_df[layer_df["step"] == layer_df["step"].min()].set_index("layer")
        for _, row in gg.iterrows():
            li = int(row["layer"])
            upper = gg[gg["layer"] == li + 1]
            gap_to_upper = np.nan
            gap_to_upper_ref = np.nan
            if not upper.empty:
                gap_to_upper = float(upper.iloc[0]["z_mean"] - row["z_mean"])
                if li in ref.index and (li + 1) in ref.index:
                    gap_to_upper_ref = float(ref.loc[li + 1, "z_mean"] - ref.loc[li, "z_mean"])
            rows.append({
                "step": int(step),
                "stage": str(row.get("stage", stage_label(int(step), StageParams()))),
                "layer": li,
                "z_mean": float(row["z_mean"]),
                "z_std_corrugation_proxy": float(row.get("z_std", np.nan)),
                "uz_rms": float(row.get("uz_rms", np.nan)),
                "grad_uz_mag_rms": float(row.get("grad_uz_mag_rms", np.nan)),
                "gap_to_upper": gap_to_upper,
                "gap_to_upper_change_from_initial": gap_to_upper - gap_to_upper_ref if np.isfinite(gap_to_upper) and np.isfinite(gap_to_upper_ref) else np.nan,
                "note": "z_std 和 gap_to_upper 是 vertical geometry proxy；不是严格 buried-interface local gap",
            })
    return pd.DataFrame(rows).sort_values(["step", "layer"]) if rows else pd.DataFrame()


def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return df[col].to_numpy(float)


def _stage_tick_labels(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    steps = df["step"].to_numpy(int)
    labels = []
    line_break = chr(10)
    for _, r in df.iterrows():
        st = str(r.get("stage", ""))
        if st in ["initial", "compressed_hold", "unload", "P0_hold", "first_opening_ramp", "first_opening_hold"]:
            labels.append(str(int(r["step"])) + line_break + st)
        else:
            labels.append(str(int(r["step"])))
    return steps, labels


def plot_stage_path_figure(time_df: pd.DataFrame, out_dir: Path) -> None:
    """图 1：阶段路径图。证明 Pxy/Pz 是否在 hold 后继续演化。"""
    if time_df.empty:
        return
    ensure_dir(out_dir)
    steps = time_df["step"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(steps, _safe_col(time_df, "clean_dPxy_rms_mean"), marker="o", label="clean |dPxy|")
    ax.plot(steps, _safe_col(time_df, "clean_dPz_rms_mean"), marker="s", label="clean |dPz|")
    ax.plot(steps, _safe_col(time_df, "layer5_dPxy_rms_mean"), marker="^", label="layer 5 |dPxy|")
    ax.plot(steps, _safe_col(time_df, "layer5_dPz_rms_mean"), marker="v", label="layer 5 |dPz|")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMS polarization change (C/m$^2$)")
    ax.set_title("Stage path of Pxy and Pz redistribution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "Fig_stage_path_Pxy_Pz.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_layer_ladder_figure(layer_df: pd.DataFrame, selected: Dict[str, Optional[int]], out_dir: Path) -> None:
    """图 2：层分辨 ladder 图。突出 layer 5 与 clean layers 的大小关系。"""
    if layer_df.empty:
        return
    step = selected.get("P0_hold_end") or selected.get("compressed_hold_end") or int(layer_df["step"].max())
    g = layer_df[layer_df["step"] == step].sort_values("layer")
    if g.empty:
        return
    ensure_dir(out_dir)
    layers = g["layer"].to_numpy(int)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    x = np.arange(len(layers))
    ax.bar(x - width/2, g["dPxy_mag_C_m2_rms"].to_numpy(float), width=width, label="|dPxy|")
    ax.bar(x + width/2, g["dPz_C_m2_rms"].to_numpy(float), width=width, label="|dPz|")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("RMS polarization change (C/m$^2$)")
    ax.set_title(f"Layer-resolved Pxy/Pz redistribution at step {step}")
    ax.legend(frameon=False)
    for xi, li in zip(x, layers):
        if li == BACKGROUND_LAYER:
            ax.text(xi, ax.get_ylim()[1] * 0.92, "L5", ha="center", va="top", fontsize=8)
        elif li in CLEAN_LAYERS:
            ax.text(xi, ax.get_ylim()[1] * 0.82, "clean", ha="center", va="top", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "Fig_layer_ladder_Pxy_Pz.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_vertical_channel_figure(time_df: pd.DataFrame, out_dir: Path) -> None:
    """图 3：Pz 与 vertical geometry proxy 的阶段路径。"""
    if time_df.empty:
        return
    ensure_dir(out_dir)
    steps = time_df["step"].to_numpy(int)
    fig, ax1 = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax1.plot(steps, _safe_col(time_df, "clean_dPz_rms_mean"), marker="o", label="clean |dPz|")
    ax1.plot(steps, _safe_col(time_df, "layer5_dPz_rms_mean"), marker="s", label="layer 5 |dPz|")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("|dPz| RMS (C/m$^2$)")
    ax2 = ax1.twinx()
    ax2.plot(steps, _safe_col(time_df, "clean_grad_uz_rms_mean"), marker="^", linestyle="--", label="clean |grad uz|")
    ax2.plot(steps, _safe_col(time_df, "layer5_grad_uz_rms_mean"), marker="v", linestyle="--", label="layer 5 |grad uz|")
    ax2.set_ylabel("|grad uz| RMS")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("Vertical channel: Pz and out-of-plane geometry proxy")
    ax1.grid(True, alpha=0.25)
    fig.savefig(out_dir / "Fig_vertical_channel_Pz_graduz.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _clip_values(v: np.ndarray, percentile: float = 98.0) -> Tuple[float, float]:
    vv = np.asarray(v, float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return 0.0, 1.0
    hi = float(np.nanpercentile(np.abs(vv), percentile))
    if hi <= 1e-12:
        hi = float(np.nanmax(np.abs(vv))) if vv.size else 1.0
    return 0.0, hi if hi > 0 else 1.0


def plot_spatial_mechanism_figure(selected_cell_df: pd.DataFrame, selected: Dict[str, Optional[int]], out_dir: Path) -> None:
    """图 4：空间图，只展示 layer 5 与一个 clean layer 的 |dPxy|、|dPz|、|grad_uz|。"""
    if selected_cell_df.empty:
        return
    step = selected.get("P0_hold_end") or selected.get("compressed_hold_end")
    if step is None:
        return
    layers = [SPATIAL_CLEAN_LAYER_FOR_FIGURE, BACKGROUND_LAYER]
    g0 = selected_cell_df[(selected_cell_df["step"] == step) & (selected_cell_df["layer"].isin(layers))].copy()
    if g0.empty:
        return
    ensure_dir(out_dir)
    fields = [
        ("dPxy_mag_C_m2", "|dPxy|"),
        ("abs_dPz", "|dPz|"),
        ("grad_uz_mag", "|grad uz|"),
    ]
    g0["abs_dPz"] = np.abs(g0["dPz_C_m2"].to_numpy(float))
    fig, axes = plt.subplots(len(layers), len(fields), figsize=(9.5, 5.8), constrained_layout=True)
    if len(layers) == 1:
        axes = np.asarray([axes])
    for i, li in enumerate(layers):
        gg = g0[g0["layer"] == li]
        if len(gg) < 4:
            continue
        tri = mtri.Triangulation(gg["x_ref"].to_numpy(float), gg["y_ref"].to_numpy(float))
        for j, (col, title) in enumerate(fields):
            ax = axes[i, j]
            vals = gg[col].to_numpy(float)
            vmin, vmax = _clip_values(vals)
            im = ax.tripcolor(tri, np.clip(vals, vmin, vmax), shading="gouraud", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{li} {title}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            if col == "dPxy_mag_C_m2":
                idx = np.arange(len(gg))[::max(1, len(gg)//80)]
                ax.quiver(
                    gg["x_ref"].to_numpy(float)[idx],
                    gg["y_ref"].to_numpy(float)[idx],
                    gg["dPx_C_m2"].to_numpy(float)[idx],
                    gg["dPy_C_m2"].to_numpy(float)[idx],
                    angles="xy", scale_units="xy", scale=None, width=0.0025
                )
    fig.suptitle(f"Spatial redistribution at step {step}: clean layer vs layer 5", fontsize=11)
    fig.savefig(out_dir / "Fig_spatial_clean_vs_layer5.png", dpi=FIGURE_DPI)
    plt.close(fig)


def generate_mechanism_figures(time_df: pd.DataFrame, layer_df: pd.DataFrame, selected_cell_df: pd.DataFrame, selected: Dict[str, Optional[int]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    plot_stage_path_figure(time_df, out_dir)
    plot_layer_ladder_figure(layer_df, selected, out_dir)
    plot_vertical_channel_figure(time_df, out_dir)
    plot_spatial_mechanism_figure(selected_cell_df, selected, out_dir)


def plot_layer_map(cell_df: pd.DataFrame, step: int, layer: int, out_dir: Path) -> None:
    g = cell_df[(cell_df["step"] == step) & (cell_df["layer"] == layer)].copy()
    if len(g) < 4:
        return
    x = g["x_ref"].to_numpy(float)
    y = g["y_ref"].to_numpy(float)
    tri = mtri.Triangulation(x, y)
    fields = [
        ("dPxy_mag_C_m2", "|dPxy|"),
        ("eps_xy", "eps_xy"),
        ("omega_z", "omega_z"),
        ("sigma_zz_GPa", "sigma_zz_GPa"),
        ("curl_dPxy", "curl_z(dPxy)"),
        ("grad_uz_mag", "|grad uz|"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, fields):
        val = g[col].to_numpy(float)
        im = ax.tripcolor(tri, val, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"step {step}, layer {layer}")
    fig.savefig(out_dir / f"maps_step{step}_L{layer:02d}.png", dpi=180)
    plt.close(fig)

    # dPxy 矢量图
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mag = g["dPxy_mag_C_m2"].to_numpy(float)
    im = ax.tripcolor(tri, mag, shading="gouraud")
    idx = np.arange(len(g))[::max(1, len(g)//120)]
    ax.quiver(x[idx], y[idx], g["dPx_C_m2"].to_numpy(float)[idx], g["dPy_C_m2"].to_numpy(float)[idx])
    ax.set_aspect("equal")
    ax.set_title(f"dPxy vector, step {step}, layer {layer}")
    fig.colorbar(im, ax=ax, label="|dPxy| C/m2")
    fig.savefig(out_dir / f"dPxy_vector_step{step}_L{layer:02d}.png", dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, root: Path, params: StageParams, snapshots: Sequence[DumpSnapshot], ref_grid: ReferenceGrid, selected: Dict[str, int]) -> None:
    lines = []
    lines.append("# tensor_dump 机制分析报告")
    lines.append("")
    lines.append("## 输入信息")
    lines.append(f"- root: `{root}`")
    lines.append(f"- 读取 frame 数: {len(snapshots)}")
    lines.append(f"- 第一个 step: {snapshots[0].step}")
    lines.append(f"- 最后一个 step: {snapshots[-1].step}")
    lines.append(f"- Ti-centered cell 数量: {len(ref_grid.ti_ids)}")
    lines.append(f"- 选中的关键 step: {selected}")
    lines.append("")
    lines.append("## 相比只读 PLT 后处理结果的改进")
    lines.append("- 本脚本直接读取新版 `tensor_dump` 原子级 dump，而不是只读取 `Pol_grid_ALL_layers.plt`。")
    lines.append("- displacement 优先使用 `c_dis[1:3]`；如果没有该字段，则退回使用 `xu/yu/zu - reference`。")
    lines.append("- stress 使用 LAMMPS 输出的 raw `c_stress[1:6]`，并通过显式 Ti-cell volume 约定转换为 local stress proxy；在 units metal 中，除以体积后为 bar，再乘 1e-4 转为 GPa。")
    lines.append("- polarization 由 core-shell 电荷和 Ti-centered perovskite 邻居重新计算，再与 `Pxy`、`dPxy`、`Pz` 和 topology proxies 一起比较。")
    lines.append("")
    lines.append("## 重要限制")
    lines.append("- `c_stress[1:6]` 是真实 MD virial 输出，但 local stress 转换为 GPa 时必须引入体积约定；这里采用 layer thickness × in-plane area per Ti cell。注意：units metal 下除以体积后是 bar，不是 eV/A^3。")
    lines.append("- local strain 是 Ti-grid small-displacement-gradient proxy，不是完整有限变形应变张量。")
    lines.append("- Ti-centered polarization 重构应与旧 `polarization_20260514.py` 输出交叉检查后，再做定量判断。")
    lines.append("- layer 5 应保持解释为 `background + redistribution`，不能写成 clean h-gradient / shear-gradient layer。")
    lines.append("- first opening 只是 crossover 检查，不是 buried-interface 主机制证明窗口。")
    lines.append("")
    lines.append("## 输出文件阅读建议")
    lines.append("- `cell_fields/cell_fields_step*.csv`: Ti-cell resolved stress/strain/polarization 字段。")
    lines.append("- `core_mechanism_summary.csv`: 最精简的机制判断总表，默认最先看这个。")
    lines.append("- `frame_summary.csv`: 每个 step 的 analysis/clean/layer5/medium/weak 分组趋势，包含 Pxy 与 Pz。")
    lines.append("- `vertical_channel_summary.csv`: 专门检查 Pz vertical/normal channel 及其与 uz/grad_uz 的阶段趋势。")
    lines.append("- `time_path_summary.csv`: validation/figure/full 模式下输出，用于检查 P0_hold 增强是否沿时间连续演化。")
    lines.append("- `time_continuity_evidence.csv`: 检查 P0_hold 增强是否像连续重排，而不是单帧跳变。")
    lines.append("- `geometry_channel_summary.csv`: validation/figure/full 模式下输出，用 z_mean/z_std/gap_to_upper 做简化 vertical geometry 检查。")
    lines.append("- `mechanism_evidence_summary.csv`: 针对 Pz-h、Pxy-template、layer-dependent amplification、P0 delayed redistribution 的猜想逐条给出计算证据。")
    lines.append("- `spatial_evidence_summary.csv`: 位置级证据表，检查高 |dPz| 是否和高 |dh| 同位/邻近，高 |dPxy| 是否和 shear-gradient/rotation/vorticity/stress 等模板同位/邻近。")
    lines.append("- `pxy_template_ranking.csv`: Pxy 候选 mechanical template 的位置重合排名，用于判断 shear-gradient 强区是否真的领先。")
    lines.append("- `pxy_direction_evidence.csv`: Pxy 方向与候选 vector template 的对齐证据，不只看大小或位置。")
    lines.append("- `pxy_direction_ranking.csv`: Pxy 候选方向模板排名，用于判断 shear-gradient 是否可能作为方向模板。")
    lines.append("- `pxy_nature_style_evidence.csv`: 更接近 Nature 文献图像的 Pxy 证据：component sign、vector direction、curl/chirality。")
    lines.append("- `pxy_nature_style_ranking.csv`: Nature-style Pxy 证据排名，用于判断 shear-gradient 是否仍可作为非幅值意义上的模板。")
    lines.append("- `mechanism_figures/`: validation/figure/full 模式下输出更干净的机制图，包括阶段路径、层分辨 ladder、vertical channel 和空间对比。")
    lines.append("- `layer_summary.csv`: 逐层原始汇总表；默认保留，便于追查。")
    lines.append("- 只有当 `WRITE_DETAILED_TABLES=True` 或 `ANALYSIS_MODE='full'` 时，才会额外写出 stage/magnitude/pattern/vector/model 等诊断表。")
    lines.append("- `plots/` 下的图: 仅用于快速诊断 layer 4/5/6/7 的空间分布，不建议直接作为论文图。")
    (out_dir / "mechanism_tensor_report.md").write_text(chr(10).join(lines), encoding="utf-8")


def build_final_conclusion(
    frame_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    stage_response_df: Optional[pd.DataFrame] = None,
    magnitude_relation_df: Optional[pd.DataFrame] = None,
    pattern_overlap_df: Optional[pd.DataFrame] = None,
    vector_alignment_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """根据输出表格生成简短自动总结。

    这只是运行结束后的粗略提示，不是论文级结论。
    """
    lines: List[str] = []
    if frame_df.empty or layer_df.empty:
        return ["未生成有效 summary；请检查 dump 是否成功读取，以及必要字段是否存在。"]

    final_row = frame_df.iloc[-1]
    final_step = int(final_row["step"])
    final_stage = str(final_row["stage"])
    clean_dpxy = final_row.get("clean_dPxy_rms_mean", np.nan)
    analysis_dpxy = final_row.get("analysis_dPxy_rms_mean", np.nan)
    layer5_dpxy = final_row.get("layer5_dPxy_rms_mean", np.nan)
    clean_szz = final_row.get("clean_szz_mean", np.nan)
    clean_dpz = final_row.get("clean_dPz_rms_mean", np.nan)
    analysis_dpz = final_row.get("analysis_dPz_rms_mean", np.nan)
    layer5_dpz = final_row.get("layer5_dPz_rms_mean", np.nan)
    clean_graduz = final_row.get("clean_grad_uz_rms_mean", np.nan)

    lines.append(f"最终处理到 step = {final_step}，阶段 = {final_stage}。")
    lines.append(f"Pxy 通道：clean layers 平均 |dPxy| RMS = {clean_dpxy:.4g} C/m^2；分析层 layers 2-9 平均 = {analysis_dpxy:.4g} C/m^2；layer 5 = {layer5_dpxy:.4g} C/m^2。")
    lines.append(f"Pz 通道：clean layers 平均 |dPz| RMS = {clean_dpz:.4g} C/m^2；分析层 layers 2-9 平均 = {analysis_dpz:.4g} C/m^2；layer 5 = {layer5_dpz:.4g} C/m^2。")
    lines.append(f"vertical geometry proxy：clean layers 平均 |grad_uz| RMS = {clean_graduz:.4g}；clean layers 平均 sigma_zz proxy = {clean_szz:.4g} GPa。")

    last_layers = layer_df[layer_df["step"] == final_step]
    if not last_layers.empty and "dPxy_mag_C_m2_rms" in last_layers.columns:
        layer5 = last_layers[last_layers["layer"] == BACKGROUND_LAYER]
        clean = last_layers[last_layers["layer"].isin(CLEAN_LAYERS)]
        if not layer5.empty and not clean.empty:
            layer5_val = float(layer5["dPxy_mag_C_m2_rms"].mean())
            clean_val = float(clean["dPxy_mag_C_m2_rms"].mean())
            if np.isfinite(layer5_val) and np.isfinite(clean_val):
                if layer5_val > 1.2 * clean_val:
                    lines.append("最终处理步中 layer 5 的 dPxy redistribution 强于 clean layers，这与“background + redistribution layer”判断一致，不应把 layer 5 当作 clean readout layer。")
                else:
                    lines.append("最终处理步中 layer 5 并没有明显强于 clean layers；在使用强 layer-5 background 结论前，需要先检查空间图。")

    if magnitude_relation_df is not None and not magnitude_relation_df.empty:
        dpxy_mag = magnitude_relation_df[magnitude_relation_df["variable"] == "dPxy_mag_C_m2_rms"].copy()
        l5_comp = dpxy_mag[(dpxy_mag["comparison"] == "layer5_vs_clean") & (dpxy_mag["stage"] == "compressed_hold")]
        l5_p0 = dpxy_mag[(dpxy_mag["comparison"] == "layer5_vs_clean") & (dpxy_mag["stage"] == "P0_hold")]
        clean_res = dpxy_mag[(dpxy_mag["comparison"] == "clean_residual_ratio") & (dpxy_mag["stage"] == "P0_vs_compressed")]
        if not l5_comp.empty:
            lines.append(f"大小关系：compressed_hold 中 layer 5 的 |dPxy| 约为 clean layers 的 {float(l5_comp.iloc[0]['ratio_to_clean']):.3g} 倍，趋势为“{str(l5_comp.iloc[0]['trend_label'])}”。")
        if not l5_p0.empty:
            lines.append(f"大小关系：P0_hold 中 layer 5 的 |dPxy| 约为 clean layers 的 {float(l5_p0.iloc[0]['ratio_to_clean']):.3g} 倍，趋势为“{str(l5_p0.iloc[0]['trend_label'])}”。")
        if not clean_res.empty:
            lines.append(f"残余关系：clean layers 的 P0_hold/compressed_hold |dPxy| 比值约为 {float(clean_res.iloc[0]['ratio_to_clean']):.3g}，判断为“{str(clean_res.iloc[0]['trend_label'])}”。")

        dpz_mag = magnitude_relation_df[magnitude_relation_df["variable"] == "dPz_C_m2_rms"].copy()
        l5_pz = dpz_mag[(dpz_mag["comparison"] == "layer5_vs_clean") & (dpz_mag["stage"] == "P0_hold")]
        clean_pz_res = dpz_mag[(dpz_mag["comparison"] == "clean_residual_ratio") & (dpz_mag["stage"] == "P0_vs_compressed")]
        if not l5_pz.empty:
            lines.append(f"面外 Pz 大小关系：P0_hold 中 layer 5 的 |dPz| 约为 clean layers 的 {float(l5_pz.iloc[0]['ratio_to_clean']):.3g} 倍，趋势为“{str(l5_pz.iloc[0]['trend_label'])}”。")
        if not clean_pz_res.empty:
            lines.append(f"面外 Pz 残余关系：clean layers 的 P0_hold/compressed_hold |dPz| 比值约为 {float(clean_pz_res.iloc[0]['ratio_to_clean']):.3g}，判断为“{str(clean_pz_res.iloc[0]['trend_label'])}”。")

    if stage_response_df is not None and not stage_response_df.empty:
        dpxy_stage = stage_response_df[stage_response_df["variable"] == "dPxy_mag_C_m2_rms"].copy()
        clean_stage = dpxy_stage[dpxy_stage["layer"].isin(CLEAN_LAYERS)]
        if not clean_stage.empty:
            n_loading = int((clean_stage["trend_label"].str.contains("加载响应明显", na=False)).sum())
            n_residual = int((clean_stage["trend_label"].str.contains("残余", na=False)).sum())
            lines.append(f"趋势判断：clean layers 中有 {n_loading}/{len(clean_stage)} 层表现出明显 loading-induced dPxy 响应，其中 {n_residual} 层在 P0_hold 后仍有残余趋势。")
        layer5_stage = dpxy_stage[dpxy_stage["layer"] == BACKGROUND_LAYER]
        if not layer5_stage.empty:
            lines.append(f"layer 5 的 dPxy 阶段趋势：{str(layer5_stage.iloc[0]['trend_label'])}。这比单一相关系数更适合判断 background + redistribution 行为。")

    if pattern_overlap_df is not None and not pattern_overlap_df.empty:
        clean_overlap = pattern_overlap_df[pattern_overlap_df["layer"].isin(CLEAN_LAYERS)].copy()
        if not clean_overlap.empty:
            med_enrich = clean_overlap.groupby("comparator")["enrichment_vs_random"].median(numeric_only=True).sort_values(ascending=False)
            if not med_enrich.empty:
                best_comp = str(med_enrich.index[0])
                best_enrich = float(med_enrich.iloc[0])
                lines.append(f"空间重合判断：clean layers 中 |dPxy| 强区最常与 {best_comp} 强区重合，enrichment ≈ {best_enrich:.3g}。该指标用于趋势判断，不代表逐点线性关系。")

    if vector_alignment_df is not None and not vector_alignment_df.empty:
        clean_vec = vector_alignment_df[vector_alignment_df["layer"].isin(CLEAN_LAYERS)].copy()
        if not clean_vec.empty:
            med_vec = clean_vec.groupby("vector_proxy")["abs_mean_cos_high_dPxy"].median(numeric_only=True).sort_values(ascending=False)
            if not med_vec.empty:
                best_vec = str(med_vec.index[0])
                best_cos = float(med_vec.iloc[0])
                lines.append(f"矢量方向判断：clean layers 的高 |dPxy| 区域中，与 {best_vec} 的中位 |cosθ| ≈ {best_cos:.3g}。方向一致性比单分量相关更适合判断矢量场趋势。")

    if not metrics_df.empty and "corr" in metrics_df.columns:
        clean_metrics = metrics_df[metrics_df["layer"].isin(CLEAN_LAYERS)].copy()
        if not clean_metrics.empty:
            med = clean_metrics.groupby("model")["corr"].median(numeric_only=True).sort_values(ascending=False)
            if not med.empty:
                best_name = str(med.index[0])
                best_corr = float(med.iloc[0])
                lines.append(f"逐点线性相关检查：clean layers 的最高中位相关性来自 {best_name}，corr ≈ {best_corr:.3g}。")
                abs_corr = abs(best_corr) if np.isfinite(best_corr) else np.nan
                if not np.isfinite(abs_corr) or abs_corr < 0.2:
                    lines.append("逐点线性相关很弱：这不否定 moiré-coupled redistribution 机制，只说明不能把 Pxy 响应降成单一局域 scalar proxy。")
                elif abs_corr < 0.5:
                    lines.append("逐点线性相关处于弱到中等水平，只能作为探索性线索，不能作为定量机制证明。")
                else:
                    lines.append("逐点线性相关较高，但仍需结合阶段趋势、空间重合和矢量方向判断，避免过度数值化。")

    return lines


def main() -> int:
    import time
    t0 = time.time()

    root = Path(ROOT_DIR).resolve()
    out_dir = (root / OUT_DIR).resolve() if not Path(OUT_DIR).is_absolute() else Path(OUT_DIR).resolve()
    max_frames = TEST_MAX_FRAMES if TEST_MODE else FORMAL_MAX_FRAMES
    skip_plots = TEST_SKIP_PLOTS if TEST_MODE else FORMAL_SKIP_PLOTS
    save_cell_fields = FORMAL_SAVE_CELL_FIELDS
    stride = FORMAL_STRIDE
    analysis_mode = str(ANALYSIS_MODE).lower().strip()
    write_detailed_tables = bool(WRITE_DETAILED_TABLES or analysis_mode == "full")
    make_validation_outputs = analysis_mode in ["validation", "figure", "full"]
    make_mechanism_figures = bool(GENERATE_MECHANISM_FIGURES and analysis_mode in ["validation", "figure", "full"])

    ensure_dir(out_dir)
    ensure_dir(out_dir / "cell_fields")
    ensure_dir(out_dir / "plots")
    fig_dir = out_dir / FIGURE_DIR
    if make_mechanism_figures:
        ensure_dir(fig_dir)

    print("=" * 72, flush=True)
    print("tensor_dump_mechanism_analysis.py", flush=True)
    print("直接运行模式：只需要修改脚本中的 USER SETTINGS，不需要在命令行输入额外参数。", flush=True)
    print(f"[mode] TEST_MODE = {TEST_MODE}", flush=True)
    print(f"[path] root      = {root}", flush=True)
    print(f"[path] tensor    = {root / DUMP_DIR}", flush=True)
    print(f"[path] output    = {out_dir}", flush=True)
    print(f"[设置] max_frames={max_frames}, stride={stride}, skip_plots={skip_plots}, save_cell_fields={save_cell_fields}", flush=True)
    print(f"[机制判断] analysis_layers={ANALYSIS_LAYERS}, clean_layers={CLEAN_LAYERS}, background_layer={BACKGROUND_LAYER}", flush=True)
    print(f"[机制判断] pointwise_metrics={COMPUTE_POINTWISE_METRICS}, top_fraction={TOP_FRACTION_FOR_PATTERN}, high_response_fraction={HIGH_RESPONSE_FRACTION}", flush=True)
    print(f"[输出设置] mode={analysis_mode}, detailed_tables={write_detailed_tables}, selected_cell_fields={WRITE_SELECTED_CELL_FIELDS}, layer_summary={WRITE_LAYER_SUMMARY}, mechanism_figures={make_mechanism_figures}", flush=True)
    print("=" * 72, flush=True)

    print("[读取] 正在读取 tensor dump snapshots...", flush=True)
    snapshots = read_all_snapshots(root, dump_dir=DUMP_DIR, max_frames=max_frames, stride=stride)
    if len(snapshots) < 2:
        raise RuntimeError("至少需要两个 snapshot：一个参考帧和一个响应帧。")

    ref = snapshots[0]
    params = parse_inby_stage_params(root)
    selected = nearest_existing_steps(
        [s.step for s in snapshots],
        [0, params.load_end, params.comp_hold_end, params.unload_end, params.p0_hold_end, params.opening_ramp_end, params.opening_hold_end],
    )
    selected_steps = sorted({v for v in selected.values() if v is not None})

    print(f"[读取] frames={len(snapshots)}, first={snapshots[0].step}, last={snapshots[-1].step}", flush=True)
    print(f"[selected] {selected}", flush=True)

    print("[网格] 正在建立 Ti-centered reference grid...", flush=True)
    ref_grid = build_reference_grid(ref, n_layers=N_LAYERS)
    print(f"[网格] Ti cells={len(ref_grid.ti_ids)}, layers={N_LAYERS}", flush=True)
    if PRINT_LAYER_GRID:
        for li in range(1, N_LAYERS + 1):
            nli = int(np.sum(ref_grid.layer == li))
            print(f"       layer {li:02d}: nTi={nli}, zmean={ref_grid.layer_means_z[li]:.4f}, Vcell={ref_grid.layer_cell_volume[li]:.4f} A^3", flush=True)

    print("[参考帧] 正在计算参考帧场量...", flush=True)
    ref_cell = build_cell_dataframe(ref, ref, ref_grid, k=KNN_DERIVATIVE)
    ref_cell = add_delta_p_and_topology(ref_cell, ref_cell, ref_grid, k=KNN_DERIVATIVE)
    if COMPUTE_LOCAL_GAP_TEMPLATE:
        ref_cell = add_local_gap_template(ref_cell, k_interp=INTERP_K_FOR_GAP, k_grad=KNN_DERIVATIVE)
        ref_cell = add_delta_gap_template(ref_cell, ref_cell)

    all_layer_summaries = []
    all_cell_selected = []

    total = len(snapshots)
    for iframe, snap in enumerate(snapshots, start=1):
        stage = stage_label(snap.step, params)
        if PRINT_EVERY_FRAME:
            print(f"[frame {iframe:03d}/{total:03d}] step={snap.step}, stage={stage}: 正在计算 stress/strain/polarization...", flush=True)
        cell = build_cell_dataframe(snap, ref, ref_grid, k=KNN_DERIVATIVE)
        cell = add_delta_p_and_topology(cell, ref_cell, ref_grid, k=KNN_DERIVATIVE)
        if COMPUTE_LOCAL_GAP_TEMPLATE:
            cell = add_local_gap_template(cell, k_interp=INTERP_K_FOR_GAP, k_grad=KNN_DERIVATIVE)
            cell = add_delta_gap_template(cell, ref_cell)
        cell["stage"] = stage
        layer_sum = summarize_layer(cell)
        layer_sum["stage"] = stage
        all_layer_summaries.append(layer_sum)

        if save_cell_fields or (WRITE_SELECTED_CELL_FIELDS and snap.step in selected_steps):
            cell.to_csv(out_dir / "cell_fields" / f"cell_fields_step{snap.step}.csv", index=False)
        if snap.step in selected_steps:
            print(f"    [关键帧] 正在保存 step {snap.step} 的 selected-cell 数据和图像", flush=True)
            all_cell_selected.append(cell)
            if not skip_plots:
                for li in MAIN_LAYERS:
                    plot_layer_map(cell, snap.step, li, out_dir / "plots")

    print("[写出] 正在写出 CSV summary 文件...", flush=True)
    layer_df = pd.concat(all_layer_summaries, ignore_index=True)
    if WRITE_LAYER_SUMMARY or write_detailed_tables:
        layer_df.to_csv(out_dir / "layer_summary.csv", index=False)

    frame_df = summarize_frame(layer_df, params)
    frame_df.to_csv(out_dir / "frame_summary.csv", index=False)

    # validation 模式下输出时间路径和简化 vertical geometry 表，用于判断 P0_hold 增强是否连续、是否可能是真实现象。
    time_path_df = build_time_path_summary(frame_df)
    geometry_channel_df = build_geometry_channel_summary(layer_df)
    time_evidence_df = build_time_continuity_evidence(frame_df, params)
    if make_validation_outputs:
        time_path_df.to_csv(out_dir / "time_path_summary.csv", index=False)
        geometry_channel_df.to_csv(out_dir / "geometry_channel_summary.csv", index=False)
        time_evidence_df.to_csv(out_dir / "time_continuity_evidence.csv", index=False)

    # 阶段趋势表：比逐点相关性更重要，用于判断 loading response / recovery / residual。
    # 默认不全部写出，只在 WRITE_DETAILED_TABLES=True 时保存，避免输出文件过多。
    stage_response_df = build_stage_response_summary(layer_df, selected)
    qualitative_summary_df = build_qualitative_mechanism_summary(stage_response_df)
    magnitude_relation_df = build_magnitude_relation_summary(stage_response_df)
    vertical_channel_df = build_vertical_channel_summary(stage_response_df, magnitude_relation_df)
    vertical_channel_df.to_csv(out_dir / "vertical_channel_summary.csv", index=False)
    if write_detailed_tables:
        stage_response_df.to_csv(out_dir / "stage_response_summary.csv", index=False)
        qualitative_summary_df.to_csv(out_dir / "qualitative_mechanism_summary.csv", index=False)
        magnitude_relation_df.to_csv(out_dir / "magnitude_relation_summary.csv", index=False)

    if all_cell_selected:
        selected_cell_df = pd.concat(all_cell_selected, ignore_index=True)
        if WRITE_SELECTED_CELL_FIELDS or write_detailed_tables:
            selected_cell_df.to_csv(out_dir / "selected_cell_fields.csv", index=False)
        metrics = model_metrics(selected_cell_df, selected_steps=selected_steps)
        if write_detailed_tables:
            metrics.to_csv(out_dir / "model_metrics.csv", index=False)

        # 旧的泛化 pattern/vector 诊断默认关闭；当前重点改为机制证据表。
        if COMPUTE_LEGACY_PATTERN_VECTOR or write_detailed_tables:
            pattern_overlap_df = build_pattern_overlap_metrics(selected_cell_df, selected_steps=selected_steps, fraction=TOP_FRACTION_FOR_PATTERN)
            vector_alignment_df = build_vector_alignment_metrics(selected_cell_df, selected_steps=selected_steps)
            if write_detailed_tables:
                pattern_overlap_df.to_csv(out_dir / "pattern_overlap_metrics.csv", index=False)
                vector_alignment_df.to_csv(out_dir / "vector_alignment_metrics.csv", index=False)
        else:
            pattern_overlap_df = pd.DataFrame()
            vector_alignment_df = pd.DataFrame()

        spatial_evidence_df = build_spatial_evidence_summary(selected_cell_df, selected, fraction=EVIDENCE_TOP_FRACTION)
        pxy_ranking_df = build_pxy_template_ranking(spatial_evidence_df)
        if COMPUTE_PXY_DIRECTION_EVIDENCE:
            pxy_direction_df = build_pxy_direction_evidence(selected_cell_df, selected)
            pxy_direction_ranking_df = build_pxy_direction_ranking(pxy_direction_df)
        else:
            pxy_direction_df = pd.DataFrame()
            pxy_direction_ranking_df = pd.DataFrame()
        if COMPUTE_PXY_NATURE_STYLE_EVIDENCE:
            pxy_nature_df = build_pxy_nature_style_evidence(selected_cell_df, selected)
            pxy_nature_ranking_df = build_pxy_nature_style_ranking(pxy_nature_df)
        else:
            pxy_nature_df = pd.DataFrame()
            pxy_nature_ranking_df = pd.DataFrame()
        if WRITE_EVIDENCE_TABLES or make_validation_outputs:
            spatial_evidence_df.to_csv(out_dir / "spatial_evidence_summary.csv", index=False)
            pxy_ranking_df.to_csv(out_dir / "pxy_template_ranking.csv", index=False)
            pxy_direction_df.to_csv(out_dir / "pxy_direction_evidence.csv", index=False)
            pxy_direction_ranking_df.to_csv(out_dir / "pxy_direction_ranking.csv", index=False)
            pxy_nature_df.to_csv(out_dir / "pxy_nature_style_evidence.csv", index=False)
            pxy_nature_ranking_df.to_csv(out_dir / "pxy_nature_style_ranking.csv", index=False)
    else:
        selected_cell_df = pd.DataFrame()
        metrics = pd.DataFrame()
        pattern_overlap_df = pd.DataFrame()
        vector_alignment_df = pd.DataFrame()
        spatial_evidence_df = pd.DataFrame()
        pxy_ranking_df = pd.DataFrame()
        pxy_direction_df = pd.DataFrame()
        pxy_direction_ranking_df = pd.DataFrame()
        pxy_nature_df = pd.DataFrame()
        pxy_nature_ranking_df = pd.DataFrame()
        if write_detailed_tables:
            pd.DataFrame().to_csv(out_dir / "model_metrics.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "pattern_overlap_metrics.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "vector_alignment_metrics.csv", index=False)

    mechanism_evidence_df = build_mechanism_evidence_summary(
        stage_response_df,
        layer_df,
        selected_cell_df,
        time_evidence_df,
        selected,
        spatial_evidence_df=spatial_evidence_df,
        pxy_ranking_df=pxy_ranking_df,
        pxy_direction_ranking_df=pxy_direction_ranking_df,
        pxy_nature_ranking_df=pxy_nature_ranking_df,
    )
    if WRITE_EVIDENCE_TABLES or make_validation_outputs:
        mechanism_evidence_df.to_csv(out_dir / "mechanism_evidence_summary.csv", index=False)

    if make_mechanism_figures:
        print(f"[绘图] 正在生成机制图到 {fig_dir}", flush=True)
        generate_mechanism_figures(time_path_df, layer_df, selected_cell_df, selected, fig_dir)

    core_summary_df = build_core_mechanism_summary(
        frame_df=frame_df,
        stage_df=stage_response_df,
        magnitude_df=magnitude_relation_df,
        vertical_df=vertical_channel_df,
        pattern_df=pattern_overlap_df,
        vector_df=vector_alignment_df,
    )
    core_summary_df.to_csv(out_dir / "core_mechanism_summary.csv", index=False)

    role_rows = []
    for li in ANALYSIS_LAYERS:
        if li in CLEAN_LAYERS:
            role = "clean response/readout layer；主机制证据层"
        elif li == BACKGROUND_LAYER:
            role = "background + redistribution layer；重要但不 clean"
        elif li in [3, 8]:
            role = "medium inheritance；辅助证据层"
        else:
            role = "weak inheritance/boundary；次级检查层"
        role_rows.append({"layer": li, "role": role})
    pd.DataFrame(role_rows).to_csv(out_dir / "layer_roles.csv", index=False)

    final_lines = build_final_conclusion(
        frame_df,
        layer_df,
        metrics,
        stage_response_df=stage_response_df,
        magnitude_relation_df=magnitude_relation_df,
        pattern_overlap_df=pattern_overlap_df,
        vector_alignment_df=vector_alignment_df,
    )
    if not mechanism_evidence_df.empty:
        for _, r in mechanism_evidence_df.iterrows():
            final_lines.append("机制证据 - " + str(r.get("hypothesis", "")) + ": " + str(r.get("verdict", "")) + "；" + str(r.get("computed_evidence", "")))
    write_report(out_dir, root, params, snapshots, ref_grid, selected)
    with (out_dir / "mechanism_tensor_report.md").open("a", encoding="utf-8") as fh:
        fh.write(chr(10) + "## 自动总结结论" + chr(10))
        for line in final_lines:
            fh.write("- " + str(line) + chr(10))

    print("=" * 72, flush=True)
    print("[自动总结结论]", flush=True)
    for line in final_lines:
        print("- " + line, flush=True)
    print(f"[完成] 输出文件已写入 {out_dir}", flush=True)
    print(f"[耗时] elapsed = {time.time() - t0:.1f} s", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
