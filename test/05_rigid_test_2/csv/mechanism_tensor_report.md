# tensor_dump 机制分析报告

## 输入信息
- root: `/home/xbtian/boyu/magic_angle/11.test`
- 读取 frame 数: 33
- 第一个 step: 0
- 最后一个 step: 160000
- Ti-centered cell 数量: 4880
- 选中的关键 step: {'initial': 0, 'load_end': 30000, 'compressed_hold_end': 80000, 'unload_end': 110000, 'P0_hold_end': 160000, 'first_opening_ramp_end': None, 'first_opening_hold_end': None}

## 相比只读 PLT 后处理结果的改进
- 本脚本直接读取新版 `tensor_dump` 原子级 dump，而不是只读取 `Pol_grid_ALL_layers.plt`。
- displacement 优先使用 `c_dis[1:3]`；如果没有该字段，则退回使用 `xu/yu/zu - reference`。
- stress 使用 LAMMPS 输出的 raw `c_stress[1:6]`，并通过显式 Ti-cell volume 约定转换为 local stress proxy；在 units metal 中，除以体积后为 bar，再乘 1e-4 转为 GPa。
- polarization 由 core-shell 电荷和 Ti-centered perovskite 邻居重新计算，再与 `Pxy`、`dPxy`、`Pz` 和 topology proxies 一起比较。

## 重要限制
- `c_stress[1:6]` 是真实 MD virial 输出，但 local stress 转换为 GPa 时必须引入体积约定；这里采用 layer thickness × in-plane area per Ti cell。注意：units metal 下除以体积后是 bar，不是 eV/A^3。
- local strain 是 Ti-grid small-displacement-gradient proxy，不是完整有限变形应变张量。
- Ti-centered polarization 重构应与旧 `polarization_20260514.py` 输出交叉检查后，再做定量判断。
- layer 5 应保持解释为 `background + redistribution`，不能写成 clean h-gradient / shear-gradient layer。
- first opening 只是 crossover 检查，不是 buried-interface 主机制证明窗口。

## 输出文件阅读建议
- `cell_fields/cell_fields_step*.csv`: Ti-cell resolved stress/strain/polarization 字段。
- `core_mechanism_summary.csv`: 最精简的机制判断总表，默认最先看这个。
- `frame_summary.csv`: 每个 step 的 analysis/clean/layer5/medium/weak 分组趋势，包含 Pxy 与 Pz。
- `vertical_channel_summary.csv`: 专门检查 Pz vertical/normal channel 及其与 uz/grad_uz 的阶段趋势。
- `time_path_summary.csv`: validation/figure/full 模式下输出，用于检查 P0_hold 增强是否沿时间连续演化。
- `time_continuity_evidence.csv`: 检查 P0_hold 增强是否像连续重排，而不是单帧跳变。
- `geometry_channel_summary.csv`: validation/figure/full 模式下输出，用 z_mean/z_std/gap_to_upper 做简化 vertical geometry 检查。
- `mechanism_evidence_summary.csv`: 针对 Pz-h、Pxy-template、layer-dependent amplification、P0 delayed redistribution 的猜想逐条给出计算证据。
- `spatial_evidence_summary.csv`: 位置级证据表，检查高 |dPz| 是否和高 |dh| 同位/邻近，高 |dPxy| 是否和 shear-gradient/rotation/vorticity/stress 等模板同位/邻近。
- `pxy_template_ranking.csv`: Pxy 候选 mechanical template 的位置重合排名，用于判断 shear-gradient 强区是否真的领先。
- `pxy_direction_evidence.csv`: Pxy 方向与候选 vector template 的对齐证据，不只看大小或位置。
- `pxy_direction_ranking.csv`: Pxy 候选方向模板排名，用于判断 shear-gradient 是否可能作为方向模板。
- `pxy_nature_style_evidence.csv`: 更接近 Nature 文献图像的 Pxy 证据：component sign、vector direction、curl/chirality。
- `pxy_nature_style_ranking.csv`: Nature-style Pxy 证据排名，用于判断 shear-gradient 是否仍可作为非幅值意义上的模板。
- `mechanism_figures/`: validation/figure/full 模式下输出更干净的机制图，包括阶段路径、层分辨 ladder、vertical channel 和空间对比。
- `layer_summary.csv`: 逐层原始汇总表；默认保留，便于追查。
- 只有当 `WRITE_DETAILED_TABLES=True` 或 `ANALYSIS_MODE='full'` 时，才会额外写出 stage/magnitude/pattern/vector/model 等诊断表。
- `plots/` 下的图: 仅用于快速诊断 layer 4/5/6/7 的空间分布，不建议直接作为论文图。
## 自动总结结论
- 最终处理到 step = 160000，阶段 = P0_hold。
- Pxy 通道：clean layers 平均 |dPxy| RMS = 0.04173 C/m^2；分析层 layers 2-9 平均 = 0.08839 C/m^2；layer 5 = 0.5121 C/m^2。
- Pz 通道：clean layers 平均 |dPz| RMS = 0.1404 C/m^2；分析层 layers 2-9 平均 = 0.133 C/m^2；layer 5 = 0.5103 C/m^2。
- vertical geometry proxy：clean layers 平均 |grad_uz| RMS = 9.195；clean layers 平均 sigma_zz proxy = nan GPa。
- 最终处理步中 layer 5 的 dPxy redistribution 强于 clean layers，这与“background + redistribution layer”判断一致，不应把 layer 5 当作 clean readout layer。
- 大小关系：compressed_hold 中 layer 5 的 |dPxy| 约为 clean layers 的 12.2 倍，趋势为“明显更强”。
- 大小关系：P0_hold 中 layer 5 的 |dPxy| 约为 clean layers 的 12.3 倍，趋势为“明显更强”。
- 残余关系：clean layers 的 P0_hold/compressed_hold |dPxy| 比值约为 1.88，判断为“P0_hold 后继续增强/重排”。
- 面外 Pz 大小关系：P0_hold 中 layer 5 的 |dPz| 约为 clean layers 的 3.64 倍，趋势为“明显更强”。
- 面外 Pz 残余关系：clean layers 的 P0_hold/compressed_hold |dPz| 比值约为 4.04，判断为“P0_hold 后继续增强/重排”。
- 趋势判断：clean layers 中有 3/3 层表现出明显 loading-induced dPxy 响应，其中 3 层在 P0_hold 后仍有残余趋势。
- layer 5 的 dPxy 阶段趋势：加载响应明显，卸载后残余较强。这比单一相关系数更适合判断 background + redistribution 行为。
- 机制证据 - Pz vertical-gap channel: 支持；clean P0/compressed |dPz|=4.04, |dh_upper|=1.49, |grad_h_upper|=1.35
- 机制证据 - Layer-dependent amplification instead of strain-copy: 支持；P0 layer5/clean ratios: |dPxy|=12.3, |dPz|=3.64, |grad_uz|=1.09
- 机制证据 - P0_hold delayed redistribution: 需要检查；clean_dPxy: jump_fraction=0.846, 可能有跳变或非单调; clean_dPz: jump_fraction=0.876, 可能有跳变或非单调; layer5_dPxy: jump_fraction=1.55, 可能有跳变或非单调; layer5_dPz: jump_fraction=0.718, 连续增强倾向
- 机制证据 - Spatial Pz-h positional match: 较弱；top |dPz| with top |dh_upper|: median enrichment=0.842, median nearest-distance/norm=0.0321
- 机制证据 - Pxy mechanical-template ranking: 不支持单一 shear-gradient 主导；clean layers best=Pxy_vs_grad_h_lower enrichment=2.12; shear-gradient rank=7, enrichment=0.918
- 机制证据 - Pxy direction-template ranking: 不支持单一 shear-gradient 方向主导；clean layers best=grad_h_best median |cos|=0.95; literature shear-gradient direction rank=7, median |cos|=0.679
- 机制证据 - Pxy Nature-style evidence ranking: 文献式 Pxy-template 证据不足；clean layers best=vector_shear_gradient_direct value=0.695; component-sign rank=3, value=0.582
