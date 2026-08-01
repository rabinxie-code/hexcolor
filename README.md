# 100B crop HEX annotation feasibility lab

这是一个可运行的 CPU gold/reference 实现与证据包，用于判断如何给约 100B 个 image crops 生成可解释的主色 HEX。项目实现七条统一基准、运行真实样本的质量/效率测试，并把 3 / 7 / 15 天容量判断发布为交互式报告。

## 结论

当前生产建议不是简单选“最快的一行”，而是：**保留 Adaptive 的 decode-once + cheap stats + 按区域路由架构，flat/mild/gradient 维持 observed 语义，同时把 native Octree 作为 texture palette 的首个吞吐候选，再进入 fused segmented kernel 与 1B-region pilot**。逐 crop CPU 结果只用于比较语义和实现形态，不能直接成为 100B / 3 天部署方案。

- **Adaptive Hex v1**：直接按 `hexcode_extraction_at_scale.html` 的 decode-once、cheap first pass、flat/mild/texture/gradient routing 与紧凑输出设计落地；q12 统计加 64 像素确定性采样，所有主色和 gradient stops 都是源图真实出现过的 sRGB 像素。
- **Hue / HSV 基线**：用户给出的五步草图没有固定 bins、平滑核、低饱和处理或 RGB 回映射。本项目补全为 60H × 4S × 4V、环形 1-2-3-2-1 平滑、4 个灰度桶和 observed-pixel recovery。它适合高饱和、色相语义明确的区域，但并非灰度、多峰纹理和渐变的通用解。Name That Color 的 1500+ 人工命名锚点没有混进像素直方图：对本任务更稳妥的用法是先输出可审计 HEX，再把命名作为可选后处理，而且原始列表包含单独的署名/版权边界。
- **Pixelero RGB histogram**：按文章实现“每个通道 1D 直方图聚类 → 最多 8³ 个 3D bins → 最终 top-3 聚类”。本次 palette coverage 最好，但 centroid 不是 observed pixel。
- **Octree**：用 Pillow 原生 FASTOCTREE 比较 Gervautz–Purgathofer 算法族；它是本次最快的统一实现，但输出为叶均值，因此要先确认下游接受非 observed HEX。
- **Color Thief v3**：官方 3.4.0 + Sharp 原版运行。当前 v3 默认不是“旧版 RGB MMCQ”的简单同义词，而是 quality=10 采样后在缩放 OKLCH 坐标上运行 MMCQ；MIT。速度很高，但只读取约 1/10 像素且忽略近白。
- **pngquant / libimagequant**：通过 imagequant-python 1.1.5 调用 libimagequant 2.15.1 palette/remap 引擎，不计与 HEX 标注无关的 PNG 编码和 dithering；这不是当前 pngquant v3 CLI 的端到端实测。稳定性和 coverage 都较好，生产采用前必须处理 GPL v3+ / 商业许可选择。
- **ColorPipette-inspired**：用 SLIC、确定性对比显著性和 L/C harmony 实现相同研究结构，只用于统一基准；它不是原仓库的 bit-identical 复现，输出也不保证在源图出现。
- **ColorPipette 原版**：实际运行了 BASNet + SpixelNet 的单图 CPU 审计。该项目是和谐 palette 生成器，不是 crop HEX 标注器；双模型、旧运行栈、逐像素 Python 聚合及非 observed palette 语义不适合成为 100B 默认路径。

## 实测范围与结果

真实数据来自一个内部授权的图像数据集：在 1,365 个数字 shard 上做固定种子的分层采样，共使用 10,000 张互不重复的原图；每张只取 1 个 320 × 320 JPEG crop，top-left / center / bottom-right 三种位置均衡轮换。另有 6 个可判定的合成样本。主计时包含 decode 与方法执行；Python 方法使用 Pillow，Color Thief 按官方 Sharp loader。JPEG Q60、50% down/up 稳定性变体不计入延迟。为避免遍历超大 shard，每个被选 shard 只读取固定 64-object 候选窗口后做 SHA-256 排序，因此它是可复现的广覆盖样本，不应宣称为全库严格均匀随机样本。

硬件：Intel Xeon Platinum 8559C，Python 3.12.13，单进程 CPU，Devbox 中 GPU 不可见。

| 方法 | P50 | P95 | 单进程 crops/s | method MPix/s | JPEG ΔE | resize ΔE | palette coverage ΔE | observed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive Hex v1 | 8.05 ms | 15.16 ms | 110.4 | 12.12 | 2.85 | 3.41 | 7.95 | 100% |
| Hue / HSV | 22.39 ms | 31.20 ms | 43.3 | 4.55 | 1.76 | 1.34 | 6.07 | 100% |
| Pixelero RGB histogram | 16.08 ms | 18.30 ms | 62.6 | 6.66 | 1.40 | 2.38 | 4.35 | 0% |
| Octree · native | 1.87 ms | 2.16 ms | 532.6 | 79.29 | 0.55 | 1.36 | 5.12 | 0% |
| Color Thief v3 · OKLCH q10 | 5.89 ms | 6.82 ms | 176.2 | 36.78 | 1.77 | 2.40 | 7.12 | 0% |
| pngquant · libimagequant | 9.43 ms | 19.96 ms | 98.2 | 10.67 | 1.07 | 1.84 | 4.59 | 0% |
| ColorPipette-inspired | 50.11 ms | 54.60 ms | 19.8 | 2.05 | 2.87 | 3.66 | 20.81 | 0% |

Color Thief 的 `quality=10` 是官方默认值，意味着每 10 个像素取 1 个；因此它的吞吐不能当作全像素扫描的等价比较。Octree 的高吞吐来自原生实现，而 Pixelero 与当前 Hue 版本是可审计的 NumPy gold code。所有这些差异都已在报告中显式标注。

ColorPipette 原版的单图审计为：模型加载 1.59 s、首图 1.96 s、热重复 0.52 s、峰值 RSS 1,684 MB，重复 palette 一致。这个单样本结果只说明运行形态，不能与 10,000-crop 基准作统计等价比较。

3 / 7 / 15 天生产场景如下；provisioned 已包含 1.5× safety：

| 完成窗口 | required crops/s | provisioned crops/s | source images/s | source Gpix/s | 压缩输入 | decoded RGB | fused workers @ 0.5 Gpix/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 天 | 385,802 | 578,704 | 19,290 | 19.29 | 9.65 GB/s | 57.87 GB/s | 58 |
| 7 天 | 165,344 | 248,016 | 8,267 | 8.27 | 4.13 GB/s | 24.80 GB/s | 25 |
| 15 天 | 77,160 | 115,741 | 3,858 | 3.86 | 1.93 GB/s | 11.57 GB/s | 12 |

当前逐 crop CPU 实现的等效并发仅用于说明不可直接横向扩容：

| 完成窗口 | Adaptive | Hue | Pixelero | Octree | Color Thief q10 | libimagequant | inspired | ColorPipette 原版 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 天 | 5,240 | 13,367 | 9,242 | 1,087 | 3,285 | 5,895 | 29,263 | 302,371 |
| 7 天 | 2,246 | 5,729 | 3,961 | 466 | 1,408 | 2,527 | 12,542 | 129,588 |
| 15 天 | 1,048 | 2,674 | 1,849 | 218 | 657 | 1,179 | 5,853 | 60,475 |

更合理的生产口径是按 source image decode-once：以上沿用参考方案的 5B source images、平均 20 crops/image、1 MP/image 和 0.5 MB 压缩大小。fused worker 数假设每个 worker 能持续完成 0.5 Gpixel/s 的 decode + segmented aggregation，必须由 1B pilot 的真实 sustained throughput 替换。如果 crops 已物化为独立对象并逐个 GET/decode，则这项 20× 解码摊销不成立。

## 快速使用

环境已有 `cg` conda 时：

```bash
conda run -n cg python -m unittest discover -s tests -v
conda run -n cg python scripts/extract_hex.py data/synthetic/solid.png --method adaptive_v1 --full
conda run -n cg python scripts/extract_hex.py /path/to/authorized/images --method adaptive_v1 --workers 4 --output /tmp/hex.jsonl
```

紧凑 JSONL 默认保存 `rgb24`、`method_id`、route、confidence、`palette_rgb24` 和 `palette_weight_u16`；`hex` 是方便查看/导出的冗余字段。`--full` 会附加 RGB 数组、HEX palette、源图 metadata 与 diagnostics。这个 CLI 用于正确性验证和小规模 pilot，不应直接当成 100B 分布式执行器。

## 交互报告

React 报告包含结论、方法、实现说明、样本、效率和规模化六个页面。报告数据由 benchmark 流程生成到 `site/`，随后通过 Vite 构建到 `dist/`。概念到实现的逐项对照见 `design/FIDELITY_LEDGER.md`。

授权数据集、逐样本结果、生成后的站点文件和环境专用的数据下载工具不会提交到这个公开仓库。运行报告前，请使用自己有权处理的图片和 manifest 生成 `site/data.js`、样本索引及相关图片资源。

## 完整复现

```bash
conda run -n cg pip install -r requirements.txt
npm install
conda run -n cg python -m unittest discover -s tests -v
conda run -n cg python scripts/run_benchmark.py \
  --source-dir /path/to/authorized/images \
  --manifest /path/to/manifest.json
npm run build
```

基准产物默认写入 `data/results/benchmark.json`，六个 Python 方法的完整明细以 JSONL 保存在 `data/results/checkpoints_10k/`，Color Thief 的完整明细在同目录 JSON checkpoint。仪表盘的聚合数据与少量首页装饰记录在 `site/data.js`；全部 crop 通过 `site/sample-index.json`、逐 crop 明细 JSON 和按需加载图片开放浏览，静态站点输出到 `dist/`。这些目录均属于本地生成内容。

原版 ColorPipette 审计需要单独克隆上游仓库、取得其模型权重并应用 `third_party/colorpipette_compat.patch`，再运行：

```bash
conda run -n cg python scripts/audit_colorpipette_original.py --repo /tmp/hex-colorpipette-source
```

兼容补丁仅做现代运行环境适配：当前 scikit-image connectivity 扩展、禁止重复下载 torchvision 权重、允许加载可信旧 checkpoint，以及在隔离审计中 stub 未使用的 Flask-CORS wiring。审计观察到上游 NumPy uint8 在 saliency 平均和 Lab→LCh 中发生溢出/下溢警告；上游仓库根目录没有 license 文件，因此这里没有复制其源码。

## 代码地图

- `hexbench/methods.py`：六种 Python 统一接口实现（Adaptive、Hue、Pixelero、Octree、libimagequant、ColorPipette-inspired）。
- `scripts/benchmark_colorthief.mjs`：官方 Color Thief v3 的 Sharp decode + OKLCH/MMCQ 精确运行适配。
- `hexbench/batch.py`、`scripts/extract_hex.py`：单图/目录批量标注 API 与 CLI。
- `hexbench/benchmark.py`：端到端计时、稳定性、coverage、scale projection。
- `hexbench/browser_export.py`：把 10k checkpoint 导出为轻量索引、逐 crop 七方法明细和按需加载图片。
- `scripts/audit_colorpipette_original.py`：明确限定为单图的上游原版审计。
- `src/`：六页 React 证据报告；`design/` 保存设计规范与 fidelity ledger。

## 指标边界

- Oklab ΔE 在 JSON 中以方便阅读的 ×100 标度报告；值越低代表颜色更接近。
- JPEG/resize ΔE 只衡量输出稳定，不代表语义正确。
- palette coverage 是像素到最近 palette 色的平均距离，偏好覆盖全图，不等同人工审美偏好。
- 10,000 个独立 source crops 已足以让 P50/P95/P99、路由占比和平均稳定性不再由几十张图主导，但仍不足以估计 100B 数据分布的极端尾部，也没有替代严格均匀随机采样。正式 Go/No-Go 仍需要顺序 shard、decode-once、批内统计、去重/幂等、失败重试和 1B-region pilot。
