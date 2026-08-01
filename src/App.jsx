import React, { useEffect, useMemo, useState } from 'react';

const data = window.BENCHMARK_DATA;
const SAMPLE_CACHE_VERSION = encodeURIComponent(data?.generated_at || 'latest');
let sampleIndexPromise;
const sampleRecordPromises = new Map();

function fetchJson(path) {
  return fetch(`${path}?v=${SAMPLE_CACHE_VERSION}`).then((response) => {
    if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
    return response.json();
  });
}

function loadSampleIndex() {
  if (!sampleIndexPromise) sampleIndexPromise = fetchJson('/sample-index.json');
  return sampleIndexPromise;
}

function loadSampleRecord(path) {
  if (!sampleRecordPromises.has(path)) sampleRecordPromises.set(path, fetchJson(`/${path}`));
  return sampleRecordPromises.get(path);
}

const NAV_ITEMS = [
  ['overview', '结论'],
  ['methods', '方法'],
  ['implementation', '说明'],
  ['samples', '样本'],
  ['efficiency', '效率'],
  ['scale', '规模化'],
];

const METHOD_ORDER = [
  'adaptive_v1',
  'tencent_hsv',
  'pixelero_rgb_hist',
  'octree',
  'color_thief_v3',
  'pngquant_liq',
  'colorpipette_inspired',
];
const METHOD_LABELS = {
  adaptive_v1: 'Adaptive Hex v1',
  tencent_hsv: '腾讯 HSV 直方图',
  pixelero_rgb_hist: 'Pixelero RGB 直方图',
  octree: 'Octree · native',
  color_thief_v3: 'Color Thief v3 · OKLCH',
  pngquant_liq: 'pngquant · libimagequant',
  colorpipette_inspired: 'ColorPipette-inspired',
};
const ROUTE_LABELS = {
  flat: 'flat',
  mild: 'mild',
  texture: 'texture',
  gradient: 'gradient',
  hsv_histogram: 'HSV',
  achromatic_histogram: 'achromatic',
  channel_histogram_kmeans: 'RGB hist + k-means',
  fast_octree_native: 'fast octree',
  oklch_mmcq_q10: 'OKLCH + MMCQ q10',
  libimagequant_palette: 'libimagequant',
  saliency_superpixels_harmony: 'saliency + superpixels',
};
const ROUTE_COLORS = {
  flat: '#0874ed',
  mild: '#25a9ad',
  texture: '#6546ca',
  gradient: '#ff7a1a',
};

const fmt = (value, digits = 2) =>
  Number.isFinite(Number(value))
    ? Number(value).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
    : '—';

const fmtCompact = (value, digits = 1) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  if (Math.abs(number) >= 1e6) return `${fmt(number / 1e6, digits)}M`;
  if (Math.abs(number) >= 1e3) return `${fmt(number / 1e3, digits)}K`;
  return fmt(number, digits);
};

function ChevronIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="icon icon-chevron">
      <path d="m4 6 4 4 4-4" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 18 18" aria-hidden="true" className="icon">
      <path d="M9 2v9m0 0 3.5-3.5M9 11 5.5 7.5M3 15h12" />
    </svg>
  );
}

function TerminalIcon() {
  return (
    <svg viewBox="0 0 18 18" aria-hidden="true" className="icon">
      <path d="m3 4 4 4-4 4m6 1h6" />
    </svg>
  );
}

function SortIcon() {
  return (
    <svg viewBox="0 0 12 16" aria-hidden="true" className="sort-icon">
      <path d="m3 6 3-3 3 3M9 10l-3 3-3-3" />
    </svg>
  );
}

function ScopePanel({ onClose }) {
  const original = data.external_audit.colorpipette.original_cpu_audit;
  return (
    <div className="scope-panel" role="dialog" aria-label="数据口径">
      <div className="scope-panel-head">
        <strong>数据口径</strong>
        <button type="button" onClick={onClose} aria-label="关闭数据口径">×</button>
      </div>
      <dl>
        <div><dt>真实数据</dt><dd>{data.scope.source_images.toLocaleString()} 张独立 S3 原图 / {data.scope.real_crops.toLocaleString()} crops</dd></div>
        <div><dt>Shard</dt><dd>{data.scope.source_shards.toLocaleString()} 个分层抽样 batch</dd></div>
        <div><dt>Crop</dt><dd>{data.scope.crop_size.join(' × ')} px，每张源图 1 个；三种位置均衡轮换</dd></div>
        <div><dt>浏览范围</dt><dd>全部 {data.scope.real_crops.toLocaleString()} crops；索引与七方法明细按需加载</dd></div>
        <div><dt>计时</dt><dd>{data.scope.timing_scope}</dd></div>
        <div><dt>硬件</dt><dd>{data.hardware.cpu} · {data.hardware.logical_cpus} logical CPUs</dd></div>
        <div><dt>原版审计</dt><dd>{original ? '1 crop，冷加载 + 首张 + 热重复' : '未运行'}</dd></div>
      </dl>
    </div>
  );
}

function Header({ active, onNavigate }) {
  const [scopeOpen, setScopeOpen] = useState(false);
  return (
    <header className="app-header">
      <div className="brand-lockup">
        <span className="brand-mark">HX</span>
        <span className="brand-rule" />
        <span className="brand-title">HEX / 100B 可行性实验</span>
      </div>
      <nav className="main-nav" aria-label="主要页面">
        {NAV_ITEMS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={active === key ? 'active' : ''}
            aria-current={active === key ? 'page' : undefined}
            onClick={() => onNavigate(key)}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="scope-control-wrap">
        <button type="button" className="scope-control" onClick={() => setScopeOpen((open) => !open)} aria-expanded={scopeOpen}>
          <svg viewBox="0 0 18 18" aria-hidden="true" className="icon database-icon">
            <ellipse cx="9" cy="4" rx="5.5" ry="2.2" />
            <path d="M3.5 4v5c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V4M3.5 9v5c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V9" />
          </svg>
          数据口径
          <ChevronIcon />
        </button>
        {scopeOpen && <ScopePanel onClose={() => setScopeOpen(false)} />}
      </div>
    </header>
  );
}

function MetricStrip() {
  const metrics = [
    [data.scope.real_crops.toLocaleString(), '独立 S3 crops', `${data.scope.source_images.toLocaleString()} 张不同源图 · ${data.scope.source_shards.toLocaleString()} shards`],
    [fmtCompact(data.scale.target_crops_s, 2), '最低持续吞吐', `${data.scale.days} 天主目标；3 / 7 / 15 天场景见规模化`],
    [`${data.scope.methods} + 1`, '测试方法', `${data.scope.methods} 个统一基准 + ColorPipette 原版审计`],
  ];
  return (
    <div className="metric-strip">
      {metrics.map(([value, label, note]) => (
        <div className="metric-item" key={label}>
          <span className="metric-label">{label}</span>
          <strong>{value}</strong>
          <span className="metric-note">{note}</span>
        </div>
      ))}
    </div>
  );
}

function MethodComparison({ selected, onSelect }) {
  return (
    <div className="table-shell method-table-shell">
      <table className="method-table">
        <thead>
          <tr>
            <th>方法 <SortIcon /></th>
            <th>P50 <SortIcon /></th>
            <th>P95 <SortIcon /></th>
            <th>crops/s <SortIcon /></th>
            <th>100B 判断 <SortIcon /></th>
          </tr>
        </thead>
        <tbody>
          {data.summaries.map((summary) => (
            <tr
              key={summary.method}
              className={selected === summary.method ? 'selected' : ''}
              onClick={() => onSelect(summary.method)}
            >
              <td>
                <button type="button" className="row-select" aria-label={`选择 ${summary.label}`}>
                  <span className="radio-dot" />
                  {summary.label}
                </button>
              </td>
              <td className="mono">{fmt(summary.latency_ms.p50, 2)} ms</td>
              <td className="mono">{fmt(summary.latency_ms.p95, 2)} ms</td>
              <td className="mono">{fmt(summary.throughput.crops_s_single_process_e2e, 1)}</td>
              <td><span className={`verdict verdict-${summary.method}`}>{summary.verdict}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RouteDistribution() {
  const [showPercent, setShowPercent] = useState(true);
  const adaptive = data.summaries.find((item) => item.method === 'adaptive_v1');
  const routes = ['flat', 'mild', 'texture', 'gradient'];
  const total = Object.values(adaptive.route_counts).reduce((sum, value) => sum + value, 0);
  const palette = data.records
    .filter((record) => record.method === 'adaptive_v1')
    .filter((_, index) => index % 5 === 0)
    .slice(0, 10)
    .map((record) => record.hex);
  return (
    <div className="route-section">
      <div className="route-main">
        <div className="subsection-head">
          <h2>路线分布 <span>（按 crops）</span></h2>
          <button type="button" className={`switch ${showPercent ? 'on' : ''}`} onClick={() => setShowPercent((value) => !value)} aria-pressed={showPercent}>
            <span />
            显示百分比
          </button>
        </div>
        <div className="route-bar" aria-label="Adaptive 路由分布">
          {routes.map((route) => {
            const count = adaptive.route_counts[route] || 0;
            return <span key={route} style={{ width: `${(count / total) * 100}%`, background: ROUTE_COLORS[route] }} />;
          })}
        </div>
        <div className="route-legend">
          {routes.map((route) => {
            const count = adaptive.route_counts[route] || 0;
            return (
              <div key={route}>
                <span className="legend-key" style={{ background: ROUTE_COLORS[route] }} />
                <b>{route}</b>
                <span className="mono">{showPercent ? `${fmt((count / total) * 100, 1)}%` : `${count}`}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="route-explain">
        <h2>路线说明</h2>
        <dl>
          <div><dt><i style={{ background: ROUTE_COLORS.flat }} />flat</dt><dd>低方差 / 近似纯色区域</dd></div>
          <div><dt><i style={{ background: ROUTE_COLORS.mild }} />mild</dt><dd>轻度纹理或低频变化</dd></div>
          <div><dt><i style={{ background: ROUTE_COLORS.texture }} />texture</dt><dd>高频纹理或复杂结构</dd></div>
          <div><dt><i style={{ background: ROUTE_COLORS.gradient }} />gradient</dt><dd>平滑渐变，输出 observed stops</dd></div>
        </dl>
      </div>
      <div className="palette-ruler">
        <h2>样例主色 <span>（Adaptive · 非分布）</span></h2>
        <div className="ruler-swatches">
          {palette.map((color, index) => <span key={`${color}-${index}`} style={{ background: color }} title={color} />)}
        </div>
      </div>
    </div>
  );
}

function Overview({ onNavigate }) {
  const [selected, setSelected] = useState('adaptive_v1');
  return (
    <main className="page page-overview">
      <section className="decision-hero">
        <h1>Adaptive Hex v1 仍是生产语义方向</h1>
        <p>Octree 原生实现吞吐最高；Color Thief q10 次之；Pixelero / libimagequant 覆盖更好。速度、observed 语义与许可需要分开判断。</p>
      </section>
      <MetricStrip />
      <MethodComparison selected={selected} onSelect={setSelected} />
      <RouteDistribution />
      <div className="overview-action-rail">
        <button type="button" className="primary-action" onClick={() => onNavigate('samples')}>
          <svg viewBox="0 0 18 18" aria-hidden="true" className="icon"><path d="M2.5 3.5h13v11h-13zM2.5 7h13M7 3.5v11" /></svg>
          查看逐图结果
        </button>
        <p>逐图并排查看主 HEX、top-3 palette、路由与耗时。</p>
        <div className="build-stamp">
          <span>数据更新：{new Date(data.generated_at).toLocaleString('zh-CN', { timeZone: 'Australia/Sydney' })}</span>
          <span>schema：v{data.schema_version}</span>
        </div>
      </div>
      <EvidenceNote />
    </main>
  );
}

function EvidenceNote() {
  return (
    <aside className="evidence-note">
      <strong>结论边界</strong>
      <p>当前是单机 CPU gold/reference 实测，不是 100B 生产吞吐证明。最终 Go/No-Go 仍需真实顺序 shard、批量解码、GPU segmented aggregation 与 1B-region pilot。</p>
    </aside>
  );
}

const METHOD_DETAILS = [
  {
    key: 'adaptive_v1',
    index: '01',
    title: 'Adaptive Hex v1',
    decision: '生产方向',
    copy: '一次 q12 统计 + 64 像素确定性采样，按复杂度选择 exact heavy hitter、sample medoid、top-3 LUT palette 或 gradient stops。',
    semantics: '主色和 stops 均选自真实 sRGB 像素；同时输出 method、confidence 与诊断字段。',
  },
  {
    key: 'tencent_hsv',
    index: '02',
    title: '腾讯 HSV 直方图',
    decision: '基线',
    copy: '文章只给出色相量化、直方图排序/平滑和 RGB 回映射的五步草图。本实现固定 60H × 4S × 4V，并为低饱和像素增加 4 个灰度桶。',
    semantics: '输出 winning HSV bucket 中真实出现过的像素。Name That Color 的 1500+ 命名锚点未混入统计；它更适合作为 HEX 之后的可选命名层。',
  },
  {
    key: 'pixelero_rgb_hist',
    index: '03',
    title: 'Pixelero RGB 直方图',
    decision: '覆盖基线',
    copy: '先分别对 R/G/B 通道做 8 组加权直方图聚类，再形成最多 8³ 个 3D bins，最后把非空 bins 聚成 top-3。',
    semantics: '严格按文章结构补全初始化与收敛规则；输出是 RGB centroid，不保证源图中真实出现。',
  },
  {
    key: 'octree',
    index: '04',
    title: 'Octree · native',
    decision: '原生基线',
    copy: '沿 Gervautz–Purgathofer 的八叉树归并思路，用 Pillow FASTOCTREE 原生实现压到 top-3，并关闭 dithering。',
    semantics: '这是算法族的工程适配，不是论文 Pascal 伪代码逐行复刻；叶节点均值不是 observed pixel。',
  },
  {
    key: 'color_thief_v3',
    index: '05',
    title: 'Color Thief v3 · OKLCH',
    decision: '速度参考',
    copy: `官方 ${data?.external_audit?.color_thief?.exact_audit?.version ?? 'v3'} 包原版实跑：默认每 10 像素采样、忽略近白，在缩放后的 OKLCH 坐标上运行 MMCQ。`,
    semantics: 'MIT；官方 Sharp 解码器。默认 q10 的速度不能与全像素扫描脱离采样口径单独解读。',
  },
  {
    key: 'pngquant_liq',
    index: '06',
    title: 'pngquant · libimagequant',
    decision: '质量参考',
    copy: '通过 imagequant-python 1.1.5 调用 libimagequant 2.15.1，在已解码 RGBA 上生成并 remap top-3；不计 PNG 编码和 dithering。',
    semantics: 'palette centroid 非 observed；闭源生产使用需评估 GPL v3+ 或商业许可。',
  },
  {
    key: 'colorpipette_inspired',
    index: '07',
    title: 'ColorPipette-inspired',
    decision: '研究参考',
    copy: '用 SLIC + 确定性色彩对比显著性复刻“分割 → 显著候选 → L/C 和谐化”的研究结构，便于在统一数据上观察 palette 语义。',
    semantics: '和谐化颜色不保证在源图出现；这不是原版 BASNet/SpixelNet 的 bit-identical 复现。',
  },
];

const IMPLEMENTATION_AUDIT = [
  {
    key: 'adaptive_v1',
    index: '01',
    title: 'Adaptive Hex v1',
    provenance: '本项目自研',
    tone: 'custom',
    direct: '未调用外部主色仓库；依据规模化方案实现 CPU gold/reference。',
    pipeline: 'sRGB / alpha → RGB q12（4096 bins）→ 固定抽样 64 像素 → flat / mild / texture / gradient 硬阈值路由。',
    output: '第一色随 route 分别表示 heavy hitter、sample medoid、q12 top-bin 或 gradient 中间 stop；权重也随 route 改变，并非统一面积占比。',
    failure: '64 点抽样和未校准阈值会误路由；同一列“第一色”内部也不是单一语义。',
  },
  {
    key: 'tencent_hsv',
    index: '02',
    title: '腾讯 HSV 直方图',
    provenance: '文章补全',
    tone: 'interpreted',
    direct: '未使用官方代码。文章来源只有五步草图，本项目补全所有 bins、阈值、平滑和 RGB 回选规则。',
    pipeline: '60H × 4S × 4V；S < 0.08 进入 4 个灰度桶；hue 使用 1-2-3-2-1 环形平滑，再从 winning bucket 回选 observed pixel。',
    output: '第一色是 winning HSV / 灰度 bucket 的真实像素代表；权重近似 bucket 像素质量。Name That Color 未参与计算。',
    failure: '粗分桶、低饱和阈值和面积投票容易被背景主导，也无法理解文字、logo 或前景。',
    sourceUrl: data?.external_audit?.tencent?.url,
    sourceLabel: '文章原文',
  },
  {
    key: 'pixelero_rgb_hist',
    index: '03',
    title: 'Pixelero RGB 直方图',
    provenance: '文章补全',
    tone: 'interpreted',
    direct: '未复制仓库实现；按照博客结构重新实现，并自行固定初始化、32 次迭代和收敛规则。',
    pipeline: 'R / G / B 各做 8 类一维加权聚类 → 组合为最多 8³ 个 bins → 对非空 bins 做 weighted RGB k-means → top-3。',
    output: '第一色是最终人口最大的 RGB cluster centroid；权重是归一化 cluster population；颜色不保证在原图出现。',
    failure: '独立切分三个通道会破坏颜色相关性；RGB 欧氏距离非感知均匀，centroid 可能产生人工未认可的中间色。',
    sourceUrl: data?.external_audit?.pixelero?.url,
    sourceLabel: '博客原文',
  },
  {
    key: 'octree',
    index: '04',
    title: 'Octree · native',
    provenance: '现有原生库',
    tone: 'direct',
    direct: '直接调用 Pillow Image.quantize(..., FASTOCTREE, dither=NONE)；不是论文 Pascal 代码的逐行复刻。',
    pipeline: 'RGBA → Pillow FASTOCTREE 压缩到 3 色 → 按量化后 palette population 排序。',
    output: '第一色是量化后人口最大的 octree palette entry；权重是 remapped population；叶均值不是 observed pixel。',
    failure: '目标是快速降低量化误差，不是语义主色；RGB 树边界、背景面积和接近并列的颜色都会改变排序。',
    sourceUrl: data?.external_audit?.octree?.url,
    sourceLabel: '算法论文',
  },
  {
    key: 'color_thief_v3',
    index: '05',
    title: 'Color Thief v3 · OKLCH',
    provenance: '官方 npm 包',
    tone: 'direct',
    direct: `直接运行官方 colorthief ${data?.external_audit?.color_thief?.exact_audit?.version ?? '3.4.0'} 的 internals：NodePixelLoader、MmcqQuantizer 与 extractPalette。`,
    pipeline: '官方 Sharp loader → 每 10 像素取 1 个 → 默认忽略近白 → OKLCH 坐标中的 MMCQ → top-3。',
    output: '第一色是官方返回 palette 的第一项；权重是 sampled palette proportion，不等于全像素面积；centroid 非 observed。',
    failure: 'q10 会漏掉细小结构，ignoreWhite 会改变白底图语义；MMCQ 仍优化颜色分箱而非人工“重要性”。',
    sourceUrl: data?.external_audit?.color_thief?.url,
    sourceLabel: '官方项目',
  },
  {
    key: 'pngquant_liq',
    index: '06',
    title: 'libimagequant 2.15.1',
    provenance: '现有原生引擎',
    tone: 'direct',
    direct: '通过 imagequant-python 1.1.5 调用真实 libimagequant 2.15.1；没有运行 pngquant CLI，也不包含 PNG 编码。',
    pipeline: '已解码 RGBA → libimagequant max_colors=3、quality=0..100、dithering=0 → remap → 按可见像素质量排序。',
    output: '第一色是 remap 后质量最大的 palette centroid；权重是 remapped visible mass；颜色不保证 observed。',
    failure: '它为压缩重建优化，不为主色语义优化；小而显著的前景会输给大背景，接近并列时顺序容易翻转。',
    sourceUrl: data?.external_audit?.pngquant?.url,
    sourceLabel: 'pngquant / libimagequant',
  },
  {
    key: 'colorpipette_inspired',
    index: '07',
    title: 'ColorPipette-inspired',
    provenance: '本项目 proxy',
    tone: 'proxy',
    direct: '未运行原版 BASNet / SpixelNet；只直接使用 scikit-image SLIC，其余显著性、候选选择和 harmony 均为本项目近似。',
    pipeline: '缩放 → SLIC → Oklab 对比度 + center prior → 显著 segment 选择 → 自定义 L/C harmony → top-3。',
    output: '第一色是得分最高的显著 segment 经 harmony 后的颜色；权重是 saliency × √area，不是面积；非 observed。',
    failure: 'center prior 和对比度会偏爱高反差小物体；背景 palette 虽被计算但只写入 diagnostics，没有进入最终 palette。',
    sourceUrl: data?.external_audit?.colorpipette?.url,
    sourceLabel: '原版仓库',
  },
];

function MethodsView() {
  const original = data.external_audit.colorpipette.original_cpu_audit;
  return (
    <main className="page page-methods">
      <section className="page-heading">
        <h1>七种统一基准，一种原版审计</h1>
        <p>先锁定输出语义，再讨论速度；“一个 HEX”对设计 fill 和自然纹理不是同一个问题。</p>
      </section>
      <div className="method-detail-list">
        {METHOD_DETAILS.map((method) => (
          <article className="method-detail-row" key={method.key}>
            <span className="method-index mono">{method.index}</span>
            <div>
              <h2>{method.title}</h2>
              <p>{method.copy}</p>
            </div>
            <div className="method-semantics">
              <span>{method.decision}</span>
              <p>{method.semantics}</p>
            </div>
          </article>
        ))}
      </div>

      <section className="schema-section">
        <div>
          <h2>统一输出</h2>
          <p>生产侧保存紧凑整数和 enum；HEX 只在 UI / CSV 导出时格式化。</p>
        </div>
        <pre>{`rgb24: uint32
method: uint8
confidence: float16
palette_rgb24: fixed_list<uint32>[3]
palette_weight: fixed_list<uint16>[3]
diagnostics: optional struct`}</pre>
      </section>

      <section className="source-audit-section">
        <div className="section-title-row">
          <h2>外部方法审计</h2>
          <span>源码事实与本次实现严格分开</span>
        </div>
        <div className="audit-grid">
          <article>
            <h3>色相量化草图</h3>
            <p>没有代码，也没有固定 hue bins、平滑核、低饱和处理和 RGB 恢复规则；本项目给出 60H × 4S × 4V 补全，并把 Name That Color 留在输出后的命名层。</p>
            <a href={data.external_audit.tencent.url} target="_blank" rel="noreferrer">打开原文 ↗</a>
          </article>
          <article>
            <h3>Pixelero</h3>
            <p>文章明确了单通道直方图、最多 512 个 RGB bins 与最终聚类，但初始化和停止条件需要补全。</p>
            <a href={data.external_audit.pixelero.url} target="_blank" rel="noreferrer">打开文章 ↗</a>
          </article>
          <article>
            <h3>Octree 论文</h3>
            <p>原始方法把颜色插入八叉树，并从最深叶向根归并；本次用原生 fast-octree 比较这一算法族。</p>
            <a href={data.external_audit.octree.url} target="_blank" rel="noreferrer">打开 TU Wien 论文 ↗</a>
          </article>
          <article>
            <h3>Color Thief v3</h3>
            <p>当前版默认 OKLCH + MMCQ、quality=10；本次运行官方包，不再用旧版 MMCQ 印象代替当前实现。</p>
            <a href={data.external_audit.color_thief.url} target="_blank" rel="noreferrer">打开项目 ↗</a>
          </article>
          <article>
            <h3>pngquant / libimagequant</h3>
            <p>CLI 负责 PNG I/O，核心库负责 RGBA palette 与 remap；本次为 libimagequant 2.15.1 核心适配，不冒充当前 v3 CLI 端到端实测。</p>
            <a href={data.external_audit.pngquant.url} target="_blank" rel="noreferrer">打开仓库 ↗</a>
          </article>
          <article>
            <h3>ColorPipette 源码</h3>
            <p>目标是可视化用和谐 palette；依赖 BASNet、SpixelNet、Lab/LCh 和 Python 像素循环，不是 crop HEX 标注器。</p>
            <a href={data.external_audit.colorpipette.url} target="_blank" rel="noreferrer">打开仓库 ↗</a>
          </article>
        </div>
        {original && (
          <div className="original-audit-rail">
            <div><span>模型权重</span><strong>{fmt((original.weights.basnet_bytes + original.weights.spixelnet_bytes) / 1e6, 1)} MB</strong></div>
            <div><span>冷加载</span><strong>{fmt(original.model_load_seconds, 2)} s</strong></div>
            <div><span>首张</span><strong>{fmt(original.first_inference_seconds, 2)} s</strong></div>
            <div><span>热重复</span><strong>{fmt(original.inference_seconds, 2)} s</strong></div>
            <div><span>峰值 RSS</span><strong>{fmt(original.peak_rss_mb, 0)} MB</strong></div>
            <div><span>结果一致</span><strong>{original.repeat_identical ? '是' : '否'}</strong></div>
          </div>
        )}
        <div className="warning-line">
          <strong>原版运行警告</strong>
          <span>上游 NumPy uint8 在 saliency 平均与 Lab→LCh 中发生溢出/下溢；仓库根目录没有 license 文件。</span>
        </div>
      </section>
    </main>
  );
}

function ImplementationView() {
  const representativeColors = data.records
    .filter((record) => record.method === 'adaptive_v1')
    .filter((_, index) => index % 5 === 0)
    .slice(0, 10)
    .map((record) => record.hex);
  const original = data.external_audit.colorpipette.original_cpu_audit;

  return (
    <main className="page page-implementation">
      <section className="page-heading">
        <h1>实现、来源与结果边界</h1>
        <p>七列统一成 top-3 只是为了并排查看，并不表示它们在回答同一个“主色”问题；直接库、文章补全和研究 proxy 必须分开理解。</p>
      </section>

      <section className="implementation-summary">
        <div>
          <span className="audit-kicker">PROVENANCE FIRST</span>
          <h2>当前不是七个成熟上游算法的公平质量排名</h2>
          <p>其中三项调用真实现有引擎，三项是本项目对设计或文章的具体补全，一项是研究结构 proxy。当前 10k 数据没有人工颜色真值，因此只能比较运行形态与输出差异。</p>
        </div>
        <dl>
          <div><dt>3</dt><dd>现有 / 官方引擎<br /><span>Color Thief、Pillow Octree、libimagequant</span></dd></div>
          <div><dt>3</dt><dd>本项目实现<br /><span>Adaptive、腾讯 HSV、Pixelero</span></dd></div>
          <div><dt>1</dt><dd>研究 proxy<br /><span>ColorPipette-inspired</span></dd></div>
        </dl>
      </section>

      <section className="representative-disclosure">
        <div className="representative-visual">
          <span className="audit-kicker">首页组件</span>
          <h2>“样例主色”不是颜色分布</h2>
          <div className="disclosure-palette" aria-label={`十个 Adaptive 样例主色：${representativeColors.join(', ')}`}>
            {representativeColors.map((color, index) => (
              <div key={`${color}-${index}`}>
                <span style={{ background: color }} />
                <code>{color}</code>
              </div>
            ))}
          </div>
        </div>
        <div className="representative-explanation">
          <code>60 个内嵌 Adaptive 样例 → 每隔 5 条取 1 条 → 前 10 个主 HEX</code>
          <p>它不是 10k 最常见颜色、hue histogram、百分位或七方法综合结果；只是一组来自真实输出的视觉样例。首页已明确标为“非分布”。</p>
          <strong>不能据此判断数据集颜色构成或方法质量。</strong>
        </div>
      </section>

      <section className="semantic-boundary">
        <div>
          <span className="audit-kicker">统一容器 ≠ 统一语义</span>
          <h2>第一色和色条长度在不同方法中含义不同</h2>
        </div>
        <p>有的是像素面积，有的是采样后 cluster proportion，有的是 saliency score，还有的是 route-specific confidence。页面中的色条只能在同一方法、同一张图内部辅助阅读，不能跨方法当作统一概率比较。</p>
      </section>

      <section className="implementation-methods">
        <div className="section-title-row">
          <h2>七种方法逐项审计</h2>
          <span>实际流程 · 上游复用 · 输出语义 · 常见失败</span>
        </div>
        <div className="implementation-card-list">
          {IMPLEMENTATION_AUDIT.map((method) => (
            <article className="implementation-card" key={method.key}>
              <div className="implementation-card-title">
                <span className="method-index mono">{method.index}</span>
                <h3>{method.title}</h3>
                <span className={`provenance-badge provenance-${method.tone}`}>{method.provenance}</span>
                <p>{method.direct}</p>
                {method.sourceUrl && <a href={method.sourceUrl} target="_blank" rel="noreferrer">{method.sourceLabel} ↗</a>}
              </div>
              <div className="implementation-card-column">
                <span>实际流程</span>
                <p>{method.pipeline}</p>
              </div>
              <div className="implementation-card-column">
                <span>第一色 / 权重语义</span>
                <p>{method.output}</p>
              </div>
              <div className="implementation-card-column limitation-column">
                <span>人工检查常见问题</span>
                <p>{method.failure}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      {original && (
        <section className="original-source-disclosure">
          <div>
            <span className="audit-kicker">单独审计，不在 10k 七方法表中</span>
            <h2>ColorPipette 原版确实运行过，但只有一个 crop</h2>
            <p>这里克隆了上游仓库，加载 BASNet 与 SpixelNet 权重，并直接调用原 Flask endpoint；只做了现代运行环境兼容补丁。它不能为 10k 的 ColorPipette-inspired proxy 背书。</p>
          </div>
          <dl>
            <div><dt>范围</dt><dd>1 crop</dd></div>
            <div><dt>模型权重</dt><dd>{fmt((original.weights.basnet_bytes + original.weights.spixelnet_bytes) / 1e6, 1)} MB</dd></div>
            <div><dt>首张 CPU</dt><dd>{fmt(original.first_inference_seconds, 2)} s</dd></div>
            <div><dt>峰值 RSS</dt><dd>{fmt(original.peak_rss_mb, 0)} MB</dd></div>
          </dl>
        </section>
      )}

      <section className="manual-review-section">
        <div className="section-title-row">
          <h2>为什么人工看会觉得每种方法都有问题</h2>
          <span>当前缺少统一的颜色标注目标</span>
        </div>
        <div className="manual-review-grid">
          <div><span>01</span><h3>目标未定义</h3><p>面积主导色、前景显著色、设计 fill、observed palette 与重建 palette 是不同任务。</p></div>
          <div><span>02</span><h3>第一色不可横比</h3><p>七种方法分别返回 heavy hitter、bucket representative、centroid、量化 palette 或显著 segment。</p></div>
          <div><span>03</span><h3>没有人工真值</h3><p>10k 运行只能稳定估计速度和输出分布，不能证明颜色语义正确。</p></div>
          <div><span>04</span><h3>统一 top-3 不等于公平</h3><p>相同输出数量掩盖了采样、白色过滤、空间信息、色彩空间和权重含义的差异。</p></div>
        </div>
      </section>

      <aside className="implementation-next-step">
        <strong>下一步质量评测</strong>
        <p>先定义人工目标，再建立分层人工集；标注颜色角色与可接受集合，用 permutation-invariant palette matching、角色准确率和失败类型评估，而不是只比较第一 HEX。</p>
      </aside>
    </main>
  );
}

function PaletteBars({ colors, weights = [] }) {
  const visibleColors = colors.slice(0, 3);
  const visibleWeights = visibleColors.map((_, index) => {
    const weight = Number(weights[index]);
    return Number.isFinite(weight) && weight > 0 ? weight : 0;
  });
  const maximumWeight = Math.max(...visibleWeights, 0);
  return (
    <div className="palette-bars" aria-label={`palette ${visibleColors.join(', ')}`}>
      {visibleColors.map((color, index) => {
        const relativeWidth = maximumWeight > 0 ? (visibleWeights[index] / maximumWeight) * 100 : 100;
        return (
          <div className="palette-bar-row" key={`${color}-${index}`}>
            <span className="palette-bar-track" aria-hidden="true">
              <span
                className="palette-bar-fill"
                style={{ background: color, width: `${Math.max(relativeWidth, 8)}%` }}
              />
            </span>
            <strong className="mono palette-bar-hex">{color}</strong>
          </div>
        );
      })}
    </div>
  );
}

function MethodCell({ record, method }) {
  return (
    <div className="sample-method-cell">
      <span className="mobile-method-label">{METHOD_LABELS[method]}</span>
      <PaletteBars colors={record.palette_hex} weights={record.weights} />
      <dl className="sample-cell-stats">
        <div><dt>route</dt><dd>{ROUTE_LABELS[record.route] || record.route}</dd></div>
        <div><dt>ms</dt><dd className="mono">{fmt(record.elapsed_ms, 2)}</dd></div>
      </dl>
    </div>
  );
}

function SamplesView() {
  const [sampleIndex, setSampleIndex] = useState([]);
  const [visibleGroups, setVisibleGroups] = useState([]);
  const [indexLoading, setIndexLoading] = useState(true);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [search, setSearch] = useState('');
  const [batch, setBatch] = useState('all');
  const [route, setRoute] = useState('all');
  const [sort, setSort] = useState('difference');
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(3);
  const [pageDraft, setPageDraft] = useState('1');
  const [selectedId, setSelectedId] = useState();

  useEffect(() => {
    let active = true;
    loadSampleIndex()
      .then((payload) => {
        if (!active) return;
        if (payload.count !== data.scope.real_crops || payload.records.length !== payload.count) {
          throw new Error(`sample index count ${payload.records.length} does not match ${data.scope.real_crops}`);
        }
        setSampleIndex(payload.records);
        setIndexLoading(false);
      })
      .catch((error) => {
        if (!active) return;
        setLoadError(String(error));
        setIndexLoading(false);
      });
    return () => { active = false; };
  }, []);

  const batches = useMemo(
    () => [...new Set(sampleIndex.map((item) => item.batch))].sort((left, right) => Number(left) - Number(right)),
    [sampleIndex],
  );

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    let groups = sampleIndex.filter((group) => batch === 'all' || group.batch === batch);
    groups = groups.filter((group) => route === 'all' || group.route === route);
    if (query) {
      groups = groups.filter((group) => (
        `${group.crop_id} ${group.source_file} ${group.batch}`.toLowerCase().includes(query)
      ));
    }
    groups = [...groups];
    if (sort === 'difference') groups.sort((left, right) => right.difference - left.difference);
    if (sort === 'latency') groups.sort((left, right) => right.latency - left.latency);
    if (sort === 'source') groups.sort((a, b) => a.crop_id.localeCompare(b.crop_id));
    return groups;
  }, [sampleIndex, search, batch, route, sort]);

  useEffect(() => setPage(0), [search, batch, route, sort, pageSize]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visibleIndex = filtered.slice(page * pageSize, page * pageSize + pageSize);
  const visibleKey = visibleIndex.map((item) => item.record_path).join('|');

  useEffect(() => {
    if (page >= pageCount) setPage(pageCount - 1);
  }, [page, pageCount]);

  useEffect(() => setPageDraft(String(page + 1)), [page]);

  useEffect(() => {
    let active = true;
    if (!visibleKey) {
      setVisibleGroups([]);
      return () => { active = false; };
    }
    setVisibleGroups([]);
    setLoadError('');
    setRecordsLoading(true);
    Promise.all(visibleIndex.map((item) => loadSampleRecord(item.record_path)))
      .then((groups) => {
        if (!active) return;
        setVisibleGroups(groups);
        setRecordsLoading(false);
      })
      .catch((error) => {
        if (!active) return;
        setLoadError(String(error));
        setRecordsLoading(false);
      });
    return () => { active = false; };
  }, [visibleKey]);

  const selected = visibleGroups.find((group) => group.crop_id === selectedId) || visibleGroups[0];
  const commitPage = () => {
    const requested = Number.parseInt(pageDraft, 10);
    if (Number.isFinite(requested)) setPage(Math.max(0, Math.min(pageCount - 1, requested - 1)));
    else setPageDraft(String(page + 1));
  };

  return (
    <main className="page page-samples">
      <section className="page-heading compact-heading">
        <h1>逐图结果</h1>
        <p>全部 {data.scope.real_crops.toLocaleString()} 个 crops 均可搜索、筛选、排序、翻页和逐条查看；当前页图片与七方法明细按需加载。</p>
      </section>
      <div className="sample-filters">
        <label className="sample-search">
          <span className="sr-only">搜索样本</span>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索 crop ID / source / batch" />
        </label>
        <label>
          <span className="sr-only">来源</span>
          <select value={batch} onChange={(event) => setBatch(event.target.value)}>
            <option value="all">全部来源</option>
            {batches.map((item) => <option key={item} value={item}>S3 · batch {item}</option>)}
          </select>
          <ChevronIcon />
        </label>
        <label>
          <span className="sr-only">Adaptive 路由</span>
          <select value={route} onChange={(event) => setRoute(event.target.value)}>
            <option value="all">全部路由</option>
            {['flat', 'mild', 'texture', 'gradient'].map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <ChevronIcon />
        </label>
        <label className="selected-filter">
          <span className="sr-only">排序</span>
          <select value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="difference">按差异排序</option>
            <option value="latency">按耗时排序</option>
            <option value="source">按来源排序</option>
          </select>
          <ChevronIcon />
        </label>
        <label className="page-size-filter">
          <span className="sr-only">每页数量</span>
          <select value={pageSize} onChange={(event) => setPageSize(Number(event.target.value))}>
            {[3, 10, 25].map((size) => <option key={size} value={size}>每页 {size} 条</option>)}
          </select>
          <ChevronIcon />
        </label>
        <span className="filter-count mono">{filtered.length.toLocaleString()} / {data.scope.real_crops.toLocaleString()} 可查看</span>
      </div>

      <div className="sample-matrix-shell">
        <div className="sample-matrix-header sample-matrix-grid" style={{ '--method-count': METHOD_ORDER.length }}>
          <div>crop（来源 · 批次）</div>
          {METHOD_ORDER.map((method) => <div key={method}>{METHOD_LABELS[method]} <SortIcon /></div>)}
        </div>
        {(indexLoading || recordsLoading) && <div className="empty-row sample-loading">正在加载当前页…</div>}
        {loadError && <div className="empty-row sample-error">加载失败：{loadError}</div>}
        {!indexLoading && !recordsLoading && !loadError && visibleGroups.map((group) => (
          <button
            type="button"
            className={`sample-row sample-matrix-grid ${selected?.crop_id === group.crop_id ? 'selected' : ''}`}
            style={{ '--method-count': METHOD_ORDER.length }}
            key={group.crop_id}
            onClick={() => setSelectedId(group.crop_id)}
          >
            <div className="crop-cell">
              <span className="radio-dot" />
              <div>
                <strong>S3 · batch {group.batch}</strong>
                <span>{group.variant.replaceAll('_', ' ')}</span>
                <img src={`/${group.asset_path}`} alt={`S3 crop ${group.crop_id}`} loading="lazy" decoding="async" />
              </div>
            </div>
            {METHOD_ORDER.map((method) => <MethodCell key={method} method={method} record={group.methods[method]} />)}
          </button>
        ))}
        {!indexLoading && !recordsLoading && !loadError && visibleGroups.length === 0 && <div className="empty-row">没有符合筛选条件的 crop。</div>}
        {selected && <SampleDetail group={selected} />}
      </div>

      <div className="pagination">
        <button type="button" onClick={() => setPage(0)} disabled={page === 0}>首页</button>
        <button type="button" onClick={() => setPage((value) => Math.max(0, value - 1))} disabled={page === 0}>上一页</button>
        <form className="page-jump" onSubmit={(event) => { event.preventDefault(); commitPage(); }}>
          <span>第</span>
          <input aria-label="跳转页码" inputMode="numeric" value={pageDraft} onChange={(event) => setPageDraft(event.target.value)} onBlur={commitPage} />
          <span className="mono">/ {pageCount.toLocaleString()} 页</span>
        </form>
        <button type="button" className="next" onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))} disabled={page >= pageCount - 1}>下一页</button>
        <button type="button" onClick={() => setPage(pageCount - 1)} disabled={page >= pageCount - 1}>末页</button>
      </div>
    </main>
  );
}

function SampleDetail({ group }) {
  const adaptive = group.methods.adaptive_v1;
  return (
    <div className="sample-detail-rail">
      <div><h3>输入信息</h3><dl><div><dt>尺寸</dt><dd>{group.width} × {group.height}</dd></div><div><dt>像素</dt><dd>{adaptive.pixels.toLocaleString()}</dd></div></dl></div>
      <div><h3>方法输出</h3><dl><div><dt>主色</dt><dd className="mono">{adaptive.hex}</dd></div><div><dt>route</dt><dd>{adaptive.route}</dd></div></dl></div>
      <div><h3>Coverage 代理</h3><dl><div><dt>palette coverage</dt><dd>{fmt(adaptive.palette_coverage_delta_e_ok, 2)} ΔE</dd></div></dl></div>
    </div>
  );
}

function EfficiencyView() {
  const maximumP95 = Math.max(...data.summaries.map((summary) => summary.latency_ms.p95));
  const original = data.external_audit.colorpipette.original_cpu_audit;
  return (
    <main className="page page-efficiency">
      <section className="page-heading">
        <h1>效率与输出特征</h1>
        <p>同一台 Devbox、{data.scope.real_crops.toLocaleString()} 个独立 JPEG crops；均计 decode + extraction。Color Thief 按官方 Sharp loader，其余方法用 Pillow。</p>
      </section>

      <section className="latency-chart-section">
        <div className="section-title-row"><h2>P50 / P95 端到端延迟</h2><span>越短越好 · ms/crop</span></div>
        <div className="latency-bars">
          {data.summaries.map((summary) => (
            <div className="latency-row" key={summary.method}>
              <strong>{summary.label}</strong>
              <div className="latency-track">
                <span className="latency-p95" style={{ width: `${(summary.latency_ms.p95 / maximumP95) * 100}%` }} />
                <span className="latency-p50" style={{ width: `${(summary.latency_ms.p50 / maximumP95) * 100}%` }} />
              </div>
              <span className="mono">{fmt(summary.latency_ms.p50, 2)} / {fmt(summary.latency_ms.p95, 2)}</span>
            </div>
          ))}
        </div>
      </section>

      <div className="table-shell efficiency-table-shell">
        <table className="efficiency-table">
          <thead><tr><th>方法</th><th>单进程 crops/s</th><th>method MPix/s</th><th>palette coverage</th><th>observed</th></tr></thead>
          <tbody>
            {data.summaries.map((summary) => (
              <tr key={summary.method}>
                <td><strong>{summary.label}</strong></td>
                <td className="mono">{fmt(summary.throughput.crops_s_single_process_e2e, 2)}</td>
                <td className="mono">{fmt(summary.throughput.mpix_s_method_only, 2)}</td>
                <td className="mono">{fmt(summary.quality_proxy.palette_coverage_mean_delta_e_ok, 2)}</td>
                <td className="mono">{fmt(summary.quality_proxy.observed_output_rate * 100, 0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {original && (
        <section className="original-performance-band">
          <div><h2>ColorPipette 原版 · 单张审计</h2><p>不并入 {data.scope.real_crops.toLocaleString()}-crop 汇总；这是兼容层后的真实 BASNet + SpixelNet CPU 执行。</p></div>
          <dl>
            <div><dt>cold load</dt><dd>{fmt(original.model_load_seconds, 2)} s</dd></div>
            <div><dt>first crop</dt><dd>{fmt(original.first_inference_seconds, 2)} s</dd></div>
            <div><dt>warm repeat</dt><dd>{fmt(original.inference_seconds, 2)} s</dd></div>
            <div><dt>peak RSS</dt><dd>{fmt(original.peak_rss_mb, 0)} MB</dd></div>
          </dl>
        </section>
      )}

      <section className="methodology-rails">
        <div><h3>硬件</h3><p>{data.hardware.cpu}</p><span>{data.hardware.logical_cpus} logical CPUs · GPU 不可见</span></div>
        <div><h3>采样口径</h3><p>Color Thief v3 保留官方默认 quality=10。</p><span>其速度来自每 10 像素采 1 个；其余统一基准读取全部有效像素。</span></div>
        <div><h3>输出语义</h3><p>observed 表示主色来自输入中实际出现的 sRGB 像素。</p><span>centroid / palette mean 则不保证是原图中的真实像素。</span></div>
        <div><h3>Coverage</h3><p>像素到最近 palette 色的平均 Oklab 距离。</p><span>该指标偏好覆盖全图，不等同人工 palette 偏好。</span></div>
      </section>
    </main>
  );
}

function PipelineIcon({ kind }) {
  const paths = {
    storage: <><ellipse cx="10" cy="5" rx="6" ry="2.5" /><path d="M4 5v6c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5V5M4 11v4c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5v-4" /></>,
    decode: <><rect x="3" y="3" width="14" height="14" /><path d="m8 7 5 3-5 3z" /></>,
    stats: <path d="M4 16v-5h3v5M9 16V7h3v9M14 16V3h3v13" />,
    route: <path d="M3 16c0-5 3-6 7-6h6m-4-4 4 4-4 4M7 10V6" />,
    output: <><path d="M5 2h8l4 4v12H5zM13 2v5h4" /><path d="M8 11h6M8 14h6" /></>,
  };
  return <svg viewBox="0 0 20 20" aria-hidden="true" className="pipeline-icon">{paths[kind]}</svg>;
}

function CommandsDialog({ onClose }) {
const commands = `./scripts/download_s3_samples.sh
conda run -n cg python -m unittest discover -s tests -v
conda run -n cg python scripts/extract_hex.py data/samples/s3_10k --method adaptive_v1 --workers 4
conda run -n cg python scripts/audit_colorpipette_original.py
conda run -n cg python scripts/run_benchmark.py
npm run build`;
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(commands);
    setCopied(true);
  };
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="commands-dialog" role="dialog" aria-modal="true" aria-labelledby="commands-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="scope-panel-head"><strong id="commands-title">复现命令</strong><button type="button" onClick={onClose} aria-label="关闭">×</button></div>
        <p>原版审计需要先克隆 ColorPipette 并应用 <code>third_party/colorpipette_compat.patch</code>。</p>
        <pre>{commands}</pre>
        <button type="button" className="secondary-action" onClick={copy}>{copied ? '已复制' : '复制命令'}</button>
      </div>
    </div>
  );
}

function CapacityScenarioSelector({ scenarios, selectedDays, onSelect }) {
  return (
    <div className="scenario-selector" role="group" aria-label="完成时间场景">
      {scenarios.map((scenario) => (
        <button
          type="button"
          key={scenario.days}
          className={selectedDays === scenario.days ? 'active' : ''}
          aria-pressed={selectedDays === scenario.days}
          onClick={() => onSelect(scenario.days)}
        >
          <strong>{scenario.days} 天</strong>
          <span className="mono">{fmtCompact(scenario.target_crops_s, 2)}/s required</span>
          <small className="mono">{fmtCompact(scenario.provisioned_crops_s, 2)}/s · {fmt(scenario.decode_once_model.estimated_fused_workers_1_5x, 0)} fused workers</small>
        </button>
      ))}
    </div>
  );
}

function DecodeOnceCapacity({ scenario }) {
  const model = scenario.decode_once_model;
  return (
    <section className="capacity-model" aria-label="Decode-once production capacity model">
      <div className="capacity-model-head">
        <div><span className="eyebrow">Decode-once model</span><h2>{scenario.days} 天生产侧的可工程化口径</h2></div>
        <p>假设 5B source images、平均 20 crops/image、1 MP/image；所有区域在一次解码后批内聚合。</p>
      </div>
      <div className="capacity-model-grid">
        <div><span>source images/s</span><strong className="mono">{fmtCompact(model.images_s, 2)}</strong><small>5B images / {scenario.days} 天</small></div>
        <div><span>source Gpixel/s</span><strong className="mono">{fmt(model.source_gpix_s, 2)}</strong><small>@ 1 MP / image</small></div>
        <div><span>compressed input</span><strong className="mono">{fmt(model.compressed_input_gb_s, 2)} GB/s</strong><small>@ 0.5 MB / image</small></div>
        <div><span>decoded RGB</span><strong className="mono">{fmt(model.decoded_rgb_gb_s, 2)} GB/s</strong><small>3 bytes / pixel</small></div>
        <div><span>fused workers</span><strong className="mono">{fmt(model.estimated_fused_workers_1_5x, 0)}</strong><small>@ {fmt(model.worker_gpix_s, 1)} Gpix/s sustained · 1.5×</small></div>
      </div>
      <p className="capacity-warning">如果 100B crops 已经物化成独立对象并逐个 GET/decode，这个 20× 解码摊销不成立，I/O 与 worker 需求会显著上升。</p>
    </section>
  );
}

function ScaleView() {
  const [commandsOpen, setCommandsOpen] = useState(false);
  const scenarios = data.scale.scenarios;
  const [selectedDays, setSelectedDays] = useState(data.scale.days);
  const selectedScenario = scenarios.find((scenario) => scenario.days === selectedDays) || scenarios[0];
  const originalProjection = data.external_audit.colorpipette.original_projection;
  const rows = data.summaries.map((summary) => ({
    method: summary.method,
    label: summary.label,
    crops: summary.throughput.crops_s_single_process_e2e,
    processes: selectedScenario.cpu_equivalent_processes_1_5x[summary.method],
    hours: summary.throughput.worker_hours_per_1b,
    risk: summary.method === 'adaptive_v1'
      ? `高：${selectedScenario.days} 天目标必须 fused batch kernel 与真实 shard 验证`
      : summary.risk,
    verdict: summary.verdict,
    isOriginal: false,
  }));
  rows.push({
    method: 'colorpipette_original',
    label: 'ColorPipette 原版',
    crops: originalProjection?.crops_s_single_process_warm_model,
    processes: selectedScenario.cpu_equivalent_processes_1_5x.colorpipette_original,
    hours: originalProjection?.worker_hours_per_1b,
    risk: '极高：双模型、1.7 GB RSS、旧运行栈与像素循环',
    verdict: 'NO：不进入 100B 默认路径',
    isOriginal: true,
  });
  return (
    <main className="page page-scale">
      <section className="page-heading compact-heading">
        <h1>100B 规模判断</h1>
        <p>切换 3 / 7 / 15 天完成窗口；required、1.5× provisioned、decode-once I/O 与 CPU-equivalent 会同步更新。</p>
      </section>
      <CapacityScenarioSelector scenarios={scenarios} selectedDays={selectedDays} onSelect={setSelectedDays} />
      <div className="scale-equation">
        <div><strong>100B crops</strong><span>目标总量</span></div><b>÷</b>
        <div><strong>{selectedScenario.days} 天</strong><span>时间预算</span></div><b>=</b>
        <div><strong>{fmtCompact(selectedScenario.target_crops_s, 2)} crops/s</strong><span>最低持续吞吐</span></div><b>→</b>
        <div><strong>{fmtCompact(selectedScenario.provisioned_crops_s, 2)} crops/s</strong><span>{data.scale.safety_factor}× 建议配置</span></div>
      </div>

      <DecodeOnceCapacity scenario={selectedScenario} />

      <div className="table-shell scale-table-shell">
        <table className="scale-table">
          <thead><tr><th>方法</th><th>实测 crops/s</th><th>{selectedScenario.days} 天等效进程数（1.5×）</th><th>每 1B worker-hours</th><th>扩展风险</th><th>结论</th></tr></thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className={row.method === 'adaptive_v1' ? 'selected' : ''}>
                <td><span className="radio-dot" /><strong>{row.label}</strong>{row.isOriginal && <small>1 crop audit</small>}</td>
                <td className="mono">{fmt(row.crops, row.isOriginal ? 3 : 1)}</td>
                <td className="mono">{fmt(row.processes, 0)}</td>
                <td className="mono">{fmtCompact(row.hours, 1)}</td>
                <td>{row.risk}</td>
                <td className={row.isOriginal ? 'danger-text' : ''}>{row.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="pipeline-section" aria-label="推荐生产流水线">
        <div className="pipeline-main">
          {[
            ['storage', 'sharded S3'],
            ['decode', 'decode once'],
            ['stats', 'batch stats'],
            ['route', 'adaptive route'],
            ['output', 'Parquet RGB24'],
          ].map(([kind, label], index) => (
            <React.Fragment key={label}>
              <div className="pipeline-node"><PipelineIcon kind={kind} /><span>{label}</span></div>
              {index < 4 && <span className="pipeline-arrow" aria-hidden="true">→</span>}
            </React.Fragment>
          ))}
        </div>
        <div className="rejected-branch"><span>×</span> per-crop neural segmentation</div>
      </section>

      <section className="audit-evidence-rail">
        <div><h2>Hue / RGB histogram</h2><p>两种草图均已固定 bins、平滑和聚类规则</p></div>
        <div><h2>Octree / libimagequant</h2><p>原生实现；centroid 输出不是 observed pixel</p></div>
        <div><h2>Color Thief v3</h2><p>官方 {data.external_audit.color_thief.exact_audit.version} · OKLCH · q10</p></div>
        <div><h2>本次实验</h2><p>{data.scope.source_images.toLocaleString()} 张独立 S3 原图 · {data.scope.real_crops.toLocaleString()} crops · 7 方法</p></div>
      </section>

      <section className="go-band">
        <h2>Conditional Go：保留 observed 语义，吸收 q10 / native quantizer 的吞吐优势，再进 1B pilot</h2>
        <div>
          <a className="secondary-action" href="/benchmark.json" download><DownloadIcon />下载 benchmark.json</a>
          <button type="button" className="secondary-action" onClick={() => setCommandsOpen(true)}><TerminalIcon />查看复现命令</button>
        </div>
      </section>
      {commandsOpen && <CommandsDialog onClose={() => setCommandsOpen(false)} />}
    </main>
  );
}

function Footer() {
  return (
    <footer className="app-footer">
      <span>HEX / 100B feasibility lab</span>
      <span>结果生成于 {new Date(data.generated_at).toISOString().slice(0, 10)}</span>
      <span><a href={data.external_audit.color_thief.url} target="_blank" rel="noreferrer">Color Thief</a> · <a href={data.external_audit.pngquant.url} target="_blank" rel="noreferrer">pngquant</a> · <a href={data.external_audit.octree.url} target="_blank" rel="noreferrer">Octree</a> · <a href={data.external_audit.pixelero.url} target="_blank" rel="noreferrer">Pixelero</a></span>
    </footer>
  );
}

export default function App() {
  const [active, setActive] = useState('overview');
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, [active]);
  if (!data) return <div className="data-error">benchmark data 未加载。请先运行 scripts/run_benchmark.py。</div>;
  const pages = {
    overview: <Overview onNavigate={setActive} />,
    methods: <MethodsView />,
    implementation: <ImplementationView />,
    samples: <SamplesView />,
    efficiency: <EfficiencyView />,
    scale: <ScaleView />,
  };
  return (
    <div className="app-shell">
      <Header active={active} onNavigate={setActive} />
      {pages[active]}
      <Footer />
    </div>
  );
}
