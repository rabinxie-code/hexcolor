# HEX / 100B dashboard design spec

Accepted concepts: `overview.png`, `samples.png`, `scale.png` (1536×1024 each).

## System

- Background: near-white `#fdfdfc`; surfaces remain white, never cream or gray.
- Text: `#0b0b0d`; secondary `#666970`; rules `#d7d9dd`.
- Primary accent: cobalt `#075ff7`; selected fill `#f4f8ff`.
- Warning: `#ef2929`; supporting route colors blue, teal, violet, orange.
- Typography: Helvetica/Arial/PingFang-style Swiss grotesk; data uses SFMono/Menlo.
- Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64 px. Desktop gutters 28–40 px.
- Geometry: square to 4 px radii, hairline borders, almost no shadow.
- Container model: open horizontal bands, table rails and one bordered benchmark surface; no bento grid or nested cards.

## Screens and locked first-viewport copy

- Header: `HX`, `HEX / 100B 可行性实验`, `结论`, `方法`, `样本`, `效率`, `规模化`, `数据口径`.
- Overview: `Adaptive Hex v1 仍是生产语义方向`; second line separates native Octree throughput, Color Thief q10 sampling, palette coverage, observed semantics, and licensing.
- Samples: `逐图结果`; `同一 crop 并排比较主 HEX、top-3 palette、路由与耗时。`
- Scale: `100B 规模判断`; scale warning and final Go recommendation from concept.

## Components

- Quiet sticky header with active cobalt underline and horizontally scrollable mobile nav.
- Sortable ruled comparison table, single selected row, radio-style selection marker.
- Chromatic swatches aligned to a thin tick ruler; palette strips have no decorative container.
- Native selects/buttons with deliberate 14 px control typography and 2 px radius.
- Sample matrix with real crop frames, seven method columns, one selectable row, expanded detail rail, filters and pagination.
- Scale equation rail, five-stage pipeline and red rejected neural branch.
- Modal-style data-scope and reproduction-command drawers when controls are activated.

## Responsive behavior

- At 900 px, metric rails wrap to two columns and tables retain horizontal scroll.
- At 680 px, header brand stays fixed, nav scrolls, sample comparison becomes one crop followed by method rows, and pipeline stacks vertically.
- No clipped controls, hidden data, or mobile-only copy changes.

## Icon inventory

- `HX` is a text mark.
- Chevron, download, code/terminal and table glyphs are small custom stroke SVGs at 16–18 px, 1.5 px stroke, square line caps where practical.
- No decorative icon set.
