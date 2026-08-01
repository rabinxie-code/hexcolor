#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import {
  MmcqQuantizer,
  NodePixelLoader,
  extractPalette,
  validateOptions,
} from 'colorthief/internals';

const inputPath = process.argv[2];
if (!inputPath) {
  throw new Error('usage: node scripts/benchmark_colorthief.mjs INPUT.json');
}

const packageInfo = JSON.parse(
  await readFile(new URL('../node_modules/colorthief/package.json', import.meta.url), 'utf8'),
);
const input = JSON.parse(await readFile(inputPath, 'utf8'));
const loader = new NodePixelLoader();
const quantizer = new MmcqQuantizer();
await quantizer.init();

// These are Color Thief v3 defaults except colorCount, fixed to the shared
// top-3 output budget used by this benchmark.
const options = validateOptions({
  colorCount: 3,
  quality: 10,
  colorSpace: 'oklch',
  ignoreWhite: true,
});

function normalize(palette) {
  const safePalette = palette ?? [];
  return {
    primary: safePalette[0]?.array() ?? [0, 0, 0],
    palette: safePalette.map((color) => color.array()),
    weights: safePalette.map((color) => color.proportion),
    populations: safePalette.map((color) => color.population),
  };
}

async function extract(path, timed = false) {
  const start = performance.now();
  const pixels = await loader.load(path);
  const decoded = performance.now();
  const palette = extractPalette(
    pixels.data,
    pixels.width,
    pixels.height,
    options,
    quantizer,
    pixels.colorSpace ?? 'srgb',
  );
  const finished = performance.now();
  return {
    ...normalize(palette),
    ...(timed
      ? {
          decode_ms: decoded - start,
          method_ms: finished - decoded,
          elapsed_ms: finished - start,
          width: pixels.width,
          height: pixels.height,
        }
      : {}),
  };
}

if (input.items.length > 0) {
  // Match the Python benchmark: warm decoder/module allocations and the first
  // quantization before collecting per-crop measurements.
  await extract(input.items[0].path);
}

const records = [];
for (const item of input.items) {
  records.push({
    id: item.id,
    kind: item.kind,
    base: await extract(item.path, item.kind === 'real'),
    jpeg: item.jpeg_path ? await extract(item.jpeg_path) : null,
    resize: item.resize_path ? await extract(item.resize_path) : null,
  });
}

process.stdout.write(
  JSON.stringify({
    package: 'colorthief',
    version: packageInfo.version,
    runtime: process.version,
    options: {
      colorCount: options.colorCount,
      quality: options.quality,
      colorSpace: options.colorSpace,
      ignoreWhite: options.ignoreWhite,
      whiteThreshold: options.whiteThreshold,
      alphaThreshold: options.alphaThreshold,
    },
    records,
  }),
);
