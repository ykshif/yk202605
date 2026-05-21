#!/usr/bin/env python3
"""Local browser UI for standardized Capytaine hydrodynamic NetCDF runs."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json
import math
import sys
import threading
import traceback
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    ArrayHydrodynamicsConfig,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    StructuralGridSpec,
    degrees_to_radians,
    module_structural_node_mappings,
    omega_values_from_wavelengths,
    omega_values_from_range,
    parse_float_sequence,
    preview_layout,
    run_array_hydrodynamics,
)


DEFAULT_OUTPUT = "results/hydrodynamics_ui/array_hydrodynamics.nc"
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RODM 水动力计算窗口</title>
  <link rel="stylesheet" href="/app.css">
  <script src="/app.js" defer></script>
</head>
<body>
  <main class="app-shell">
    <aside class="controls" aria-label="水动力计算参数">
      <header class="app-title">
        <div class="title-mark">H</div>
        <div>
          <p>RODM Hydrodynamics</p>
          <h1>水动力计算窗口</h1>
        </div>
      </header>

      <form id="hydroForm" class="control-form">
        <section class="section">
          <h2>浮体模块</h2>
          <div class="field-grid two">
            <label>单模块长度 m<input name="length_m" type="number" step="0.1" value="30"></label>
            <label>宽度 m<input name="width_m" type="number" step="0.1" value="60"></label>
            <label>高度 m<input name="height_m" type="number" step="0.1" value="2"></label>
            <label>吃水 m<input name="draft_m" type="number" step="0.1" value="0.5"></label>
            <label>水平网格 m<input name="mesh_size_m" type="number" step="0.1" value="6"></label>
            <label>竖向网格 m<input name="vertical_mesh_size_m" type="number" step="0.1" value="0.8"></label>
          </div>
          <label class="full">质量 kg<input name="mass_kg" type="number" step="1" placeholder="留空则按 rho * 排水体积"></label>
        </section>

        <section class="section">
          <h2>阵列布局</h2>
          <div class="field-grid two">
            <label>总长度 m<input name="total_length_m" type="number" step="0.1" value="300"></label>
            <label>列数<input name="columns" type="number" min="1" step="1" value="10"></label>
            <label>行数<input name="rows" type="number" min="1" step="1" value="1"></label>
            <label>X 间距 m<input name="spacing_x_m" type="number" step="0.1" value="30"></label>
            <label>Y 间距 m<input name="spacing_y_m" type="number" step="0.1" value="60"></label>
            <label>划分模式<select name="division_mode">
              <option value="uniform">Uniform division</option>
              <option value="custom">Custom non-uniform division</option>
              <option value="random">Random non-uniform division</option>
            </select></label>
            <label>随机种子<input name="random_seed" type="number" step="1" value="42"></label>
            <label>FEM dx m<input name="structural_grid_dx_m" type="number" step="0.1" value="5"></label>
            <label>FEM dy m<input name="structural_grid_dy_m" type="number" step="0.1" value="5"></label>
          </div>
          <label class="check-row"><input name="align_centers_to_structural_grid" type="checkbox" checked> Align module centers to 5 m FEM nodes</label>
          <label class="full">自定义模块长度 m<input name="module_lengths_x_m" type="text" value="20, 40, 30, 30, 20, 40, 20, 40, 30, 30"></label>
          <label class="full command-label">模块划分结果<textarea id="divisionSummary" spellcheck="false" readonly></textarea></label>
        </section>

        <section class="section">
          <h2>海况与求解</h2>
          <div class="field-grid two">
            <label>水深 m<input name="water_depth_m" type="number" step="0.1" value="58.5"></label>
            <label>密度 kg/m3<input name="rho" type="number" step="1" value="1025"></label>
            <label>重力 m/s2<input name="g" type="number" step="0.01" value="9.81"></label>
            <label>并行核数<input name="n_jobs" type="number" min="1" step="1" value="1"></label>
          </div>
          <label class="check-row"><input name="infinite_depth" type="checkbox"> 无限水深</label>
          <label class="full">波浪方向 deg<input name="wave_directions_deg" type="text" value="0" placeholder="例如 0, 90, 180"></label>
        </section>

        <section class="section">
          <h2>频率</h2>
          <div class="mode-row" role="radiogroup" aria-label="频率输入模式">
            <label><input type="radio" name="omega_mode" value="single" checked> 单频</label>
            <label><input type="radio" name="omega_mode" value="range"> 范围</label>
            <label><input type="radio" name="omega_mode" value="list"> 列表</label>
            <label><input type="radio" name="omega_mode" value="wavelength"> 波长</label>
          </div>
          <div class="omega-panel" data-mode="single">
            <label class="full">omega rad/s<input name="omega_single" type="number" step="0.0001" value="0.5851"></label>
          </div>
          <div class="omega-panel is-hidden" data-mode="range">
            <div class="field-grid three">
              <label>起点<input name="omega_start" type="number" step="0.0001" value="0.1"></label>
              <label>终点<input name="omega_stop" type="number" step="0.0001" value="2.0"></label>
              <label>数量<input name="omega_count" type="number" min="1" step="1" value="40"></label>
            </div>
          </div>
          <div class="omega-panel is-hidden" data-mode="list">
            <label class="full">omega 列表<input name="omega_values" type="text" value="0.5851" placeholder="例如 0.4, 0.5851, 0.8"></label>
          </div>
          <div class="omega-panel is-hidden" data-mode="wavelength">
            <label class="full">波长列表 m<input name="wavelength_values_m" type="text" value="300" placeholder="例如 180, 240, 300"></label>
          </div>
        </section>

        <section class="section">
          <h2>输出</h2>
          <label class="full">NetCDF 文件<input name="output_path" type="text" value="results/hydrodynamics_ui/array_hydrodynamics.nc"></label>
          <label class="full command-label">命令框<textarea id="commandBox" spellcheck="false"></textarea></label>
          <div class="action-row">
            <button id="runButton" type="button">计算 .nc</button>
            <button id="resetButton" type="button" class="ghost">恢复默认</button>
          </div>
        </section>

        <section class="section">
          <h2>运动预览</h2>
          <div class="field-grid three">
            <label>波幅 m<input name="visual_wave_amplitude_m" type="number" step="0.1" value="1"></label>
            <label>运动倍率<input name="visual_motion_scale" type="number" min="0" step="0.5" value="4"></label>
            <label>播放速度<input name="visual_speed" type="number" min="0.1" step="0.1" value="1"></label>
          </div>
        </section>
      </form>
    </aside>

    <section class="workspace" aria-label="浮体阵列预览">
      <header class="workspace-top">
        <div>
          <p>Array Preview</p>
          <h2 id="previewTitle">3 x 3 modules</h2>
        </div>
        <div class="motion-pill"><span></span><strong id="motionMode">参数预览</strong></div>
        <div class="metric-strip">
          <span><b id="metricBodies">9</b> 浮体</span>
          <span><b id="metricDofs">54</b> DOF</span>
          <span><b id="metricProblems">55</b> BEM</span>
          <span><b id="metricRao">待算</b> RAO</span>
        </div>
      </header>
      <div class="preview-stage">
        <canvas id="arrayCanvas" aria-label="阵列预览"></canvas>
        <div class="stage-overlay">
          <span id="waveReadout">omega 0.5851 rad/s</span>
          <span id="motionReadout">unit-wave motion preview</span>
        </div>
      </div>
      <div class="status-bar">
        <div>
          <p>状态</p>
          <strong id="jobStatus">待计算</strong>
        </div>
        <div>
          <p>输出文件</p>
          <strong id="outputPath">results/hydrodynamics_ui/array_hydrodynamics.nc</strong>
        </div>
      </div>
      <pre id="logBox" class="log-box"></pre>
    </section>
  </main>
</body>
</html>
"""


APP_CSS = r""":root {
  color-scheme: light;
  --ink: #17201c;
  --muted: #66736f;
  --line: #d6ddd9;
  --paper: #fbfcfa;
  --control: #111816;
  --control-ink: #edf6f1;
  --control-muted: #aabbb3;
  --water: #eaf2f4;
  --accent: #0b6b63;
  --accent-strong: #064d48;
  --coral: #c86b4f;
  --module: #d7a84f;
  --module-edge: #6d4c16;
  --shadow: 0 18px 50px rgba(20, 42, 38, 0.16);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  background: linear-gradient(135deg, #eef6f4 0%, #d9edf1 48%, #f7f1e7 100%);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button, input, select, textarea { font: inherit; }

.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(360px, 460px) minmax(0, 1fr);
}

.controls {
  height: 100vh;
  overflow: auto;
  background: var(--control);
  color: var(--control-ink);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  padding: 24px;
  box-shadow: 18px 0 45px rgba(7, 18, 16, 0.18);
}

.app-title {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 18px;
}

.title-mark {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(255, 255, 255, 0.26);
  border-radius: 8px;
  background: #d7a84f;
  color: #151914;
  font-weight: 800;
}

.app-title p,
.workspace-top p,
.status-bar p {
  margin: 0 0 4px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0;
}

.app-title p {
  color: var(--control-muted);
}

.app-title h1,
.workspace-top h2 {
  margin: 0;
  font-size: 24px;
  line-height: 1.2;
}

.app-title h1 {
  color: var(--control-ink);
}

.control-form {
  display: grid;
  gap: 14px;
}

.section {
  border-top: 1px solid rgba(255, 255, 255, 0.12);
  padding-top: 14px;
}

.section h2 {
  margin: 0 0 10px;
  font-size: 15px;
  color: var(--control-ink);
}

.field-grid {
  display: grid;
  gap: 10px;
}

.field-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.field-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }

label {
  display: grid;
  gap: 5px;
  min-width: 0;
  color: var(--control-muted);
  font-size: 12px;
}

label.full {
  margin-top: 10px;
}

input, select, textarea {
  width: 100%;
  min-width: 0;
  border: 1px solid #c8d1cc;
  border-radius: 6px;
  background: var(--paper);
  color: var(--ink);
  outline: none;
}

input, select {
  height: 34px;
  padding: 6px 9px;
}

textarea {
  min-height: 150px;
  resize: vertical;
  padding: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.45;
}

input:focus,
textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(11, 107, 99, 0.14);
}

.check-row,
.mode-row {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.check-row input,
.mode-row input {
  width: auto;
  height: auto;
}

.mode-row label {
  display: inline-flex;
  flex-direction: row;
  gap: 6px;
  align-items: center;
  min-height: 32px;
  padding: 0 10px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.06);
}

.omega-panel.is-hidden {
  display: none;
}

.action-row {
  display: grid;
  grid-template-columns: 1fr 120px;
  gap: 10px;
  margin-top: 12px;
}

button {
  height: 40px;
  border: 1px solid #2f9288;
  border-radius: 6px;
  background: #0f7f75;
  color: white;
  cursor: pointer;
  box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
}

button:hover { background: var(--accent-strong); }
button:disabled { opacity: 0.58; cursor: wait; }
button.ghost {
  background: rgba(255, 255, 255, 0.06);
  color: var(--control-ink);
  box-shadow: none;
}

.workspace {
  min-width: 0;
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto minmax(360px, 1fr) auto 190px;
  gap: 18px;
  padding: 24px;
}

.workspace-top,
.status-bar {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
}

.metric-strip {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.metric-strip span {
  min-width: 108px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.82);
  padding: 8px 10px;
  color: var(--muted);
  box-shadow: 0 8px 24px rgba(22, 40, 37, 0.08);
}

.metric-strip b {
  color: var(--ink);
  font-size: 18px;
}

.preview-stage {
  position: relative;
  min-height: 360px;
  border: 1px solid rgba(67, 101, 103, 0.28);
  border-radius: 8px;
  background: #d8eef1;
  overflow: hidden;
  box-shadow: var(--shadow);
}

#arrayCanvas {
  width: 100%;
  height: 100%;
  display: block;
}

.status-bar {
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 12px 0;
  background: rgba(255, 255, 255, 0.36);
}

.status-bar > div {
  min-width: 0;
}

.status-bar strong {
  display: block;
  max-width: 100%;
  overflow-wrap: anywhere;
  font-size: 14px;
}

.log-box {
  margin: 0;
  min-height: 160px;
  max-height: 190px;
  overflow: auto;
  border: 1px solid #cbd4d0;
  border-radius: 6px;
  background: #121816;
  color: #dce8df;
  padding: 12px;
  font-size: 12px;
  line-height: 1.5;
}

.motion-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid rgba(11, 107, 99, 0.24);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.74);
  color: var(--accent-strong);
  box-shadow: 0 8px 24px rgba(22, 40, 37, 0.08);
}

.motion-pill span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--coral);
  box-shadow: 0 0 0 4px rgba(200, 107, 79, 0.16);
}

.stage-overlay {
  position: absolute;
  left: 14px;
  right: 14px;
  bottom: 14px;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 8px;
  pointer-events: none;
}

.stage-overlay span {
  border: 1px solid rgba(255, 255, 255, 0.54);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.72);
  color: #24423e;
  padding: 7px 10px;
  font-size: 12px;
  backdrop-filter: blur(8px);
}

@media (max-width: 920px) {
  .app-shell {
    grid-template-columns: 1fr;
  }
  .controls {
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .workspace {
    min-height: 720px;
  }
  .workspace-top {
    display: grid;
    grid-template-columns: 1fr;
    align-items: start;
  }
  .motion-pill {
    justify-self: start;
    max-width: 100%;
    white-space: nowrap;
  }
  .metric-strip {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    justify-content: stretch;
    width: 100%;
  }
  .metric-strip span {
    min-width: 0;
  }
  .status-bar {
    display: grid;
    grid-template-columns: 1fr;
  }
}
"""


APP_JS = r"""const form = document.getElementById("hydroForm");
const commandBox = document.getElementById("commandBox");
const canvas = document.getElementById("arrayCanvas");
const runButton = document.getElementById("runButton");
const resetButton = document.getElementById("resetButton");
const logBox = document.getElementById("logBox");
const jobStatus = document.getElementById("jobStatus");
const outputPath = document.getElementById("outputPath");
const motionMode = document.getElementById("motionMode");
const metricRao = document.getElementById("metricRao");
const waveReadout = document.getElementById("waveReadout");
const motionReadout = document.getElementById("motionReadout");
const divisionSummary = document.getElementById("divisionSummary");

const defaults = {};
let raoPreview = null;
let raoBodyMap = new Map();
let raoSignature = "";
let lastPayload = null;
let animationStartedAt = performance.now();

for (const element of form.elements) {
  if (!element.name) continue;
  if (element.type === "checkbox" || element.type === "radio") {
    defaults[element.name + ":" + element.value] = element.checked;
  } else {
    defaults[element.name] = element.value;
  }
}

function value(name) {
  return form.elements[name].value.trim();
}

function numberValue(name, fallback = 0) {
  const raw = value(name);
  if (raw === "") return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function intValue(name, fallback = 1) {
  return Math.max(1, Math.round(numberValue(name, fallback)));
}

function parseNumbers(text) {
  return text.split(/[\s,;]+/).map((item) => item.trim()).filter(Boolean).map(Number).filter(Number.isFinite);
}

function currentOmegaMode() {
  return form.querySelector("input[name='omega_mode']:checked").value;
}

function currentDivisionMode() {
  return value("division_mode") || "uniform";
}

function seededRandom(seed) {
  let state = Math.trunc(Math.abs(seed || 1)) % 2147483647;
  if (state === 0) state = 1;
  return () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  };
}

function randomModuleLengths(totalLength, count, seed) {
  const random = seededRandom(seed);
  const weights = Array.from({ length: count }, () => 0.25 + random());
  const weightSum = weights.reduce((sum, item) => sum + item, 0);
  const raw = weights.map((item) => (item / weightSum) * totalLength);
  const rounded = raw.map((item) => Number(item.toFixed(6)));
  const correction = Number((totalLength - rounded.reduce((sum, item) => sum + item, 0)).toFixed(6));
  rounded[rounded.length - 1] = Number((rounded[rounded.length - 1] + correction).toFixed(6));
  return rounded;
}

function randomGridAlignedModuleLengths(totalLength, count, gridDx, seed) {
  const unit = 2 * gridDx;
  const totalUnits = Math.round(totalLength / unit);
  if (!Number.isFinite(totalUnits) || Math.abs(totalUnits * unit - totalLength) > 1e-6) {
    throw new Error("Total length must be divisible by 2 * FEM dx for node-centered random division.");
  }
  if (totalUnits < count) {
    throw new Error("Too many modules for the requested FEM grid spacing.");
  }
  const random = seededRandom(seed);
  const base = Math.floor(totalUnits / count);
  const remainder = totalUnits - base * count;
  const units = Array.from({ length: count }, () => base);
  const indices = Array.from({ length: count }, (_, index) => index);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  for (let i = 0; i < remainder; i += 1) {
    units[indices[i]] += 1;
  }
  const minUnit = Math.max(1, base - 1);
  const maxUnit = Math.ceil(totalUnits / count) + 1;
  const transferCount = Math.min(Math.floor(count * 0.4), count - remainder);
  const donorOrder = indices.filter((index) => units[index] > minUnit);
  const receiverOrder = indices.filter((index) => units[index] < maxUnit);
  let transfers = 0;
  for (const donor of donorOrder) {
    const receiver = receiverOrder.find((index) => index !== donor && units[index] < maxUnit);
    if (receiver === undefined) break;
    units[donor] -= 1;
    units[receiver] += 1;
    transfers += 1;
    if (transfers >= transferCount) break;
  }
  return units.map((item) => Number((item * unit).toFixed(6)));
}

function moduleLengthsForPayload(mode, totalLength, count) {
  if (mode === "custom") {
    return parseNumbers(value("module_lengths_x_m"));
  }
  if (mode === "random") {
    if (form.elements.align_centers_to_structural_grid.checked) {
      return randomGridAlignedModuleLengths(
        totalLength,
        count,
        numberValue("structural_grid_dx_m", 5),
        numberValue("random_seed", 42)
      );
    }
    return randomModuleLengths(totalLength, count, numberValue("random_seed", 42));
  }
  return Array.from({ length: count }, () => totalLength / count);
}

function boundariesFromLengths(lengths) {
  const boundaries = [0];
  for (const length of lengths) {
    boundaries.push(boundaries[boundaries.length - 1] + length);
  }
  return boundaries.map((item) => Number(item.toFixed(6)));
}

function formatList(values) {
  return values.map((item) => Number(item).toFixed(6).replace(/\.?0+$/, "")).join(", ");
}

function structuralNodeMappings(payload, centers) {
  if (!payload.layout.align_centers_to_structural_grid) return [];
  const dx = payload.layout.structural_grid_dx_m || 5;
  const dy = payload.layout.structural_grid_dy_m || 5;
  const totalLength = payload.layout.total_length_m || 300;
  const width = payload.module.width_m || 60;
  const nodesPerX = Math.round(totalLength / dx) + 1;
  const yIndex = Math.round((0 + width / 2) / dy);
  return centers.map((center) => {
    const xIndex = Math.round(center / dx);
    const nodeXIndex = nodesPerX - 1 - xIndex;
    return yIndex * nodesPerX + nodeXIndex + 1;
  });
}

function buildPayload() {
  const mode = currentOmegaMode();
  const omega = { mode };
  if (mode === "single") {
    omega.single_rad_s = numberValue("omega_single", 0.5851);
  } else if (mode === "range") {
    omega.start_rad_s = numberValue("omega_start", 0.1);
    omega.stop_rad_s = numberValue("omega_stop", 2.0);
    omega.count = intValue("omega_count", 40);
  } else if (mode === "list") {
    omega.values_rad_s = parseNumbers(value("omega_values"));
  } else {
    omega.wavelength_values_m = parseNumbers(value("wavelength_values_m"));
  }

  const massText = value("mass_kg");
  const waterDepthInfinite = form.elements.infinite_depth.checked;
  const divisionMode = currentDivisionMode();
  const columns = intValue("columns", 10);
  const totalLength = numberValue("total_length_m", 300);
  const moduleLengths = moduleLengthsForPayload(divisionMode, totalLength, columns);
  const nominalModuleLength = totalLength / columns;
  const alignCentersToGrid = form.elements.align_centers_to_structural_grid.checked;
  const layout = {
    rows: intValue("rows", 1),
    columns,
    spacing_x_m: nominalModuleLength,
    spacing_y_m: numberValue("spacing_y_m", 60),
    division_mode: divisionMode,
    total_length_m: totalLength,
    align_centers_to_structural_grid: alignCentersToGrid,
    structural_grid_dx_m: numberValue("structural_grid_dx_m", 5),
    structural_grid_dy_m: numberValue("structural_grid_dy_m", 5)
  };
  if (divisionMode !== "uniform") {
    layout.module_lengths_x_m = moduleLengths;
  }
  if (divisionMode === "random") {
    layout.random_seed = Math.trunc(numberValue("random_seed", 42));
  }
  return {
    module: {
      length_m: nominalModuleLength,
      width_m: numberValue("width_m", 60),
      height_m: numberValue("height_m", 2),
      draft_m: numberValue("draft_m", 0.5),
      mesh_size_m: numberValue("mesh_size_m", 6),
      vertical_mesh_size_m: numberValue("vertical_mesh_size_m", 0.8),
      mass_kg: massText === "" ? null : Number(massText)
    },
    layout,
    hydro: {
      rho: numberValue("rho", 1025),
      g: numberValue("g", 9.81),
      n_jobs: intValue("n_jobs", 1),
      water_depth_m: waterDepthInfinite ? null : numberValue("water_depth_m", 58.5),
      wave_directions_deg: parseNumbers(value("wave_directions_deg")),
      omega
    },
    visual: {
      wave_amplitude_m: numberValue("visual_wave_amplitude_m", 1),
      motion_scale: numberValue("visual_motion_scale", 4),
      speed: numberValue("visual_speed", 1)
    },
    output_path: value("output_path") || "results/hydrodynamics_ui/array_hydrodynamics.nc"
  };
}

function calculationSignature(payload) {
  return JSON.stringify({
    module: payload.module,
    layout: payload.layout,
    hydro: payload.hydro,
    output_path: payload.output_path
  });
}

function omegaCount(payload) {
  const omega = payload.hydro.omega;
  if (omega.mode === "range") return Math.max(1, Math.round(omega.count || 1));
  if (omega.mode === "list") return Math.max(1, omega.values_rad_s.length);
  return 1;
}

function firstOmega(payload) {
  const omega = payload.hydro.omega;
  if (omega.mode === "range") return omega.start_rad_s || 0.5851;
  if (omega.mode === "list") return omega.values_rad_s[0] || 0.5851;
  if (omega.mode === "wavelength") {
    const wavelength = omega.wavelength_values_m[0] || 300;
    const k = 2 * Math.PI / wavelength;
    const depthFactor = payload.hydro.water_depth_m === null ? 1 : Math.tanh(k * payload.hydro.water_depth_m);
    return Math.sqrt(payload.hydro.g * k * depthFactor);
  }
  return omega.single_rad_s || 0.5851;
}

function refreshCommand() {
  const payload = buildPayload();
  const signature = calculationSignature(payload);
  if (raoPreview && raoSignature && signature !== raoSignature) {
    clearRaoPreview();
  }
  commandBox.value = JSON.stringify(payload, null, 2);
  updatePreview(payload);
}

function updateModePanels() {
  const mode = currentOmegaMode();
  document.querySelectorAll(".omega-panel").forEach((panel) => {
    panel.classList.toggle("is-hidden", panel.dataset.mode !== mode);
  });
}

function updatePreview(payload) {
  lastPayload = payload;
  const rows = payload.layout.rows;
  const columns = payload.layout.columns;
  const bodies = rows * columns;
  const dofs = bodies * 6;
  const waveDirections = Math.max(1, payload.hydro.wave_directions_deg.length);
  const problems = dofs * omegaCount(payload) + omegaCount(payload) * waveDirections;
  updateDivisionSummary(payload);
  document.getElementById("previewTitle").textContent = `${columns} x ${rows} modules`;
  document.getElementById("metricBodies").textContent = bodies;
  document.getElementById("metricDofs").textContent = dofs;
  document.getElementById("metricProblems").textContent = problems;
  outputPath.textContent = payload.output_path;
  waveReadout.textContent = `omega ${firstOmega(payload).toFixed(4)} rad/s`;
  updateMotionLabels(payload);
}

function updateDivisionSummary(payload) {
  const lengths = payload.layout.module_lengths_x_m || moduleLengthsForPayload(
    payload.layout.division_mode,
    payload.layout.total_length_m,
    payload.layout.columns
  );
  const boundaries = boundariesFromLengths(lengths);
  const centers = lengths.map((_, index) => Number(((boundaries[index] + boundaries[index + 1]) / 2).toFixed(6)));
  const femNodes = structuralNodeMappings(payload, centers);
  const sum = lengths.reduce((total, item) => total + item, 0);
  if (divisionSummary) {
    const lines = [
      `Mode: ${payload.layout.division_mode}`,
      `Layout: ${payload.layout.columns} x ${payload.layout.rows}`,
      `Module lengths: [${formatList(lengths)}]`,
      `Module boundaries: [${formatList(boundaries)}]`,
      `Module centers: [${formatList(centers)}]`,
      `Length sum: ${sum.toFixed(6).replace(/\.?0+$/, "")} m`
    ];
    if (femNodes.length) {
      lines.push(`FEM node ids: [${femNodes.join(", ")}]`);
    }
    divisionSummary.value = lines.join("\n");
  }
}

function updateMotionLabels(payload) {
  if (raoPreview) {
    motionMode.textContent = "RAO 驱动";
    metricRao.textContent = "已算";
    const maxHeave = maxDofAbs("Heave");
    const maxPitch = maxDofAbs("Pitch");
    motionReadout.textContent = `max |Heave RAO| ${maxHeave.toFixed(3)}, max |Pitch RAO| ${maxPitch.toFixed(3)}`;
  } else {
    motionMode.textContent = "参数预览";
    metricRao.textContent = "待算";
    motionReadout.textContent = `wave amplitude ${payload.visual.wave_amplitude_m.toFixed(2)} m, scale ${payload.visual.motion_scale.toFixed(1)}`;
  }
}

function setRaoPreview(summary, signature) {
  raoPreview = summary || null;
  raoSignature = summary ? signature : "";
  raoBodyMap = new Map();
  if (raoPreview && Array.isArray(raoPreview.bodies)) {
    for (const body of raoPreview.bodies) {
      raoBodyMap.set(body.name, body.dofs || {});
    }
  }
  updateMotionLabels(lastPayload || buildPayload());
}

function clearRaoPreview() {
  raoPreview = null;
  raoSignature = "";
  raoBodyMap = new Map();
}

function maxDofAbs(dofName) {
  let maxValue = 0;
  for (const dofs of raoBodyMap.values()) {
    const item = dofs[dofName];
    if (item && Number.isFinite(item.abs)) maxValue = Math.max(maxValue, item.abs);
  }
  return maxValue;
}

function animate(now) {
  const payload = lastPayload || buildPayload();
  const elapsed = ((now - animationStartedAt) / 1000) * Math.max(0.1, payload.visual.speed);
  drawArray(payload, elapsed);
  requestAnimationFrame(animate);
}

function drawArray(payload, elapsed) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const rows = payload.layout.rows;
  const cols = payload.layout.columns;
  const width = payload.module.width_m;
  const sy = payload.layout.spacing_y_m;
  const lengths = payload.layout.module_lengths_x_m || moduleLengthsForPayload(
    payload.layout.division_mode || "uniform",
    payload.layout.total_length_m || payload.module.length_m * cols,
    cols
  );
  const boundaries = boundariesFromLengths(lengths);
  const spanX = boundaries[boundaries.length - 1];
  const spanY = (rows - 1) * sy + width;
  const margin = 54;
  const scale = Math.min((rect.width - 2 * margin) / Math.max(spanX, 1), (rect.height - 2 * margin) / Math.max(spanY, 1));
  const cx = rect.width / 2;
  const cy = rect.height / 2 + 8;
  const moduleH = Math.max(12, width * scale);

  drawWaveField(ctx, rect.width, rect.height, payload, elapsed);

  const modules = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const worldX = ((boundaries[c] + boundaries[c + 1]) / 2) - spanX / 2;
      const worldY = (r - (rows - 1) / 2) * sy;
      const name = `${c}_${r}`;
      const motion = bodyMotion(name, worldX, worldY, payload, elapsed);
      modules.push({
        name,
        x: cx + worldX * scale,
        y: cy + worldY * scale,
        moduleW: Math.max(12, lengths[c] * scale),
        motion
      });
    }
  }

  modules.sort((a, b) => a.y - b.y);
  const showLabels = rows * cols <= 100 && moduleH > 18;
  for (const module of modules) {
    drawModule(ctx, module, module.moduleW, moduleH, scale, payload, showLabels);
  }
}

function drawWaveField(ctx, width, height, payload, elapsed) {
  const water = ctx.createLinearGradient(0, 0, 0, height);
  water.addColorStop(0, "#cce7ec");
  water.addColorStop(0.52, "#e8f4f3");
  water.addColorStop(1, "#b9dfe8");
  ctx.fillStyle = water;
  ctx.fillRect(0, 0, width, height);

  const omega = raoPreview?.omega_rad_s || firstOmega(payload);
  const phase = elapsed * omega;
  const visualAmp = Math.max(7, Math.min(38, payload.visual.wave_amplitude_m * payload.visual.motion_scale * 7));
  for (let band = 0; band < 10; band += 1) {
    const y0 = height * (0.15 + band * 0.085);
    ctx.beginPath();
    for (let x = -20; x <= width + 20; x += 8) {
      const y = y0 + Math.sin(x * 0.025 - phase - band * 0.45) * visualAmp * (0.35 + band * 0.045);
      if (x === -20) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = band % 2 === 0 ? "rgba(20, 112, 129, 0.22)" : "rgba(255, 255, 255, 0.48)";
    ctx.lineWidth = band % 2 === 0 ? 1.4 : 1.0;
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(13, 44, 47, 0.06)";
  for (let i = 0; i < 5; i += 1) {
    ctx.fillRect(0, height * (0.18 + i * 0.17), width, 1);
  }
}

function bodyMotion(name, worldX, worldY, payload, elapsed) {
  const omega = raoPreview?.omega_rad_s || firstOmega(payload);
  const waveAmp = payload.visual.wave_amplitude_m;
  if (raoPreview && raoBodyMap.has(name)) {
    const dofs = raoBodyMap.get(name);
    return {
      surge: evalComplex(dofs.Surge, omega, elapsed) * waveAmp,
      sway: evalComplex(dofs.Sway, omega, elapsed) * waveAmp,
      heave: evalComplex(dofs.Heave, omega, elapsed) * waveAmp,
      roll: evalComplex(dofs.Roll, omega, elapsed) * waveAmp,
      pitch: evalComplex(dofs.Pitch, omega, elapsed) * waveAmp,
      yaw: evalComplex(dofs.Yaw, omega, elapsed) * waveAmp
    };
  }

  const phase = omega * elapsed - worldX * 0.045 - worldY * 0.015;
  return {
    surge: 0.10 * waveAmp * Math.sin(phase),
    sway: 0,
    heave: 0.32 * waveAmp * Math.cos(phase),
    roll: 0.015 * waveAmp * Math.sin(phase + 0.8),
    pitch: 0.022 * waveAmp * Math.sin(phase + 1.2),
    yaw: 0
  };
}

function evalComplex(item, omega, elapsed) {
  if (!item) return 0;
  const amplitude = Number(item.abs || 0);
  const phase = Number(item.phase_rad || 0);
  return amplitude * Math.cos(phase - omega * elapsed);
}

function drawModule(ctx, module, moduleW, moduleH, scale, payload, showLabel) {
  const motionScale = payload.visual.motion_scale;
  const heavePx = clamp(module.motion.heave * scale * motionScale, -86, 86);
  const surgePx = clamp(module.motion.surge * scale * motionScale, -76, 76);
  const swayPx = clamp(module.motion.sway * scale * motionScale, -42, 42);
  const pitch = clamp(module.motion.pitch * motionScale, -0.34, 0.34);
  const rollSkew = clamp(module.motion.roll * motionScale * 22, -18, 18);
  const x = module.x + surgePx + swayPx * 0.24;
  const y = module.y - heavePx;
  const depth = Math.max(8, moduleH * 0.16);

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-pitch);

  ctx.fillStyle = "rgba(26, 48, 45, 0.18)";
  ctx.beginPath();
  ctx.ellipse(0, moduleH * 0.35 + depth + Math.abs(heavePx) * 0.15, moduleW * 0.56, moduleH * 0.24, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#8d6d2d";
  ctx.beginPath();
  ctx.moveTo(-moduleW / 2, moduleH / 2 - rollSkew);
  ctx.lineTo(moduleW / 2, moduleH / 2 + rollSkew);
  ctx.lineTo(moduleW / 2 - 9, moduleH / 2 + depth + rollSkew);
  ctx.lineTo(-moduleW / 2 - 9, moduleH / 2 + depth - rollSkew);
  ctx.closePath();
  ctx.fill();

  const top = ctx.createLinearGradient(-moduleW / 2, -moduleH / 2, moduleW / 2, moduleH / 2);
  top.addColorStop(0, "#f0c66b");
  top.addColorStop(0.55, "#d7a84f");
  top.addColorStop(1, "#b6802f");
  ctx.fillStyle = top;
  ctx.strokeStyle = "#6d4c16";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(-moduleW / 2, -moduleH / 2 - rollSkew);
  ctx.lineTo(moduleW / 2, -moduleH / 2 + rollSkew);
  ctx.lineTo(moduleW / 2, moduleH / 2 + rollSkew);
  ctx.lineTo(-moduleW / 2, moduleH / 2 - rollSkew);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(-moduleW * 0.36, -moduleH * 0.22 - rollSkew * 0.5);
  ctx.lineTo(moduleW * 0.36, -moduleH * 0.22 + rollSkew * 0.5);
  ctx.stroke();

  if (showLabel) {
    ctx.fillStyle = "#1d2117";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(module.name, 0, 0);
  }
  ctx.restore();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function startRun() {
  let payload;
  try {
    payload = JSON.parse(commandBox.value);
  } catch (error) {
    jobStatus.textContent = "命令框 JSON 无效";
    logBox.textContent = String(error);
    return;
  }

  clearRaoPreview();
  updateMotionLabels(payload);
  runButton.disabled = true;
  jobStatus.textContent = "提交中";
  metricRao.textContent = "计算中";
  motionMode.textContent = "计算中";
  logBox.textContent = "";
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) {
    jobStatus.textContent = "提交失败";
    logBox.textContent = data.error || "Unknown error";
    runButton.disabled = false;
    updateMotionLabels(payload);
    return;
  }
  pollJob(data.job_id, calculationSignature(payload));
}

async function pollJob(jobId, signature) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const data = await response.json();
  jobStatus.textContent = data.status || "未知";
  if (data.output_path) outputPath.textContent = data.output_path;
  logBox.textContent = (data.logs || []).join("\n");
  logBox.scrollTop = logBox.scrollHeight;

  if (data.status === "completed" || data.status === "failed") {
    if (data.status === "completed" && data.result && data.result.rao_preview) {
      setRaoPreview(data.result.rao_preview, signature);
    } else {
      updateMotionLabels(lastPayload || buildPayload());
    }
    runButton.disabled = false;
    return;
  }
  setTimeout(() => pollJob(jobId, signature), 1200);
}

function restoreDefaults() {
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox" || element.type === "radio") {
      element.checked = Boolean(defaults[element.name + ":" + element.value]);
    } else if (Object.prototype.hasOwnProperty.call(defaults, element.name)) {
      element.value = defaults[element.name];
    }
  }
  clearRaoPreview();
  updateModePanels();
  refreshCommand();
  jobStatus.textContent = "待计算";
  logBox.textContent = "";
}

form.addEventListener("input", () => {
  updateModePanels();
  refreshCommand();
});
form.addEventListener("change", () => {
  updateModePanels();
  refreshCommand();
});
runButton.addEventListener("click", startRun);
resetButton.addEventListener("click", restoreDefaults);
window.addEventListener("resize", () => updatePreview(buildPayload()));

updateModePanels();
refreshCommand();
requestAnimationFrame(animate);
"""


class HydroUIHandler(BaseHTTPRequestHandler):
    server_version = "RODMHydroUI/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/app.css":
            self._send_text(APP_CSS, "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send_text(APP_JS, "application/javascript; charset=utf-8")
        elif path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "job not found"}, status=404)
            else:
                self._send_json(job)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_head("text/html; charset=utf-8", len(INDEX_HTML.encode("utf-8")))
        elif path == "/app.css":
            self._send_head("text/css; charset=utf-8", len(APP_CSS.encode("utf-8")))
        elif path == "/app.js":
            self._send_head("application/javascript; charset=utf-8", len(APP_JS.encode("utf-8")))
        else:
            self._send_head("application/json; charset=utf-8", 0, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/preview":
            self._handle_preview()
        elif path == "/api/run":
            self._handle_run()
        else:
            self._send_json({"error": "not found"}, status=404)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[hydro-ui] " + format % args + "\n")

    def _handle_preview(self) -> None:
        try:
            payload = self._read_json()
            config = config_from_payload(payload)
            self._send_json(preview_layout(config))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_run(self) -> None:
        try:
            payload = self._read_json()
            config_from_payload(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "logs": ["Job queued"],
                "output_path": None,
            }
        thread = threading.Thread(target=run_job, args=(job_id, payload), daemon=True)
        thread.start()
        self._send_json({"job_id": job_id, "status": "queued"})

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self._send_head(content_type, len(body), status=status)
        self.wfile.write(body)

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        self._send_head("application/json; charset=utf-8", len(body), status=status)
        self.wfile.write(body)

    def _send_head(self, content_type: str, content_length: int, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()


def config_from_payload(payload: dict[str, object]) -> ArrayHydrodynamicsConfig:
    module_data = _mapping(payload.get("module"), "module")
    layout_data = _mapping(payload.get("layout"), "layout")
    hydro_data = _mapping(payload.get("hydro"), "hydro")
    omega_data = _mapping(hydro_data.get("omega"), "hydro.omega")

    module = RectangularModuleSpec(
        length_m=_float(module_data, "length_m"),
        width_m=_float(module_data, "width_m"),
        height_m=_float(module_data, "height_m"),
        draft_m=_float(module_data, "draft_m"),
        mesh_size_m=_float(module_data, "mesh_size_m"),
        vertical_mesh_size_m=_optional_float(module_data, "vertical_mesh_size_m"),
        mass_kg=_optional_float(module_data, "mass_kg"),
    )
    division_mode = str(layout_data.get("division_mode", "uniform")).lower().replace("-", "_")
    total_length_m = _optional_float(layout_data, "total_length_m")
    align_centers_to_grid = _bool(layout_data, "align_centers_to_structural_grid", False)
    structural_grid = None
    if align_centers_to_grid:
        structural_grid = StructuralGridSpec(
            length_m=total_length_m or module.length_m * _int(layout_data, "columns"),
            width_m=module.width_m,
            dx_m=_float(layout_data, "structural_grid_dx_m", 5.0),
            dy_m=_float(layout_data, "structural_grid_dy_m", 5.0),
        )
    module_lengths_x_m = _module_lengths_from_layout(
        layout_data,
        division_mode=division_mode,
        total_length_m=total_length_m or module.length_m * _int(layout_data, "columns"),
        columns=_int(layout_data, "columns"),
        structural_grid_dx_m=structural_grid.dx_m if structural_grid is not None else None,
    )
    layout = ArrayLayoutSpec(
        rows=_int(layout_data, "rows"),
        columns=_int(layout_data, "columns"),
        spacing_x_m=_float(layout_data, "spacing_x_m", module.length_m),
        spacing_y_m=_float(layout_data, "spacing_y_m", module.width_m),
        division_mode=division_mode,
        total_length_m=total_length_m,
        module_lengths_x_m=module_lengths_x_m,
    )

    water_depth_m = _optional_float(hydro_data, "water_depth_m")
    g = _float(hydro_data, "g", 9.81)
    mode = str(omega_data.get("mode", "single"))
    if mode == "range":
        omegas = omega_values_from_range(
            _float(omega_data, "start_rad_s"),
            _float(omega_data, "stop_rad_s"),
            _int(omega_data, "count"),
        )
    elif mode == "list":
        values = omega_data.get("values_rad_s", "")
        omegas = parse_float_sequence(values if isinstance(values, str) else values or [])
    elif mode == "wavelength":
        values = omega_data.get("wavelength_values_m", "")
        omegas = omega_values_from_wavelengths(
            values if isinstance(values, str) else values or [],
            water_depth_m,
            g,
        )
    else:
        omegas = (_float(omega_data, "single_rad_s", 0.5851),)
    if not omegas:
        raise ValueError("at least one omega value is required")

    directions_deg = hydro_data.get("wave_directions_deg", [0.0])
    directions = parse_float_sequence(
        directions_deg if isinstance(directions_deg, str) else directions_deg or [0.0]
    )
    if not directions:
        directions = (0.0,)

    output_path = Path(str(payload.get("output_path") or DEFAULT_OUTPUT)).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    return ArrayHydrodynamicsConfig(
        module=module,
        layout=layout,
        omegas_rad_s=omegas,
        output_path=output_path,
        wave_directions_rad=degrees_to_radians(directions),
        water_depth_m=water_depth_m,
        rho=_float(hydro_data, "rho", 1025.0),
        g=g,
        n_jobs=_int(hydro_data, "n_jobs", 1),
        structural_grid=structural_grid,
    )


def run_job(job_id: str, payload: dict[str, object]) -> None:
    def log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS[job_id]
            job.setdefault("logs", []).append(message)

    try:
        config = config_from_payload(payload)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["output_path"] = str(config.output_path)
        log("Input deck parsed")
        log(f"Module division mode: {config.layout.division_mode}")
        log(
            "Module lengths m: "
            + _format_float_list(config.layout.module_lengths(config.module.length_m))
        )
        log(
            "Module boundaries m: "
            + _format_float_list(config.layout.x_boundaries(config.module.length_m))
        )
        centers = [item.x_m for item in config.layout.module_geometries(config.module.length_m)]
        log("Module centers x m: " + _format_float_list(centers))
        if config.structural_grid is not None:
            mappings = module_structural_node_mappings(config)
            log(
                "Structural FEM node ids: "
                + "[" + ", ".join(str(item["fem_node_one_based"]) for item in mappings) + "]"
            )
            log(
                "Structural FEM node x m: "
                + _format_float_list(item["x_m"] for item in mappings)
            )
        result = run_array_hydrodynamics(config, log=log)
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "completed",
                    "output_path": str(result.output_path),
                    "result": {
                        "body_count": result.body_count,
                        "dof_count": result.dof_count,
                        "problem_count": result.problem_count,
                        "omega_count": result.omega_count,
                        "wave_direction_count": result.wave_direction_count,
                        "water_depth_m": result.water_depth_m,
                        "rao_preview": result.rao_preview,
                    },
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id].setdefault("logs", []).append(str(exc))
            JOBS[job_id].setdefault("logs", []).append(traceback.format_exc())


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _float(data: dict[str, object], key: str, default: float | None = None) -> float:
    value = data.get(key, default)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{key} is required")
        return float(default)
    return float(value)


def _optional_float(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    return int(value)


def _bool(data: dict[str, object], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int(data: dict[str, object], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{key} is required")
        return int(default)
    return int(value)


def _module_lengths_from_layout(
    data: dict[str, object],
    *,
    division_mode: str,
    total_length_m: float,
    columns: int,
    structural_grid_dx_m: float | None = None,
) -> tuple[float, ...] | None:
    if division_mode == "uniform":
        return None
    values = data.get("module_lengths_x_m")
    if values not in (None, ""):
        lengths = parse_float_sequence(values if isinstance(values, str) else values or [])
    elif division_mode == "random":
        seed = _optional_int(data, "random_seed")
        if structural_grid_dx_m is not None:
            lengths = _random_grid_aligned_module_lengths(
                total_length_m,
                columns,
                structural_grid_dx_m,
                seed,
            )
        else:
            lengths = _random_module_lengths(total_length_m, columns, seed)
    else:
        raise ValueError("custom division requires module_lengths_x_m")
    _validate_module_lengths(
        lengths,
        total_length_m=total_length_m,
        columns=columns,
        structural_grid_dx_m=structural_grid_dx_m,
    )
    return lengths


def _validate_module_lengths(
    lengths: tuple[float, ...],
    *,
    total_length_m: float,
    columns: int,
    structural_grid_dx_m: float | None = None,
) -> None:
    if len(lengths) != columns:
        raise ValueError("number of module lengths must equal columns")
    for length in lengths:
        if length <= 0.0:
            raise ValueError("all module lengths must be positive")
    if abs(sum(lengths) - total_length_m) > 1.0e-6:
        raise ValueError("sum(module_lengths_x_m) must equal total_length_m")
    if structural_grid_dx_m is not None:
        center_unit = 2.0 * structural_grid_dx_m
        for length in lengths:
            ratio = length / center_unit
            if abs(ratio - round(ratio)) > 1.0e-6:
                raise ValueError(
                    "module lengths must be multiples of 2 * structural_grid_dx_m "
                    "so centers fall on FEM nodes"
                )


def _random_module_lengths(total_length_m: float, columns: int, seed: int | None) -> tuple[float, ...]:
    state = abs(int(seed or 1)) % 2147483647
    if state == 0:
        state = 1

    def next_random() -> float:
        nonlocal state
        state = (state * 48271) % 2147483647
        return state / 2147483647

    weights = [0.25 + next_random() for _ in range(columns)]
    weight_sum = sum(weights)
    lengths = [round(total_length_m * weight / weight_sum, 6) for weight in weights]
    lengths[-1] = round(lengths[-1] + total_length_m - sum(lengths), 6)
    return tuple(lengths)


def _random_grid_aligned_module_lengths(
    total_length_m: float,
    columns: int,
    structural_grid_dx_m: float,
    seed: int | None,
) -> tuple[float, ...]:
    center_unit = 2.0 * structural_grid_dx_m
    total_units = total_length_m / center_unit
    rounded_units = round(total_units)
    if abs(total_units - rounded_units) > 1.0e-9:
        raise ValueError("total_length_m must be divisible by 2 * structural_grid_dx_m")
    if rounded_units < columns:
        raise ValueError("columns is too large for the requested structural grid spacing")

    state = abs(int(seed or 1)) % 2147483647
    if state == 0:
        state = 1

    def next_random() -> float:
        nonlocal state
        state = (state * 48271) % 2147483647
        return state / 2147483647

    units = [rounded_units // columns for _ in range(columns)]
    remainder = rounded_units - sum(units)
    indices = list(range(columns))
    for index in range(columns - 1, 0, -1):
        swap_index = int(next_random() * (index + 1))
        indices[index], indices[swap_index] = indices[swap_index], indices[index]
    for index in indices[:remainder]:
        units[index] += 1

    min_unit = max(1, rounded_units // columns - 1)
    max_unit = math.ceil(rounded_units / columns) + 1
    transfer_count = min(int(columns * 0.4), columns - remainder)
    donor_order = [index for index in indices if units[index] > min_unit]
    receiver_order = [index for index in indices if units[index] < max_unit]
    transfers = 0
    for donor in donor_order:
        receiver = next(
            (index for index in receiver_order if index != donor and units[index] < max_unit),
            None,
        )
        if receiver is None:
            break
        units[donor] -= 1
        units[receiver] += 1
        transfers += 1
        if transfers >= transfer_count:
            break

    return tuple(float(unit * center_unit) for unit in units)


def _format_float_list(values) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def make_server(host: str, port: int) -> tuple[ThreadingHTTPServer, int]:
    for candidate in range(port, port + 50):
        try:
            return ThreadingHTTPServer((host, candidate), HydroUIHandler), candidate
        except OSError:
            continue
    raise OSError(f"could not bind any port from {port} to {port + 49}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server, port = make_server(args.host, args.port)
    url_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    print(f"RODM hydrodynamics UI: http://{url_host}:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping hydrodynamics UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
