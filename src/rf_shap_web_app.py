import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))

import joblib
import numpy as np
import pandas as pd
import shap


SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_rf_shap_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, add_theory_features, normalize_category  # noqa: E402
from predict_rf_shap_agent import korsmeyer_peppas_for_curve, nearest_rows, training_range_warnings  # noqa: E402


DEFAULT_TIMES = [0, 1, 2, 4, 8, 12, 24, 48, 72]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nanocarrier Release Agent</title>
  <style>
    :root {
      --bg: #f4f6f7;
      --ink: #182025;
      --muted: #66727b;
      --line: #d7dee3;
      --panel: #ffffff;
      --nav: #1d252b;
      --accent: #007c72;
      --accent2: #b65c2e;
      --accent3: #4267a8;
      --good: #e4f3ee;
      --warn: #fff4dc;
      --shadow: 0 12px 34px rgba(18, 32, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select { font: inherit; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
    }
    aside {
      background: var(--nav);
      color: #fff;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .brand {
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; line-height: 1.15; }
    .sub { margin: 7px 0 0; color: #aab6bd; font-size: 13px; line-height: 1.4; }
    .section-label {
      margin-top: 8px;
      color: #d8e1e6;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .field { display: grid; gap: 6px; }
    label { font-size: 12px; color: #cbd5db; font-weight: 700; }
    input, select {
      width: 100%;
      border: 1px solid #43505a;
      background: #263139;
      color: #fff;
      border-radius: 7px;
      padding: 10px 11px;
      outline: none;
    }
    input:focus, select:focus {
      border-color: #76d4c5;
      box-shadow: 0 0 0 3px rgba(118, 212, 197, 0.15);
    }
    textarea {
      width: 100%;
      min-height: 86px;
      resize: vertical;
      border: 1px solid var(--line);
      background: #fbfcfc;
      color: var(--ink);
      border-radius: 7px;
      padding: 10px 11px;
      outline: none;
      font: inherit;
      line-height: 1.4;
    }
    textarea:focus {
      border-color: #76d4c5;
      box-shadow: 0 0 0 3px rgba(118, 212, 197, 0.15);
    }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .run {
      margin-top: 8px;
      border: 0;
      border-radius: 7px;
      padding: 13px 16px;
      background: #76d4c5;
      color: #0e2422;
      font-weight: 850;
      cursor: pointer;
    }
    .run:disabled { opacity: 0.65; cursor: wait; }
    .button-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .secondary {
      border: 1px solid #43505a;
      border-radius: 7px;
      padding: 10px 11px;
      background: #202a31;
      color: #e9f0f3;
      font-weight: 760;
      cursor: pointer;
    }
    .secondary:disabled { opacity: 0.45; cursor: not-allowed; }
    main {
      min-width: 0;
      padding: 24px;
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }
    h2 { margin: 0; font-size: 28px; letter-spacing: 0; }
    .meta { margin-top: 5px; color: var(--muted); font-size: 13px; }
    .status { color: var(--muted); font-size: 13px; white-space: nowrap; }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric { padding: 14px; min-height: 88px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; font-weight: 750; }
    .metric strong { display: block; margin-top: 8px; font-size: clamp(18px, 1.35vw, 26px); letter-spacing: 0; line-height: 1.15; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(330px, 0.9fr); gap: 16px; }
    .panel { padding: 17px; min-width: 0; }
    .panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
    h3 { margin: 0; font-size: 16px; letter-spacing: 0; }
    .legend { display: flex; flex-wrap: wrap; gap: 9px; color: var(--muted); font-size: 12px; }
    .legend span { display: inline-flex; align-items: center; gap: 5px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; background: var(--accent); }
    .swatch.alt { background: var(--accent2); }
    .swatch.band { background: #d99b3d; }
    .chart-empty {
      min-height: 360px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--muted);
    }
    svg { display: block; width: 100%; height: auto; }
    .grid-line { stroke: var(--line); stroke-width: 1; }
    .axis-text { fill: var(--muted); font-size: 11px; }
    .driver { display: grid; gap: 8px; }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(120px, 1fr) 58px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .bar-bg { height: 9px; border-radius: 99px; background: #edf1f3; overflow: hidden; }
    .bar { height: 100%; border-radius: 99px; background: var(--accent); }
    .bar.neg { background: var(--accent2); }
    .analysis { display: grid; gap: 10px; }
    .note {
      border-left: 3px solid var(--accent);
      background: #f7fbfa;
      padding: 9px 10px;
      color: #344047;
      font-size: 13px;
      line-height: 1.45;
    }
    .interpretation-intro {
      color: var(--muted);
      line-height: 1.45;
      margin: 0;
      font-size: 13px;
    }
    .mini-title {
      color: #4f5d66;
      font-weight: 800;
      font-size: 12px;
      letter-spacing: 0.04em;
      margin-top: 4px;
      text-transform: uppercase;
    }
    .theory-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      color: #34444d;
      background: #fbfcfc;
      font-size: 13px;
    }
    .theory-box summary {
      cursor: pointer;
      font-weight: 800;
      color: #1f2b32;
    }
    .theory-list {
      margin: 10px 0 0;
      padding-left: 18px;
      line-height: 1.5;
    }
    .warning {
      margin-top: 10px;
      background: var(--warn);
      border: 1px solid #efd6a2;
      color: #765013;
      padding: 9px 10px;
      border-radius: 7px;
      font-size: 12px;
      line-height: 1.45;
    }
    .compare-list {
      margin-top: 12px;
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .compare-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfc;
    }
    .compare-item span { min-width: 0; overflow-wrap: anywhere; }
    .chip { display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent3); flex: 0 0 auto; }
    .error {
      margin-top: 8px;
      color: #ffd6cb;
      background: rgba(182, 92, 46, 0.18);
      border: 1px solid rgba(255, 214, 203, 0.22);
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.4;
      display: none;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      gap: 9px;
      color: #d8e1e6;
      font-size: 12px;
      font-weight: 750;
    }
    .toggle-row input {
      width: 16px;
      height: 16px;
      accent-color: #76d4c5;
    }
    .hint {
      color: #aab6bd;
      font-size: 12px;
      line-height: 1.4;
    }
    #neighbors { max-width: 100%; }
    .evidence-list { display: grid; gap: 10px; }
    .evidence-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfc;
      padding: 10px;
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .evidence-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      min-width: 0;
    }
    .evidence-id {
      min-width: 0;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.35;
      color: #24313a;
    }
    .evidence-id strong {
      display: block;
      color: var(--ink);
      font-size: 12.5px;
      margin-bottom: 2px;
    }
    .evidence-score {
      text-align: right;
      white-space: nowrap;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }
    .evidence-score strong {
      display: block;
      color: var(--ink);
      font-size: 16px;
    }
    .evidence-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-width: 0;
    }
    .meta-pill {
      display: inline-flex;
      max-width: 100%;
      border-radius: 999px;
      background: #edf3f3;
      color: #40505a;
      padding: 4px 8px;
      font-size: 11px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    @media (max-width: 1040px) {
      .app { grid-template-columns: 1fr; }
      .workspace, .metrics { grid-template-columns: 1fr; }
      aside { min-height: auto; }
      .top { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <h1>Nanocarrier Release Agent</h1>
        <p class="sub">Random Forest + SHAP for PLGA, liposome, and chitosan release screening.</p>
      </div>

      <div class="section-label">Formulation</div>
      <div class="field">
        <label for="carrier">Carrier type</label>
        <select id="carrier">
          <option>liposome</option>
          <option>PLGA nanoparticle</option>
          <option>chitosan nanoparticle</option>
        </select>
      </div>
      <div class="field">
        <label for="drug">Drug</label>
        <select id="drug">
          <option>Doxorubicin</option>
          <option>Paclitaxel</option>
          <option>Insulin</option>
          <option>Rifampicin</option>
          <option>carvacrol</option>
          <option>Gemcitabine</option>
        </select>
      </div>
      <div class="field">
        <label for="medium">Release medium</label>
        <select id="medium">
          <option>PBS</option>
          <option>not_reported</option>
        </select>
      </div>
      <label class="toggle-row">
        <input id="freeRef" type="checkbox" />
        <span>Free drug reference</span>
      </label>
      <div id="freeRefHint" class="hint"></div>

      <div class="section-label">Numeric inputs</div>
      <div class="grid-2">
        <div class="field">
          <label for="size">Size nm</label>
          <input id="size" type="number" value="127.92" step="0.01" min="1" max="5000" />
        </div>
        <div class="field">
          <label for="zeta">Zeta mV</label>
          <input id="zeta" type="number" value="-23.33" step="0.01" min="-250" max="250" />
        </div>
      </div>
      <div class="grid-2">
        <div class="field">
          <label for="pdi">PDI</label>
          <input id="pdi" type="number" value="0.2" step="0.001" min="0" max="20" />
        </div>
        <div class="field">
          <label for="ph">pH</label>
          <input id="ph" type="number" value="5.5" step="0.1" min="0" max="14" />
        </div>
      </div>
      <div class="grid-2">
        <div class="field">
          <label for="temp">Temperature C</label>
          <input id="temp" type="number" value="37" step="0.1" min="0" max="100" />
        </div>
        <div class="field">
          <label for="time">Focus time h</label>
          <input id="time" type="number" value="24" step="1" min="0" max="10000" />
        </div>
      </div>
      <div class="grid-2">
        <div class="field">
          <label for="loading">Loading %</label>
          <input id="loading" type="number" value="8" step="0.01" min="0" max="100" />
        </div>
        <div class="field">
          <label for="ee">EE %</label>
          <input id="ee" type="number" value="70" step="0.01" min="0" max="100" />
        </div>
      </div>

      <div id="inputError" class="error"></div>
      <button id="run" class="run" type="button">Predict Release</button>
      <div class="button-grid">
        <button id="saveScenario" class="secondary" type="button">Save Compare</button>
        <button id="clearScenarios" class="secondary" type="button">Clear</button>
        <button id="downloadCsv" class="secondary" type="button">CSV</button>
        <button id="downloadPng" class="secondary" type="button">PNG</button>
      </div>
    </aside>

    <main>
      <div class="top">
        <div>
          <h2>Prediction Workspace</h2>
          <div class="meta" id="modelMeta">Loading model metadata</div>
        </div>
        <div class="status" id="status">Ready</div>
      </div>

      <section class="metrics">
        <div class="metric"><span>Focus Prediction</span><strong id="pred">-</strong></div>
        <div class="metric"><span>RF 10-90% Range</span><strong id="interval">-</strong></div>
        <div class="metric"><span>SHAP Baseline</span><strong id="base">-</strong></div>
        <div class="metric"><span>Reliability</span><strong id="rel">-</strong></div>
        <div class="metric"><span>KP n Estimate</span><strong id="kp">-</strong></div>
      </section>

      <section class="workspace">
        <div class="panel">
          <div class="panel-head">
            <h3>Predicted Release Curve</h3>
            <div class="legend">
              <span><i class="swatch"></i>prediction</span>
              <span><i class="swatch" style="background:#88aee8"></i>RF 10-90% range</span>
              <span><i class="swatch alt"></i>free reference</span>
              <span><i class="swatch" style="background:#4267a8"></i>saved comparisons</span>
              <span><i class="swatch band"></i>nearest evidence band</span>
            </div>
          </div>
          <div id="chart" class="chart-empty">Run the model to draw a curve</div>
          <div id="freeSummary" class="compare-list"></div>
          <div id="compareList" class="compare-list"></div>
        </div>

        <div class="panel">
          <div class="panel-head"><h3>Local SHAP Drivers</h3></div>
          <div class="driver" id="drivers"></div>
        </div>
      </section>

      <section class="workspace">
        <div class="panel">
          <div class="panel-head"><h3>Model Interpretation</h3></div>
          <div class="analysis" id="analysis"></div>
          <div style="margin-top: 14px;">
            <label for="question" style="display:block;color:#66727b;margin-bottom:6px;">Ask about this prediction</label>
            <textarea id="question" placeholder="Example: What should I change to increase release percentage?"></textarea>
            <button id="ask" class="run" style="margin-top:10px;width:100%;" type="button">Ask About Prediction</button>
            <div class="analysis" id="answer" style="margin-top:10px;"></div>
          </div>
          <div id="warnings"></div>
        </div>
        <div class="panel">
          <div class="panel-head"><h3>Nearest Training Evidence</h3></div>
          <div id="neighbors"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const curveTimes = [0, 1, 2, 4, 8, 12, 24, 48, 72];
    const scenarioColors = ["#4267a8", "#7b5aa6", "#a45f38"];
    const state = {
      focus: null,
      carrierPoints: [],
      freePoints: [],
      freeCurve: null,
      scenarios: []
    };

    function num(id) {
      const value = $(id).value.trim();
      return value === "" ? null : Number(value);
    }

    const limits = {
      size: [1, 5000, "Size must be between 1 and 5000 nm."],
      zeta: [-250, 250, "Zeta potential must be between -250 and 250 mV."],
      pdi: [0, 20, "PDI cannot be negative. This app allows 0-20 because some papers report PDI-like values as percent."],
      ph: [0, 14, "pH must be between 0 and 14."],
      temp: [0, 100, "Temperature must be between 0 and 100 C."],
      time: [0, 10000, "Time cannot be negative or above 10000 h."],
      loading: [0, 100, "Drug loading percent must be 0-100."],
      ee: [0, 100, "Encapsulation efficiency percent must be 0-100."]
    };

    function validateInputs() {
      const messages = [];
      for (const [id, [min, max, message]] of Object.entries(limits)) {
        const value = num(id);
        if (value === null) continue;
        if (!Number.isFinite(value) || value < min || value > max) messages.push(message);
      }
      $("inputError").style.display = messages.length ? "block" : "none";
      $("inputError").innerHTML = messages.join("<br>");
      return messages.length === 0;
    }

    function requestBody(timeOverride = null) {
      return {
        carrier_type: $("carrier").value,
        drug_name: $("drug").value,
        release_medium: $("medium").value,
        particle_size_nm: num("size"),
        zeta_potential_mv: num("zeta"),
        pdi: num("pdi"),
        ph: num("ph"),
        temperature_C: num("temp"),
        time_h: timeOverride === null ? num("time") : timeOverride,
        drug_loading_content_percent: num("loading"),
        encapsulation_efficiency_percent: num("ee")
      };
    }

    function curveRequestTimes() {
      const focusTime = num("time");
      const earlyTimes = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12];
      const times = focusTime !== null && Number.isFinite(focusTime) && focusTime <= 6
        ? earlyTimes.slice()
        : curveTimes.slice();
      if (focusTime !== null && Number.isFinite(focusTime) && !times.includes(focusTime)) times.push(focusTime);
      return times.sort((a, b) => a - b);
    }

    function fillSelect(id, values, preferred, labelFn = (value) => value) {
      const current = preferred || $(id).value;
      const clean = Array.from(new Set(values.filter((value) => value && value !== "not_reported"))).sort();
      if (!clean.includes(current)) clean.unshift(current);
      $(id).innerHTML = "";
      clean.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = labelFn(value);
        $(id).appendChild(option);
      });
      $(id).value = current;
    }

    function displayDrug(value) {
      if (!value) return value;
      if (value === value.toUpperCase()) return value;
      return value.charAt(0).toUpperCase() + value.slice(1);
    }

    async function postJson(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function pct(value) {
      return `${Number(value).toFixed(1)}%`;
    }

    function intervalText(interval) {
      if (!interval) return "-";
      return `${pct(interval.p10)}-${pct(interval.p90)}`;
    }

    function scale(value, inMin, inMax, outMin, outMax) {
      if (inMax === inMin) return (outMin + outMax) / 2;
      return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
    }

    function drawChart(points, freePoints = [], comparisons = []) {
      const w = 880, h = 390;
      const pad = { left: 54, right: 18, top: 18, bottom: 40 };
      const allComparisonPoints = comparisons.flatMap((item) => item.points);
      const allPoints = [...points, ...freePoints, ...allComparisonPoints];
      const focusTime = num("time");
      const autoZoom = focusTime !== null && Number.isFinite(focusTime) && focusTime <= 6;
      const maxX = autoZoom ? 12 : Math.max(...allPoints.map((p) => p.time_h));
      const visiblePoints = points.filter((p) => p.time_h <= maxX);
      const visibleFreePoints = freePoints.filter((p) => p.time_h <= maxX);
      const visibleComparisons = comparisons.map((series) => ({
        ...series,
        points: series.points.filter((p) => p.time_h <= maxX)
      }));
      const freePredictions = freePoints.map((p) => p.prediction);
      const visibleComparisonPoints = visibleComparisons.flatMap((item) => item.points);
      const comparisonPredictions = visibleComparisonPoints.map((p) => p.prediction);
      const intervalValues = visiblePoints.flatMap((p) => p.interval ? [p.interval.p10, p.interval.p90] : []);
      const evidenceValues = visiblePoints.flatMap((p) => p.evidence_band_available ? [p.neighbor_min, p.neighbor_max] : []);
      const maxY = Math.min(100, Math.max(30, ...visiblePoints.map((p) => p.prediction), ...evidenceValues, ...visibleFreePoints.map((p) => p.prediction), ...comparisonPredictions, ...intervalValues) + 8);
      const x = (value) => scale(value, 0, maxX, pad.left, w - pad.right);
      const y = (value) => scale(value, 0, maxY, h - pad.bottom, pad.top);
      const path = visiblePoints.map((p, i) => `${i ? "L" : "M"} ${x(p.time_h).toFixed(1)} ${y(p.prediction).toFixed(1)}`).join(" ");
      const freePath = visibleFreePoints.map((p, i) => `${i ? "L" : "M"} ${x(p.time_h).toFixed(1)} ${y(p.prediction).toFixed(1)}`).join(" ");
      const intervalUpper = visiblePoints.filter((p) => p.interval).map((p) => `${x(p.time_h).toFixed(1)} ${y(p.interval.p90).toFixed(1)}`);
      const intervalLower = visiblePoints.filter((p) => p.interval).slice().reverse().map((p) => `${x(p.time_h).toFixed(1)} ${y(p.interval.p10).toFixed(1)}`);
      const intervalBand = intervalUpper.length ? `M ${intervalUpper.join(" L ")} L ${intervalLower.join(" L ")} Z` : "";
      const comparisonPaths = visibleComparisons.map((series) => ({
        color: series.color,
        path: series.points.map((p, i) => `${i ? "L" : "M"} ${x(p.time_h).toFixed(1)} ${y(p.prediction).toFixed(1)}`).join(" ")
      }));
      const evidencePoints = visiblePoints.filter((p) => p.evidence_band_available);
      const upper = evidencePoints.map((p) => `${x(p.time_h).toFixed(1)} ${y(p.neighbor_max).toFixed(1)}`);
      const lower = evidencePoints.slice().reverse().map((p) => `${x(p.time_h).toFixed(1)} ${y(p.neighbor_min).toFixed(1)}`);
      const band = upper.length ? `M ${upper.join(" L ")} L ${lower.join(" L ")} Z` : "";
      const yTicks = [0, 20, 40, 60, 80, 100].filter((v) => v <= maxY);
      const xTicks = (autoZoom ? [0, 1, 2, 4, 6, 8, 12] : [0, 12, 24, 48, 72]).filter((v) => v <= maxX);
      const grid = [
        ...yTicks.map((tick) => `<line class="grid-line" x1="${pad.left}" y1="${y(tick)}" x2="${w - pad.right}" y2="${y(tick)}"></line><text class="axis-text" x="14" y="${y(tick) + 4}">${tick}%</text>`),
        ...xTicks.map((tick) => `<line class="grid-line" x1="${x(tick)}" y1="${pad.top}" x2="${x(tick)}" y2="${h - pad.bottom}"></line><text class="axis-text" x="${x(tick) - 10}" y="${h - 13}">${tick}</text>`)
      ].join("");
      $("chart").className = "";
      $("chart").innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted release curve">
          ${grid}
          ${band ? `<path d="${band}" fill="#d99b3d" opacity="0.22"></path>` : ""}
          ${intervalBand ? `<path d="${intervalBand}" fill="#88aee8" opacity="0.35"></path>` : ""}
          <path d="${path}" fill="none" stroke="#007c72" stroke-width="4"></path>
          ${freePath ? `<path d="${freePath}" fill="none" stroke="#b65c2e" stroke-width="3.5" stroke-dasharray="8 7"></path>` : ""}
          ${comparisonPaths.map((series) => `<path d="${series.path}" fill="none" stroke="${series.color}" stroke-width="3" opacity="0.86"></path>`).join("")}
          ${visiblePoints.map((p) => `<circle cx="${x(p.time_h)}" cy="${y(p.prediction)}" r="4.5" fill="#007c72"></circle>`).join("")}
          ${visibleFreePoints.map((p) => `<circle cx="${x(p.time_h)}" cy="${y(p.prediction)}" r="4" fill="#b65c2e"></circle>`).join("")}
          <text class="axis-text" x="${w - 70}" y="${h - 13}">time h</text>
        </svg>
      `;
    }

    function renderDrivers(items) {
      const maxAbs = Math.max(...items.map((item) => Math.abs(item.shap_value)), 1);
      $("drivers").innerHTML = items.map((item) => {
        const width = Math.max(4, Math.abs(item.shap_value) / maxAbs * 100);
        const cls = item.shap_value < 0 ? "bar neg" : "bar";
        return `
          <div class="bar-row">
            <div title="${item.feature}">${item.feature}</div>
            <div class="bar-bg"><div class="${cls}" style="width:${width}%"></div></div>
            <div>${item.shap_value >= 0 ? "+" : ""}${item.shap_value.toFixed(2)}</div>
          </div>
        `;
      }).join("");
    }

    function renderAnalysis(sections, warnings) {
      const modelNotes = sections.modelNotes || [];
      const mechanismHints = sections.mechanismHints || [];
      const noteHtml = modelNotes.length
        ? `<div class="mini-title">Model notes</div>${modelNotes.map((line) => `<div class="note">${line}</div>`).join("")}`
        : "";
      const hintHtml = mechanismHints.length
        ? `<div class="mini-title">Mechanism hints</div>${mechanismHints.map((line) => `<div class="note">${line}</div>`).join("")}`
        : "";
      $("analysis").innerHTML = `
        <p class="interpretation-intro">This panel summarizes RF/SHAP drivers and release-theory hints for the current prediction. It supports interpretation, but does not replace experimental validation.</p>
        ${noteHtml}
        ${hintHtml}
        <details class="theory-box">
          <summary>Theory reference</summary>
          <ul class="theory-list">
            <li><strong>Noyes-Whitney:</strong> diffusion rate depends on exposed area and concentration driving force.</li>
            <li><strong>Higuchi:</strong> cumulative release can scale with square-root time in diffusion-controlled systems.</li>
            <li><strong>Korsmeyer-Peppas:</strong> the exponent n gives a hint about Fickian diffusion or anomalous transport.</li>
            <li><strong>SHAP local drivers:</strong> positive values raise the prediction; negative values lower it for this input.</li>
          </ul>
        </details>
      `;
      $("warnings").innerHTML = warnings.length
        ? `<div class="warning">${warnings.map((line) => `<div>${line}</div>`).join("")}</div>`
        : "";
    }

    function renderFreeSummary() {
      if (!state.freePoints.length || !state.carrierPoints.length) {
        $("freeSummary").innerHTML = "";
        return;
      }
      const focusTime = num("time");
      const carrier = nearestPoint(state.carrierPoints, focusTime);
      const free = nearestPoint(state.freePoints, focusTime);
      const delta = carrier.prediction - free.prediction;
      const direction = delta >= 0 ? "higher than" : "lower than";
      $("freeSummary").innerHTML = `
        <div class="compare-item">
          <span>At ${carrier.time_h} h, carrier prediction is ${Math.abs(delta).toFixed(1)} percentage points ${direction} free reference.</span>
        </div>
      `;
    }

    function nearestPoint(points, time) {
      return points.reduce((best, item) => Math.abs(item.time_h - time) < Math.abs(best.time_h - time) ? item : best, points[0]);
    }

    function scenarioLabel() {
      return `${$("carrier").value} | ${$("drug").selectedOptions[0].textContent} | pH ${$("ph").value} | ${$("medium").value}`;
    }

    function renderCompareList() {
      if (!state.scenarios.length) {
        $("compareList").innerHTML = "";
        return;
      }
      const summary = comparisonSummary();
      const items = state.scenarios.map((item, index) => `
        <div class="compare-item">
          <span class="chip"><i class="dot" style="background:${item.color}"></i>${item.label}</span>
          <span>${pct(nearestPoint(item.points, num("time")).prediction)}</span>
        </div>
      `).join("");
      $("compareList").innerHTML = `
        <div class="compare-item"><span>${summary}</span></div>
        ${items}
      `;
    }

    function comparisonSummary() {
      const focusTime = num("time");
      const current = nearestPoint(state.carrierPoints, focusTime);
      const series = [
        { label: "Current", value: current.prediction },
        ...state.scenarios.map((item) => ({
          label: item.label,
          value: nearestPoint(item.points, focusTime).prediction
        }))
      ];
      if (state.scenarios.length === 1) {
        const saved = series[1];
        const delta = saved.value - current.prediction;
        const direction = delta >= 0 ? "higher than" : "lower than";
        return `At ${current.time_h} h, saved curve is ${Math.abs(delta).toFixed(1)} percentage points ${direction} the current curve.`;
      }
      const ranked = series.slice().sort((a, b) => b.value - a.value);
      const high = ranked[0];
      const low = ranked[ranked.length - 1];
      const spread = high.value - low.value;
      return `At ${current.time_h} h, the highest curve is ${high.label} (${pct(high.value)}) and the lowest is ${low.label} (${pct(low.value)}), a ${spread.toFixed(1)} percentage-point spread.`;
    }

    function refreshChart() {
      if (!state.carrierPoints.length) return;
      drawChart(state.carrierPoints, state.freePoints, state.scenarios);
      renderFreeSummary();
      renderCompareList();
    }

    function renderNeighbors(rows) {
      $("neighbors").innerHTML = `
        <div class="evidence-list">
          ${rows.map((row) => `
            <div class="evidence-card">
              <div class="evidence-top">
                <div class="evidence-id">
                  <strong>${row.curve_id}</strong>
                  ${row.paper_id}
                </div>
                <div class="evidence-score">
                  <strong>${pct(row.drug_release_percent)}</strong>
                  distance ${row.distance.toFixed(3)}
                </div>
              </div>
              <div class="evidence-meta">
                <span class="meta-pill">${row.carrier_type}</span>
                <span class="meta-pill">${row.drug_name}</span>
                <span class="meta-pill">pH ${row.ph}</span>
                <span class="meta-pill">${row.time_h} h</span>
              </div>
            </div>
          `).join("")}
        </div>
      `;
    }

    async function run() {
      $("run").disabled = true;
      $("status").textContent = "Running RF/SHAP";
      try {
        if (!validateInputs()) {
          $("status").textContent = "Check inputs";
          return;
        }
        const times = curveRequestTimes();
        const focus = await postJson("/api/predict", requestBody());
        const curve = await postJson("/api/curve", { ...requestBody(), times });
        const freeCurve = $("freeRef").checked
          ? await postJson("/api/free-curve", { ...requestBody(), times })
          : { available: false, points: [] };
        $("pred").textContent = pct(focus.prediction);
        $("interval").textContent = intervalText(focus.prediction_interval);
        $("base").textContent = pct(focus.expected_value);
        $("rel").textContent = focus.reliability;
        $("kp").textContent = focus.korsmeyer_peppas ? focus.korsmeyer_peppas.n.toFixed(2) : "-";
        const carrierPoints = curve.points.map((item) => ({
          time_h: item.input.time_h,
          prediction: item.prediction,
          interval: item.prediction_interval,
          neighbor_min: item.neighbor_release_min,
          neighbor_max: item.neighbor_release_max,
          evidence_band_available: item.evidence_band_available
        }));
        const freePoints = freeCurve.available ? freeCurve.points.map((item) => ({
          time_h: item.input.time_h,
          prediction: item.prediction
        })) : [];
        state.focus = focus;
        state.carrierPoints = carrierPoints;
        state.freePoints = freePoints;
        state.freeCurve = freeCurve;
        refreshChart();
        renderDrivers(focus.shap_top);
        const reliabilityNote = focus.reliability_reason ? [focus.reliability_reason] : [];
        const evidenceNote = curve.message ? [curve.message] : [];
        const freeNote = $("freeRef").checked && freeCurve.message ? [freeCurve.message] : [];
        renderAnalysis(
          {
            modelNotes: [...reliabilityNote, ...evidenceNote, ...freeNote],
            mechanismHints: focus.interpretation
          },
          focus.warnings
        );
        renderNeighbors(focus.nearest_rows);
        $("status").textContent = "Updated";
      } catch (error) {
        $("status").textContent = "Error";
        $("chart").className = "chart-empty";
        $("chart").textContent = error.message;
      } finally {
        $("run").disabled = false;
      }
    }

    function saveScenario() {
      if (!state.carrierPoints.length) return;
      const color = scenarioColors[state.scenarios.length % scenarioColors.length];
      state.scenarios.push({
        label: scenarioLabel(),
        color,
        input: requestBody(),
        points: state.carrierPoints.map((point) => ({ ...point }))
      });
      if (state.scenarios.length > 3) state.scenarios.shift();
      refreshChart();
    }

    function clearScenarios() {
      state.scenarios = [];
      refreshChart();
    }

    function downloadBlob(filename, content, type) {
      const blob = new Blob([content], { type });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function downloadCsv() {
      if (!state.carrierPoints.length) return;
      const lines = ["series,time_h,prediction_percent,neighbor_min,neighbor_max"];
      lines[0] = "series,time_h,prediction_percent,rf_p10,rf_p90,neighbor_min,neighbor_max";
      state.carrierPoints.forEach((p) => lines.push(`current,${p.time_h},${p.prediction},${p.interval?.p10 ?? ""},${p.interval?.p90 ?? ""},${p.neighbor_min ?? ""},${p.neighbor_max ?? ""}`));
      state.freePoints.forEach((p) => lines.push(`free_reference,${p.time_h},${p.prediction},,,,`));
      state.scenarios.forEach((series) => {
        series.points.forEach((p) => lines.push(`"${series.label.replaceAll('"', '""')}",${p.time_h},${p.prediction},${p.interval?.p10 ?? ""},${p.interval?.p90 ?? ""},,`));
      });
      downloadBlob("nanocarrier_release_prediction.csv", lines.join("\n"), "text/csv;charset=utf-8");
    }

    function downloadPng() {
      const svg = $("chart").querySelector("svg");
      if (!svg) return;
      const xml = new XMLSerializer().serializeToString(svg);
      const svgUrl = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
      const image = new Image();
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 1320;
        canvas.height = 585;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(svgUrl);
        canvas.toBlob((blob) => {
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = "nanocarrier_release_curve.png";
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
        });
      };
      image.src = svgUrl;
    }

    async function askModel() {
      if (!validateInputs()) return;
      const question = $("question").value.trim();
      if (!question) {
        $("answer").innerHTML = `<div class="note">Type a question about the current condition first.</div>`;
        return;
      }
      $("ask").disabled = true;
      try {
        const response = await postJson("/api/ask", { ...requestBody(), question });
        $("answer").innerHTML = response.answer.map((line) => `<div class="note">${line}</div>`).join("");
      } catch (error) {
        $("answer").innerHTML = `<div class="warning">${error.message}</div>`;
      } finally {
        $("ask").disabled = false;
      }
    }

    async function boot() {
      const meta = await fetch("/api/metadata").then((res) => res.json());
      fillSelect("carrier", meta.carrier_types || [], "liposome");
      fillSelect("drug", meta.drug_names || [], "Doxorubicin", displayDrug);
      fillSelect("medium", meta.release_media || [], "PBS");
      window.freeDrugNames = meta.free_drug_names || [];
      updateFreeReferenceState();
      $("modelMeta").textContent = `${meta.rows} points, ${meta.curves} curves, ${meta.papers} papers | CV R2 ${meta.group_cv_r2.toFixed(3)} | RMSE ${meta.group_cv_rmse.toFixed(2)}`;
      run();
    }

    function updateFreeReferenceState() {
      const available = (window.freeDrugNames || []).includes($("drug").value);
      $("freeRef").disabled = !available;
      if (!available) $("freeRef").checked = false;
      $("freeRefHint").textContent = available ? "Reference available for this drug." : "No free reference data for this drug.";
    }

    $("run").addEventListener("click", run);
    $("ask").addEventListener("click", askModel);
    $("saveScenario").addEventListener("click", saveScenario);
    $("clearScenarios").addEventListener("click", clearScenarios);
    $("downloadCsv").addEventListener("click", downloadCsv);
    $("downloadPng").addEventListener("click", downloadPng);
    $("drug").addEventListener("change", updateFreeReferenceState);
    Object.keys(limits).forEach((id) => $(id).addEventListener("input", validateInputs));
    boot();
  </script>
</body>
</html>
"""


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


INPUT_LIMITS = {
    "particle_size_nm": (1.0, 5000.0, "particle size must be between 1 and 5000 nm"),
    "zeta_potential_mv": (-250.0, 250.0, "zeta potential must be between -250 and 250 mV"),
    "pdi": (0.0, 20.0, "PDI cannot be negative and must be 20 or lower"),
    "ph": (0.0, 14.0, "pH must be between 0 and 14"),
    "temperature_C": (0.0, 100.0, "temperature must be between 0 and 100 C"),
    "time_h": (0.0, 10000.0, "time must be between 0 and 10000 h"),
    "drug_loading_content_percent": (0.0, 100.0, "drug loading percent must be 0-100"),
    "encapsulation_efficiency_percent": (0.0, 100.0, "encapsulation efficiency percent must be 0-100"),
}


def validate_payload(row: dict) -> list[str]:
    errors = []
    for key, (minimum, maximum, message) in INPUT_LIMITS.items():
        value = row.get(key)
        if value is None:
            continue
        if not math.isfinite(float(value)) or float(value) < minimum or float(value) > maximum:
            errors.append(message)
    return errors


def query_frame(payload: dict) -> pd.DataFrame:
    release_medium = payload.get("release_medium")
    if release_medium is None:
        release_medium = payload.get("release_media")
    row = {
        "carrier_type": str(payload.get("carrier_type") or "unknown"),
        "drug_name": str(payload.get("drug_name") or "unknown"),
        "release_medium": str(release_medium or "unknown"),
        "particle_size_nm": to_float(payload.get("particle_size_nm")),
        "zeta_potential_mv": to_float(payload.get("zeta_potential_mv")),
        "pdi": to_float(payload.get("pdi")),
        "ph": to_float(payload.get("ph"), 7.4),
        "temperature_C": to_float(payload.get("temperature_C")),
        "time_h": max(0.0, to_float(payload.get("time_h"), 0.0)),
        "drug_loading_content_percent": to_float(payload.get("drug_loading_content_percent")),
        "encapsulation_efficiency_percent": to_float(payload.get("encapsulation_efficiency_percent")),
    }
    if payload.get("carrier_family"):
        row["carrier_family"] = str(payload.get("carrier_family"))
    errors = validate_payload(row)
    if errors:
        raise ValueError("; ".join(errors))
    return add_theory_features(pd.DataFrame([row]))


def sorted_values(frame: pd.DataFrame, column: str, limit: int = 80) -> list[str]:
    if column not in frame.columns:
        return []
    values = []
    for value in frame[column].dropna().astype(str).unique().tolist():
        text = value.strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            values.append(text)
    return sorted(values, key=lambda text: text.lower())[:limit]


def public_carrier_types(training: pd.DataFrame) -> list[str]:
    preferred = ["PLGA nanoparticle", "liposome", "chitosan nanoparticle"]
    available = set(sorted_values(training, "carrier_type"))
    return [carrier for carrier in preferred if carrier in available]


def public_release_media(training: pd.DataFrame) -> list[str]:
    hidden = {"unknown buffer", "not_reported"}
    return [medium for medium in sorted_values(training, "release_medium") if medium not in hidden]


def free_reference_frame(training: pd.DataFrame, drug_name: str) -> pd.DataFrame:
    canonical = normalize_category("drug_name", drug_name)
    is_free = training["carrier_type"].astype(str).str.contains("free", case=False, na=False)
    return training[is_free & training["drug_name"].eq(canonical)].copy()


def free_reference_drugs(training: pd.DataFrame) -> list[str]:
    is_free = training["carrier_type"].astype(str).str.contains("free", case=False, na=False)
    values = training.loc[is_free, "drug_name"].dropna().astype(str).unique().tolist()
    return sorted(values, key=lambda text: text.lower())


def mode_or_default(series: pd.Series, default: str) -> str:
    clean = series.dropna().astype(str)
    clean = clean[clean.str.len() > 0]
    if clean.empty:
        return default
    return str(clean.mode().iloc[0])


def free_reference_payload(training: pd.DataFrame, payload: dict) -> tuple[dict | None, pd.DataFrame, str]:
    free_rows = free_reference_frame(training, str(payload.get("drug_name") or ""))
    if free_rows.empty:
        return None, free_rows, "No free drug reference curve is available for this drug."

    medium = str(payload.get("release_medium") or payload.get("release_media") or "")
    available_media = set(free_rows["release_medium"].dropna().astype(str))
    if medium not in available_media:
        medium = mode_or_default(free_rows["release_medium"], "not_reported")

    carrier_type = mode_or_default(free_rows["carrier_type"], "free drug")
    carrier_family = mode_or_default(free_rows["carrier_family"], "other")
    temperature = to_float(payload.get("temperature_C"))
    if temperature is None:
        temperature = to_float(pd.to_numeric(free_rows["temperature_C"], errors="coerce").dropna().median(), 37.0)

    free_payload = {
        "carrier_family": carrier_family,
        "carrier_type": carrier_type,
        "drug_name": mode_or_default(free_rows["drug_name"], str(payload.get("drug_name") or "unknown")),
        "release_medium": medium,
        "particle_size_nm": None,
        "zeta_potential_mv": None,
        "pdi": None,
        "ph": to_float(payload.get("ph"), to_float(pd.to_numeric(free_rows["ph"], errors="coerce").dropna().median(), 7.4)),
        "temperature_C": temperature,
        "drug_loading_content_percent": None,
        "encapsulation_efficiency_percent": None,
    }
    message = (
        f"Free reference uses {carrier_type} evidence; size, zeta, PDI, loading, and EE are fixed as not applicable."
    )
    return free_payload, free_rows, message


def local_shap(package: dict, query: pd.DataFrame) -> list[dict]:
    pipeline = package["pipeline"]
    transformed = pipeline.named_steps["preprocessor"].transform(query)
    explainer = package.get("_tree_explainer")
    if explainer is None:
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        package["_tree_explainer"] = explainer
    shap_values = np.asarray(explainer.shap_values(transformed))[0]
    ranked = sorted(zip(package["feature_names"], shap_values), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "shap_value": float(value)} for name, value in ranked[:10]]


def theory_lines(query: pd.Series, shap_top: list[dict], kp: dict | None) -> list[str]:
    names = " ".join(item["feature"] for item in shap_top[:8]).lower()
    lines = []
    if "time" in names:
        lines.append("Time-related terms are major drivers, matching Noyes-Whitney release rate accumulating into Higuchi/Korsmeyer-Peppas style cumulative release.")
    if "ph" in names or "acidic" in names:
        if float(query["ph"]) < 7.0:
            lines.append("Acidic pH can shift release through solubility, ionization, carrier swelling, degradation, or membrane destabilization.")
        else:
            lines.append("Near-neutral pH suggests the carrier remains comparatively stable, so release is less pH-triggered.")
    if "particle_size" in names or "surface_area" in names:
        lines.append("Particle size contributes through the exposed-area term: smaller particles tend to increase effective surface area and early release.")
    if "zeta" in names:
        lines.append("Zeta potential appears as a dispersion-stability proxy; stronger absolute zeta can preserve effective surface area by reducing aggregation.")
    if "loading" in names or "encapsulation" in names:
        lines.append("Loading and encapsulation features act as concentration-gradient and retention proxies, affecting how much drug can diffuse outward.")
    if "carrier_type" in names or "drug_name" in names:
        lines.append("Carrier and drug identity encode hidden chemistry: diffusion coefficient, binding strength, swelling, erosion, and degradation behavior.")
    if kp:
        lines.append(f"Nearest-curve Korsmeyer-Peppas estimate is n={kp['n']:.3f}; {kp['interpretation']}")
    return lines or ["The prediction is mainly supported by nearest training patterns, but no single theory feature dominates the local SHAP explanation."]


def reliability(warnings: list[str], nearest_distance: float) -> str:
    if len(warnings) >= 4:
        return "low"
    if warnings:
        return "medium"
    if nearest_distance <= 0.15:
        return "high"
    if nearest_distance <= 0.35:
        return "medium-high"
    return "medium"


def forest_intervals(package: dict, query: pd.DataFrame) -> list[dict]:
    pipeline = package["pipeline"]
    transformed = pipeline.named_steps["preprocessor"].transform(query)
    tree_predictions = np.asarray(
        [estimator.predict(transformed) for estimator in pipeline.named_steps["model"].estimators_]
    )
    lower = np.percentile(tree_predictions, 10, axis=0)
    median = np.percentile(tree_predictions, 50, axis=0)
    upper = np.percentile(tree_predictions, 90, axis=0)
    spread = upper - lower
    rows = []
    for p10, p50, p90, width in zip(lower, median, upper, spread):
        rows.append(
            {
                "p10": float(np.clip(p10, 0, 100)),
                "p50": float(np.clip(p50, 0, 100)),
                "p90": float(np.clip(p90, 0, 100)),
                "width": float(max(0.0, width)),
            }
        )
    return rows


def reliability_reason(level: str, warnings: list[str], nearest_distance: float, interval: dict) -> str:
    imputed = sum("missing value" in warning for warning in warnings)
    out_of_range = sum("outside training range" in warning for warning in warnings)
    width = float(interval.get("width", 0.0))
    if nearest_distance <= 0.15:
        distance_text = "nearest evidence is close"
    elif nearest_distance <= 0.35:
        distance_text = "nearest evidence is moderately close"
    else:
        distance_text = "nearest evidence is sparse"

    reasons = [distance_text]
    if imputed:
        reasons.append(f"{imputed} input{'s' if imputed != 1 else ''} were imputed")
    if out_of_range:
        reasons.append(f"{out_of_range} input{'s' if out_of_range != 1 else ''} are outside the training range")
    if width >= 35:
        reasons.append("RF tree predictions are widely spread")
    elif width <= 15:
        reasons.append("RF tree predictions are relatively consistent")
    return f"Reliability is {level} because " + ", ".join(reasons) + "."


def predict_response(package: dict, payload: dict) -> dict:
    query = query_frame(payload)
    pipeline = package["pipeline"]
    prediction = float(np.clip(pipeline.predict(query)[0], 0, 100))
    interval = forest_intervals(package, query)[0]
    shap_top = local_shap(package, query)
    training = package["training_frame"]
    warnings = training_range_warnings(query, training)
    near = nearest_rows(query, training, n=5)
    nearest_distance = float(near.iloc[0]["distance"]) if not near.empty else 999.0
    level = reliability(warnings, nearest_distance)
    if interval["width"] >= 50 and level in {"high", "medium-high"}:
        level = "medium"
    elif interval["width"] >= 35 and level == "high":
        level = "medium-high"
    kp = None
    if not near.empty:
        curve_key = near.iloc[0]["curve_group"]
        kp = korsmeyer_peppas_for_curve(training[training["curve_group"] == curve_key])
    near_records = []
    for _, row in near.iterrows():
        near_records.append(
            {
                "distance": float(row["distance"]),
                "paper_id": str(row.get("paper_id", "")),
                "curve_id": str(row.get("curve_id", "")),
                "carrier_type": str(row.get("carrier_type", "")),
                "drug_name": str(row.get("drug_name", "")),
                "ph": to_float(row.get("ph")),
                "time_h": to_float(row.get("time_h")),
                "drug_release_percent": to_float(row.get(TARGET)),
            }
        )
    releases = [row["drug_release_percent"] for row in near_records if row["drug_release_percent"] is not None]
    return {
        "input": {key: (None if pd.isna(value) else value) for key, value in query.iloc[0].to_dict().items()},
        "prediction": prediction,
        "prediction_interval": interval,
        "expected_value": float(package["expected_value"]),
        "reliability": level,
        "nearest_distance": nearest_distance,
        "evidence_band_available": nearest_distance <= 0.35,
        "reliability_reason": reliability_reason(level, warnings, nearest_distance, interval),
        "warnings": warnings,
        "shap_top": shap_top,
        "interpretation": theory_lines(query.iloc[0], shap_top, kp),
        "korsmeyer_peppas": kp,
        "nearest_rows": near_records,
        "neighbor_release_min": float(min(releases)) if releases else prediction,
        "neighbor_release_max": float(max(releases)) if releases else prediction,
    }


def curve_response(package: dict, payload: dict) -> dict:
    times = payload.get("times") or DEFAULT_TIMES
    focus_query = query_frame(payload)
    focus_near = nearest_rows(focus_query, package["training_frame"], n=5)
    nearest_distance = float(focus_near.iloc[0]["distance"]) if not focus_near.empty else 999.0
    evidence_band_available = nearest_distance <= 0.35
    focus_releases = pd.to_numeric(focus_near[TARGET], errors="coerce").dropna().tolist()
    fallback_min = float(min(focus_releases)) if focus_releases else 0.0
    fallback_max = float(max(focus_releases)) if focus_releases else 100.0
    queries = []
    query_times = []
    rows = []
    for raw_time in times:
        query = query_frame({**payload, "time_h": raw_time})
        queries.append(query)
        query_times.append(float(query.iloc[0]["time_h"]))
    batch = pd.concat(queries, ignore_index=True)
    predictions = np.clip(package["pipeline"].predict(batch), 0, 100)
    intervals = forest_intervals(package, batch)
    for time_h, prediction, interval in zip(query_times, predictions, intervals):
        rows.append(
            {
                "input": {"time_h": time_h},
                "prediction": float(prediction),
                "prediction_interval": interval,
                "neighbor_release_min": fallback_min if evidence_band_available else None,
                "neighbor_release_max": fallback_max if evidence_band_available else None,
                "evidence_band_available": evidence_band_available,
                "nearest_distance": nearest_distance,
            }
        )
    message = None
    if not evidence_band_available:
        message = "Nearest evidence band is hidden because the closest training evidence is too far from this input."
    return {"points": rows, "evidence_band_available": evidence_band_available, "nearest_distance": nearest_distance, "message": message}


def free_curve_response(package: dict, payload: dict) -> dict:
    training = package["training_frame"]
    free_payload, free_rows, message = free_reference_payload(training, payload)
    if free_payload is None:
        return {"available": False, "message": message, "points": []}

    times = payload.get("times") or DEFAULT_TIMES
    releases = pd.to_numeric(free_rows[TARGET], errors="coerce").dropna().tolist()
    fallback_min = float(min(releases)) if releases else 0.0
    fallback_max = float(max(releases)) if releases else 100.0
    queries = []
    query_times = []
    for raw_time in times:
        query = query_frame({**free_payload, "time_h": raw_time})
        queries.append(query)
        query_times.append(float(query.iloc[0]["time_h"]))
    batch = pd.concat(queries, ignore_index=True)
    predictions = np.clip(package["pipeline"].predict(batch), 0, 100)
    points = []
    for time_h, prediction in zip(query_times, predictions):
        points.append(
            {
                "input": {"time_h": time_h},
                "prediction": float(prediction),
                "neighbor_release_min": fallback_min,
                "neighbor_release_max": fallback_max,
            }
        )
    return {
        "available": True,
        "message": message,
        "fixed_inputs": free_payload,
        "points": points,
    }


def ask_response(package: dict, payload: dict) -> dict:
    question = str(payload.get("question") or "").strip()
    if not question:
        return {"answer": ["Enter a question about the current condition."]}

    base = predict_response(package, payload)
    base_prediction = base["prediction"]
    candidates = []

    def trial(label: str, changes: dict) -> None:
        trial_payload = dict(payload)
        trial_payload.update(changes)
        try:
            result = predict_response(package, trial_payload)
        except ValueError:
            return
        candidates.append(
            {
                "label": label,
                "prediction": result["prediction"],
                "delta": result["prediction"] - base_prediction,
            }
        )

    ph = to_float(payload.get("ph"), 7.4)
    size = to_float(payload.get("particle_size_nm"))
    temp = to_float(payload.get("temperature_C"))
    loading = to_float(payload.get("drug_loading_content_percent"))
    zeta = to_float(payload.get("zeta_potential_mv"))

    if ph is not None:
        trial("lower pH by 1.0", {"ph": max(0.0, ph - 1.0)})
        trial("raise pH by 1.0", {"ph": min(14.0, ph + 1.0)})
    if size is not None:
        trial("reduce particle size by 20%", {"particle_size_nm": max(1.0, size * 0.8)})
        trial("increase particle size by 20%", {"particle_size_nm": min(5000.0, size * 1.2)})
    if temp is not None:
        trial("raise temperature by 5 C", {"temperature_C": min(100.0, temp + 5.0)})
    if loading is not None:
        trial("raise drug loading by 20% relative", {"drug_loading_content_percent": min(100.0, loading * 1.2)})
    if zeta is not None and zeta != 0:
        trial("increase absolute zeta by 20%", {"zeta_potential_mv": max(-250.0, min(250.0, zeta * 1.2))})

    ranked = sorted(candidates, key=lambda item: item["delta"], reverse=True)
    positive = [item for item in ranked if item["delta"] > 0.25]
    answer = [
        f"Current prediction is {base_prediction:.1f}% at {payload.get('time_h')} h.",
    ]
    if positive:
        best = positive[0]
        answer.append(f"Within this integrated nanocarrier model, the strongest tested lever is to {best['label']}: predicted release changes by {best['delta']:+.1f} percentage points.")
        if len(positive) > 1:
            second = positive[1]
            answer.append(f"The next candidate is to {second['label']}: {second['delta']:+.1f} percentage points.")
    elif "increase" in question.lower() or "올" in question or "높" in question:
        answer.append("The tested local changes did not clearly increase release; the model is likely relying on drug identity/time rather than a controllable input nearby.")
    else:
        answer.append("I compared local what-if changes for pH, particle size, temperature, loading, and zeta potential against the current prediction.")
        for item in ranked[:3]:
            answer.append(f"{item['label']}: {item['delta']:+.1f} percentage points.")
    answer.append("Treat this as screening guidance, not an experimental guarantee; formulation-specific validation is still needed.")
    return {"answer": answer, "what_if": ranked}


class RFShapHandler(BaseHTTPRequestHandler):
    package: dict | None = None
    summary: dict | None = None
    metadata: dict | None = None

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_html()
            return
        if path == "/api/metadata":
            self.send_json(self.metadata or {})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/predict", "/api/curve", "/api/free-curve", "/api/ask"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if path == "/api/ask":
                self.send_json(ask_response(self.package, payload))
            elif path == "/api/free-curve":
                self.send_json(free_curve_response(self.package, payload))
            elif path == "/api/curve":
                self.send_json(curve_response(self.package, payload))
            else:
                self.send_json(predict_response(self.package, payload))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> None:
    default_port = int(os.environ.get("PORT", "8766"))
    default_host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    parser = argparse.ArgumentParser(description="Run the RF/SHAP drug release UI.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rf_shap_all_nanocarrier_final_model.joblib")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "rf_shap_all_nanocarrier_final_summary.json")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()

    RFShapHandler.package = joblib.load(args.model)
    RFShapHandler.package["pipeline"].named_steps["model"].set_params(n_jobs=-1)
    RFShapHandler.package["_tree_explainer"] = shap.TreeExplainer(
        RFShapHandler.package["pipeline"].named_steps["model"]
    )
    if args.summary.exists():
        RFShapHandler.summary = json.loads(args.summary.read_text(encoding="utf-8"))
    metrics = (RFShapHandler.summary or {}).get("metrics", {})
    training = RFShapHandler.package["training_frame"]
    RFShapHandler.metadata = {
        "rows": metrics.get("rows", len(training)),
        "curves": metrics.get("curves", int(training["curve_group"].nunique())),
        "papers": metrics.get("papers", int(training["paper_id"].nunique())),
        "group_cv_r2": metrics.get("group_cv_r2", math.nan),
        "group_cv_rmse": metrics.get("group_cv_rmse", math.nan),
        "carrier_types": public_carrier_types(training),
        "drug_names": sorted_values(training, "drug_name"),
        "free_drug_names": free_reference_drugs(training),
        "release_media": public_release_media(training),
    }
    server = HTTPServer((args.host, args.port), RFShapHandler)
    print(f"Nanocarrier Release Agent running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except BaseException as exc:
        print(f"Nanocarrier Release Agent stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
