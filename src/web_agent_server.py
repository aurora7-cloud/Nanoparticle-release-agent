import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_advanced_model import predict_ensemble  # noqa: E402


DEFAULT_TIMES = [0, 24, 48, 72, 168, 336, 504, 672, 840, 1008, 1176, 1344]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Drug Release AI Agent</title>
  <style>
    :root {
      --bg: #f5f7f8;
      --ink: #182025;
      --muted: #65717a;
      --line: #d8dee3;
      --panel: #ffffff;
      --accent: #136f63;
      --accent-2: #c45a35;
      --accent-3: #315f9f;
      --soft: #e9f3f1;
      --warn: #8d5b14;
      --shadow: 0 18px 45px rgba(25, 38, 46, 0.09);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      min-height: 100vh;
    }

    button, input, select {
      font: inherit;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
    }

    aside {
      background: #182025;
      color: #fff;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
      min-height: 100vh;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding-bottom: 6px;
    }

    .mark {
      width: 40px;
      height: 40px;
      border-radius: 8px;
      background: linear-gradient(135deg, #58c2a9, #e79b54);
      display: grid;
      place-items: center;
      color: #102025;
      font-weight: 800;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .subtitle {
      color: #a8b3bb;
      margin: 4px 0 0;
      font-size: 13px;
    }

    .field {
      display: grid;
      gap: 7px;
    }

    label {
      font-size: 12px;
      color: #c8d0d6;
      font-weight: 650;
    }

    input, select {
      width: 100%;
      border: 1px solid #3b474f;
      background: #222c32;
      color: #fff;
      border-radius: 7px;
      padding: 10px 11px;
      outline: none;
    }

    input:focus, select:focus {
      border-color: #6bd1b8;
      box-shadow: 0 0 0 3px rgba(107, 209, 184, 0.16);
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .primary {
      margin-top: auto;
      border: 0;
      border-radius: 7px;
      padding: 12px 14px;
      background: #6bd1b8;
      color: #102025;
      font-weight: 800;
      cursor: pointer;
    }

    .primary:disabled {
      cursor: wait;
      opacity: 0.7;
    }

    main {
      min-width: 0;
      padding: 26px;
      display: grid;
      gap: 18px;
      align-content: start;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }

    .topbar h2 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }

    .status {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .metric {
      padding: 15px;
    }

    .metric .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .metric .value {
      margin-top: 8px;
      font-size: 26px;
      font-weight: 800;
      letter-spacing: 0;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.85fr);
      gap: 18px;
    }

    .panel {
      padding: 18px;
      min-width: 0;
    }

    .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }

    .panel-title h3 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .swatch {
      width: 18px;
      height: 3px;
      border-radius: 2px;
      background: var(--accent);
    }

    .swatch.band { background: #91c9be; }
    .swatch.ph64 { background: var(--accent-2); }
    .swatch.ph84 { background: var(--accent-3); }

    svg {
      width: 100%;
      height: auto;
      display: block;
    }

    .axis text {
      fill: var(--muted);
      font-size: 11px;
    }

    .axis line, .axis path, .grid-line {
      stroke: var(--line);
      stroke-width: 1;
    }

    .empty {
      color: var(--muted);
      min-height: 270px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
    }

    .reasoning {
      display: grid;
      gap: 10px;
    }

    .step {
      display: grid;
      grid-template-columns: 28px 1fr;
      gap: 10px;
      align-items: start;
    }

    .step-no {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: var(--soft);
      color: var(--accent);
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 12px;
    }

    .step strong {
      display: block;
      font-size: 13px;
    }

    .step p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    .warning {
      color: var(--warn);
      background: #fff7e8;
      border: 1px solid #efd5a3;
      border-radius: 7px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.45;
      margin-top: 12px;
    }

    @media (max-width: 980px) {
      .app {
        grid-template-columns: 1fr;
      }

      aside {
        min-height: auto;
      }

      .layout, .metrics {
        grid-template-columns: 1fr;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <div class="mark">AI</div>
        <div>
          <h1>Drug Release Agent</h1>
          <p class="subtitle">Curve-aware screening demo</p>
        </div>
      </div>

      <div class="field">
        <label for="carrier">Carrier</label>
        <select id="carrier">
          <option>PLGA nanoparticle</option>
          <option>DOX@PAA-MSN</option>
          <option>DOX@MSN</option>
          <option>PAA-MSN</option>
        </select>
      </div>

      <div class="field">
        <label for="drug">Drug</label>
        <select id="drug">
          <option>hydroxyl-FK866</option>
          <option>Doxorubicin hydrochloride (DOX)</option>
          <option>DOX</option>
        </select>
      </div>

      <div class="grid-2">
        <div class="field">
          <label for="size">Size nm</label>
          <input id="size" type="number" value="130" step="1" />
        </div>
        <div class="field">
          <label for="zeta">Zeta mV</label>
          <input id="zeta" type="number" value="-11.75" step="0.1" />
        </div>
      </div>

      <div class="grid-2">
        <div class="field">
          <label for="pdi">PDI</label>
          <input id="pdi" type="number" value="0.173" step="0.001" />
        </div>
        <div class="field">
          <label for="ph">pH</label>
          <input id="ph" type="number" value="7.4" step="0.1" />
        </div>
      </div>

      <div class="grid-2">
        <div class="field">
          <label for="loading">Loading %</label>
          <input id="loading" type="number" placeholder="optional" step="0.1" />
        </div>
        <div class="field">
          <label for="ee">EE %</label>
          <input id="ee" type="number" placeholder="optional" step="0.1" />
        </div>
      </div>

      <button id="run" class="primary" type="button">Run Agent</button>
    </aside>

    <main>
      <div class="topbar">
        <h2>Release Profile</h2>
        <div class="status"><span class="dot"></span><span id="status">Ready</span></div>
      </div>

      <section class="metrics">
        <div class="metric">
          <div class="label">672 h Prediction</div>
          <div class="value" id="m672">-</div>
        </div>
        <div class="metric">
          <div class="label">1344 h Prediction</div>
          <div class="value" id="m1344">-</div>
        </div>
        <div class="metric">
          <div class="label">Reliability</div>
          <div class="value" id="mRel">-</div>
        </div>
        <div class="metric">
          <div class="label">Nearest Curve</div>
          <div class="value" id="mCurve">-</div>
        </div>
      </section>

      <section class="layout">
        <div class="panel">
          <div class="panel-title">
            <h3>Predicted Curve</h3>
            <div class="legend">
              <span><i class="swatch"></i> prediction</span>
              <span><i class="swatch band"></i> neighbor range</span>
              <span><i class="swatch ph64"></i> pH 6.4</span>
              <span><i class="swatch ph84"></i> pH 8.4</span>
            </div>
          </div>
          <div id="chart" class="empty">Waiting for model output</div>
        </div>

        <div class="panel">
          <div class="panel-title">
            <h3>Agent Reasoning</h3>
          </div>
          <div class="reasoning" id="reasoning"></div>
          <div id="warnings"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>Nearest Paper Curves</h3>
        </div>
        <div id="neighbors"></div>
      </section>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const times = [0, 24, 48, 72, 168, 336, 504, 672, 840, 1008, 1176, 1344];

    function numberOrNull(id) {
      const value = $(id).value.trim();
      return value === "" ? null : Number(value);
    }

    function payload(phOverride = null) {
      return {
        carrier_type: $("carrier").value,
        drug_name: $("drug").value,
        release_medium: "PBS",
        particle_size_nm: numberOrNull("size"),
        zeta_potential_mv: numberOrNull("zeta"),
        pdi: numberOrNull("pdi"),
        ph: phOverride === null ? numberOrNull("ph") : phOverride,
        drug_loading_content_percent: numberOrNull("loading"),
        encapsulation_efficiency_percent: numberOrNull("ee"),
        times
      };
    }

    async function postJson(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    function scale(value, inMin, inMax, outMin, outMax) {
      if (inMax === inMin) return (outMin + outMax) / 2;
      return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
    }

    function pathFor(points, x, y) {
      return points.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.time_h).toFixed(2)} ${y(point.predicted_release_percent).toFixed(2)}`).join(" ");
    }

    function bandPath(points, x, y) {
      const upper = points.map((point) => `${x(point.time_h).toFixed(2)} ${y(point.upper_estimate).toFixed(2)}`);
      const lower = points.slice().reverse().map((point) => `${x(point.time_h).toFixed(2)} ${y(point.lower_estimate).toFixed(2)}`);
      return `M ${upper.join(" L ")} L ${lower.join(" L ")} Z`;
    }

    function drawChart(main, ph64, ph84) {
      const w = 900;
      const h = 430;
      const pad = { left: 54, right: 18, top: 18, bottom: 42 };
      const allPoints = [...main.points, ...ph64.points, ...ph84.points];
      const maxX = Math.max(...allPoints.map((point) => point.time_h));
      const maxY = Math.max(70, ...allPoints.flatMap((point) => [point.predicted_release_percent, point.upper_estimate || 0])) + 5;
      const x = (value) => scale(value, 0, maxX, pad.left, w - pad.right);
      const y = (value) => scale(value, 0, Math.min(maxY, 100), h - pad.bottom, pad.top);
      const yTicks = [0, 20, 40, 60, 80, 100].filter((value) => value <= Math.min(maxY, 100));
      const xTicks = [0, 168, 336, 672, 1008, 1344].filter((value) => value <= maxX);

      const grid = [
        ...yTicks.map((tick) => `<line class="grid-line" x1="${pad.left}" y1="${y(tick)}" x2="${w - pad.right}" y2="${y(tick)}"></line><text x="14" y="${y(tick) + 4}">${tick}%</text>`),
        ...xTicks.map((tick) => `<line class="grid-line" x1="${x(tick)}" y1="${pad.top}" x2="${x(tick)}" y2="${h - pad.bottom}"></line><text x="${x(tick) - 16}" y="${h - 14}">${tick}</text>`)
      ].join("");

      $("chart").className = "";
      $("chart").innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted drug release curve">
          <g class="axis">${grid}</g>
          <path d="${bandPath(main.points, x, y)}" fill="#91c9be" opacity="0.26"></path>
          <path d="${pathFor(ph64.points, x, y)}" fill="none" stroke="#c45a35" stroke-width="2.5" stroke-dasharray="7 7"></path>
          <path d="${pathFor(ph84.points, x, y)}" fill="none" stroke="#315f9f" stroke-width="2.5" stroke-dasharray="7 7"></path>
          <path d="${pathFor(main.points, x, y)}" fill="none" stroke="#136f63" stroke-width="4"></path>
          ${main.points.map((point) => `<circle cx="${x(point.time_h)}" cy="${y(point.predicted_release_percent)}" r="4.5" fill="#136f63"></circle>`).join("")}
          <text x="${w - 80}" y="${h - 14}" fill="#65717a" font-size="11">time h</text>
        </svg>
      `;
    }

    function renderReasoning(data) {
      const steps = [
        ["Normalize", `${data.input.carrier_type}, ${data.input.drug_name}, pH ${data.input.ph}`],
        ["Search", `${data.nearest_curves.length} nearest release curves selected`],
        ["Blend", `point kNN ${Math.round(data.blend.point_knn * 100)}% + curve kNN ${Math.round(data.blend.curve_knn * 100)}%`],
        ["Explain", `${data.reliability} reliability for screening`]
      ];
      $("reasoning").innerHTML = steps.map((step, index) => `
        <div class="step">
          <div class="step-no">${index + 1}</div>
          <div><strong>${step[0]}</strong><p>${step[1]}</p></div>
        </div>
      `).join("");
      $("warnings").innerHTML = data.warnings.length
        ? `<div class="warning">${data.warnings.map((item) => `<div>${item}</div>`).join("")}</div>`
        : "";
    }

    function renderNeighbors(data) {
      $("neighbors").innerHTML = `
        <table>
          <thead><tr><th>Curve</th><th>Carrier</th><th>Drug</th><th>pH</th><th>Distance</th><th>Estimate at 672 h</th></tr></thead>
          <tbody>
            ${data.nearest_curves.map((curve) => `
              <tr>
                <td>${curve.curve_id}</td>
                <td>${curve.carrier_type}</td>
                <td>${curve.drug_name}</td>
                <td>${curve.ph}</td>
                <td>${curve.distance.toFixed(3)}</td>
                <td>${curve.estimate_at_672h.toFixed(1)}%</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function updateMetrics(data) {
      const at672 = data.points.find((point) => point.time_h === 672) || data.points[data.points.length - 1];
      const at1344 = data.points.find((point) => point.time_h === 1344) || data.points[data.points.length - 1];
      $("m672").textContent = `${at672.predicted_release_percent.toFixed(1)}%`;
      $("m1344").textContent = `${at1344.predicted_release_percent.toFixed(1)}%`;
      $("mRel").textContent = data.reliability;
      $("mCurve").textContent = data.nearest_curves[0]?.curve_id || "-";
    }

    async function runAgent() {
      $("run").disabled = true;
      $("status").textContent = "Running";
      try {
        const main = await postJson("/api/predict_curve", payload());
        const ph64 = await postJson("/api/predict_curve", payload(6.4));
        const ph84 = await postJson("/api/predict_curve", payload(8.4));
        updateMetrics(main);
        drawChart(main, ph64, ph84);
        renderReasoning(main);
        renderNeighbors(main);
        $("status").textContent = "Updated";
      } catch (error) {
        $("status").textContent = "Error";
        $("chart").className = "empty";
        $("chart").textContent = error.message;
      } finally {
        $("run").disabled = false;
      }
    }

    $("run").addEventListener("click", runAgent);
    runAgent();
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


def reliability_label(warnings, nearest_curve_distance, nearest_row_distance):
    if len(warnings) >= 4:
        return "low"
    if warnings:
        return "medium-low"
    distances = [value for value in [nearest_curve_distance, nearest_row_distance] if value is not None]
    nearest = min(distances) if distances else 999.0
    if nearest <= 0.20:
        return "high"
    if nearest <= 0.45:
        return "medium-high"
    return "medium"


def range_warnings(query, package):
    warnings = []
    stats = package["numeric_stats"]
    for column, limits in stats.items():
        if column in {"log_time_h", "sqrt_time_h"}:
            continue
        value = to_float(query.get(column))
        if value is None:
            warnings.append(f"{column} was imputed with median {limits['median']:.3g}.")
        elif value < limits["min"] or value > limits["max"]:
            warnings.append(f"{column}={value:g} is outside {limits['min']:g}-{limits['max']:g}.")
    if query.get("carrier_type") not in package.get("carrier_types", []):
        warnings.append(f"carrier_type '{query.get('carrier_type')}' was not in training data.")
    if query.get("drug_name") not in package.get("drug_names", []):
        warnings.append(f"drug_name '{query.get('drug_name')}' was not in training data.")
    return warnings


def curve_response(package, payload):
    times = payload.get("times") or DEFAULT_TIMES
    clean_times = sorted({max(0.0, to_float(time, 0.0)) for time in times})
    base_query = {
        "carrier_type": str(payload.get("carrier_type") or "unknown"),
        "drug_name": str(payload.get("drug_name") or "unknown"),
        "release_medium": str(payload.get("release_medium") or "PBS"),
        "particle_size_nm": to_float(payload.get("particle_size_nm")),
        "zeta_potential_mv": to_float(payload.get("zeta_potential_mv")),
        "pdi": to_float(payload.get("pdi")),
        "ph": to_float(payload.get("ph"), 7.4),
        "drug_loading_content_percent": to_float(payload.get("drug_loading_content_percent")),
        "encapsulation_efficiency_percent": to_float(payload.get("encapsulation_efficiency_percent")),
    }

    points = []
    first_result = None
    for time_h in clean_times:
        query = dict(base_query)
        query["time_h"] = time_h
        result = predict_ensemble(
            query,
            package["training_rows"],
            package["curves"],
            package["numeric_stats"],
            int(package["k_rows"]),
            int(package["k_curves"]),
        )
        if first_result is None:
            first_result = result
        curve_estimates = [estimate for _, _, estimate in result["curve_neighbors"]]
        if curve_estimates:
            lower = min(curve_estimates)
            upper = max(curve_estimates)
        else:
            lower = upper = result["prediction"]
        points.append(
            {
                "time_h": time_h,
                "predicted_release_percent": result["prediction"],
                "point_prediction": result["point_prediction"],
                "curve_prediction": result["curve_prediction"],
                "lower_estimate": lower,
                "upper_estimate": upper,
            }
        )

    focus_query = dict(base_query)
    focus_query["time_h"] = 672.0
    focus_result = predict_ensemble(
        focus_query,
        package["training_rows"],
        package["curves"],
        package["numeric_stats"],
        int(package["k_rows"]),
        int(package["k_curves"]),
    )
    warnings = range_warnings(focus_query, package)
    nearest_curve = focus_result["curve_neighbors"][0][0] if focus_result["curve_neighbors"] else None
    nearest_row = focus_result["row_neighbors"][0][0] if focus_result["row_neighbors"] else None
    nearest_curves = []
    for distance, curve, estimate in focus_result["curve_neighbors"]:
        nearest_curves.append(
            {
                "distance": distance,
                "paper_id": curve.get("paper_id", ""),
                "curve_id": curve.get("curve_id", ""),
                "carrier_type": curve.get("carrier_type", ""),
                "drug_name": curve.get("drug_name", ""),
                "ph": curve.get("ph", ""),
                "estimate_at_672h": estimate,
            }
        )

    return {
        "input": base_query,
        "points": points,
        "blend": focus_result["blend"] if first_result else {"point_knn": 0.45, "curve_knn": 0.55},
        "reliability": reliability_label(warnings, nearest_curve, nearest_row),
        "warnings": warnings,
        "nearest_curves": nearest_curves,
    }


class AgentServer(BaseHTTPRequestHandler):
    package = None

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_html()
        elif path == "/api/health":
            self.send_json({"ok": True, "model_type": self.package.get("model_type")})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict_curve":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            self.send_json(curve_response(self.package, payload))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Run the local Drug Release AI Agent website.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "advanced_drug_release_model.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    AgentServer.package = json.loads(args.model.read_text(encoding="utf-8"))
    server = ThreadingHTTPServer((args.host, args.port), AgentServer)
    print(f"Drug Release AI Agent running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
