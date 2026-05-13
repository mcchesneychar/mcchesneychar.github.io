// Gradient Descent — in-browser demo
// JS port of linear / logistic batch GD with optional standardization,
// rendered as a live convergence chart on a <canvas>.

(function () {
  const TARGET = "Committed Crime";
  const FEATURE = "Education";

  function parseCSV(text) {
    const lines = text.trim().split("\n");
    const headers = lines[0].split(",");
    return lines.slice(1).map(line => {
      const cells = line.split(",");
      const row = {};
      headers.forEach((h, i) => {
        const v = cells[i];
        const num = Number(v);
        row[h] = Number.isNaN(num) ? v : num;
      });
      return row;
    });
  }

  function stableSigmoid(z) {
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }

  function descend(xRaw, yRaw, opts) {
    const { mode, alpha, maxSteps, standardize, tol = 1e-9 } = opts;
    const n = yRaw.length;
    let mu = 0, sigma = 1;
    if (standardize) {
      mu = xRaw.reduce((a, b) => a + b, 0) / n;
      const variance = xRaw.reduce((s, v) => s + (v - mu) ** 2, 0) / n;
      sigma = Math.sqrt(variance) || 1;
    }
    const x = xRaw.map(v => standardize ? (v - mu) / sigma : v);

    let w0 = 0, w1 = 0;
    let prevLoss = Infinity;
    const losses = [];
    let stepsDone = 0;
    for (let step = 0; step < maxSteps; step++) {
      let g0 = 0, g1 = 0, loss = 0;
      if (mode === "logistic") {
        for (let i = 0; i < n; i++) {
          const p = stableSigmoid(w0 + w1 * x[i]);
          const err = p - yRaw[i];
          g0 += err;
          g1 += err * x[i];
          const eps = 1e-15;
          loss += -(yRaw[i] * Math.log(p + eps) + (1 - yRaw[i]) * Math.log(1 - p + eps));
        }
        g0 /= n; g1 /= n; loss /= n;
        w0 -= alpha * g0;
        w1 -= alpha * g1;
        if (loss > prevLoss) { losses.push(loss); stepsDone = step + 1; break; }
      } else {
        for (let i = 0; i < n; i++) {
          const err = (w0 + w1 * x[i]) - yRaw[i];
          g0 += err;
          g1 += err * x[i];
          loss += err * err;
        }
        g0 = (2 / n) * g0;
        g1 = (2 / n) * g1;
        loss /= n;
        w0 -= alpha * g0;
        w1 -= alpha * g1;
      }
      losses.push(loss);
      stepsDone = step + 1;
      if (Math.abs(prevLoss - loss) < tol) break;
      prevLoss = loss;
    }

    // Un-standardize weights so they're interpretable in the original space
    const w0_orig = standardize ? (w0 - w1 * mu / sigma) : w0;
    const w1_orig = standardize ? (w1 / sigma) : w1;
    return { w0: w0_orig, w1: w1_orig, steps: stepsDone, losses };
  }

  function rmse(xRaw, yRaw, w0, w1, mode) {
    let s = 0;
    for (let i = 0; i < yRaw.length; i++) {
      let yhat = w0 + w1 * xRaw[i];
      if (mode === "logistic") yhat = stableSigmoid(yhat);
      s += (yhat - yRaw[i]) ** 2;
    }
    return Math.sqrt(s / yRaw.length);
  }

  // ---------- Canvas chart ----------
  function drawChart(losses) {
    const canvas = document.getElementById("lossChart");
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (losses.length < 2) return;

    const pad = 36;
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const range = maxLoss - minLoss || 1;

    // grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad + ((height - 2 * pad) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(width - pad, y);
      ctx.stroke();
    }

    // axes labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
    ctx.font = "11px 'SF Mono', monospace";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const lossAt = maxLoss - (range * i) / 4;
      const y = pad + ((height - 2 * pad) * i) / 4;
      ctx.fillText(lossAt.toFixed(3), 4, y);
    }
    ctx.textAlign = "center";
    ctx.fillText(`step 1`, pad, height - 12);
    ctx.fillText(`step ${losses.length}`, width - pad, height - 12);

    // gradient stroke
    const grad = ctx.createLinearGradient(0, 0, width, 0);
    grad.addColorStop(0, "#ff2a5f");
    grad.addColorStop(0.5, "#ff7b00");
    grad.addColorStop(1, "#ffb700");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    losses.forEach((loss, i) => {
      const x = pad + ((width - 2 * pad) * i) / (losses.length - 1);
      const y = pad + ((height - 2 * pad) * (maxLoss - loss)) / range;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  let DATA = null;

  function run() {
    if (!DATA) return;
    const mode = document.getElementById("mode").value;
    const alpha = parseFloat(document.getElementById("alpha").value);
    const maxSteps = parseInt(document.getElementById("maxSteps").value, 10);
    const standardize = document.getElementById("standardize").value === "yes";
    document.getElementById("alphaVal").textContent = alpha.toFixed(3);

    const xRaw = DATA.map(r => r[FEATURE]);
    const yRaw = DATA.map(r => r[TARGET]);

    const t0 = performance.now();
    const { w0, w1, steps, losses } = descend(xRaw, yRaw, { mode, alpha, maxSteps, standardize });
    const finalRMSE = rmse(xRaw, yRaw, w0, w1, mode);
    const elapsed = performance.now() - t0;

    document.getElementById("steps").textContent = steps;
    document.getElementById("w0").textContent = w0.toFixed(4);
    document.getElementById("w1").textContent = w1.toFixed(4);
    document.getElementById("rmse").textContent = finalRMSE.toFixed(4);

    drawChart(losses);
    document.getElementById("status").textContent =
      `Mode: ${mode} · ${standardize ? "standardized" : "raw"} feature · ${steps} steps · ${elapsed.toFixed(0)} ms`;
  }

  fetch("data/section5-dataset.csv")
    .then(r => r.text())
    .then(text => {
      DATA = parseCSV(text);
      ["mode","alpha","maxSteps","standardize"].forEach(id =>
        document.getElementById(id).addEventListener("input", run)
      );
      window.addEventListener("resize", run);
      run();
    })
    .catch(err => {
      document.getElementById("status").textContent = `Could not load dataset: ${err.message}`;
    });
})();
