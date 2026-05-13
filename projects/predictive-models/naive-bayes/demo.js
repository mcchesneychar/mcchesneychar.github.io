// Naive Bayes — in-browser demo
// JS port of discrete-Laplace-smoothed and Gaussian Nbayes.

(function () {
  const TARGET = "Committed Crime";
  const DISCRETE = ["Married","Urban","Union","HS Graduate","Brother Criminal","Probation Before","Charged Before"];
  const CONTINUOUS = ["Hours Worked","Wage","Education","Work Experience"];

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

  function uniqueValues(rows, col) {
    return Array.from(new Set(rows.map(r => r[col])));
  }

  function priors(rows) {
    const classes = uniqueValues(rows, TARGET);
    const out = new Map();
    for (const c of classes) {
      out.set(c, rows.filter(r => r[TARGET] === c).length / rows.length);
    }
    return out;
  }

  // discrete: P(feature = v | class) with Laplace smoothing
  function discreteLikelihood(rows, classRows, col, val, k) {
    const domainSize = uniqueValues(rows, col).length;
    const matches = classRows.filter(r => r[col] === val).length;
    return (matches + k) / (classRows.length + domainSize * k);
  }

  // continuous: Gaussian PDF at val using class mean/std
  function gaussianLikelihood(classRows, col, val) {
    const xs = classRows.map(r => r[col]);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
    const sigma = Math.sqrt(variance);
    if (sigma === 0) return val === mean ? 1 : 0;
    return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-((val - mean) ** 2) / (2 * variance));
  }

  function posteriors(rows, obs, features, kind, k) {
    const classes = uniqueValues(rows, TARGET);
    const pri = priors(rows);
    const scores = new Map();
    for (const c of classes) {
      const classRows = rows.filter(r => r[TARGET] === c);
      let score = pri.get(c);
      for (const f of features) {
        if (kind === "discrete") {
          score *= discreteLikelihood(rows, classRows, f, obs[f], k);
        } else {
          score *= gaussianLikelihood(classRows, f, obs[f]);
        }
      }
      scores.set(c, score);
    }
    // normalize
    const total = Array.from(scores.values()).reduce((a, b) => a + b, 0);
    const norm = new Map();
    for (const [c, s] of scores) norm.set(c, total === 0 ? 0 : s / total);
    return norm;
  }

  function predict(rows, obs, features, kind, k) {
    const scores = posteriors(rows, obs, features, kind, k);
    let bestC = null, bestS = -1;
    for (const [c, s] of scores) if (s > bestS) { bestS = s; bestC = c; }
    return bestC;
  }

  let DATA = null;

  function run() {
    if (!DATA) return;
    const featureSet = document.getElementById("featureSet").value;
    const k = parseInt(document.getElementById("laplace").value, 10);
    const rowIdxRaw = parseInt(document.getElementById("rowIdx").value, 10);
    const rowIdx = Math.max(0, Math.min(DATA.length - 1, isNaN(rowIdxRaw) ? 0 : rowIdxRaw));
    document.getElementById("laplaceVal").textContent = k;

    const features = featureSet === "discrete" ? DISCRETE : CONTINUOUS;
    const kind = featureSet === "discrete" ? "discrete" : "continuous";
    const obs = DATA[rowIdx];

    const t0 = performance.now();
    const probs = posteriors(DATA, obs, features, kind, k);
    const pred = predict(DATA, obs, features, kind, k);

    // Overall accuracy on a 200-row sample to keep things fast
    const sampleN = Math.min(200, DATA.length);
    let correct = 0;
    for (let i = 0; i < sampleN; i++) {
      if (predict(DATA, DATA[i], features, kind, k) === DATA[i][TARGET]) correct++;
    }
    const acc = correct / sampleN;
    const elapsed = performance.now() - t0;

    document.getElementById("rows").textContent = DATA.length;
    document.getElementById("predicted").textContent = pred;
    document.getElementById("actual").textContent = obs[TARGET];
    document.getElementById("acc").textContent = acc.toFixed(3);

    const bars = document.getElementById("probBars");
    bars.innerHTML = Array.from(probs.entries()).map(([c, p]) => `
      <div class="bar-row">
        <div class="label">Class ${c}</div>
        <div class="track"><div class="fill" style="width: ${(p * 100).toFixed(1)}%"></div></div>
        <div class="num">${(p * 100).toFixed(1)}%</div>
      </div>
    `).join("");

    document.getElementById("status").textContent =
      `Inspecting row ${rowIdx} · ${features.length} ${featureSet} features · accuracy on first ${sampleN} rows · ${elapsed.toFixed(0)} ms`;
  }

  fetch("data/section4-dataset.csv")
    .then(r => r.text())
    .then(text => {
      DATA = parseCSV(text);
      document.getElementById("rowIdx").max = DATA.length - 1;
      ["featureSet","laplace","rowIdx"].forEach(id =>
        document.getElementById(id).addEventListener("input", run)
      );
      run();
    })
    .catch(err => {
      document.getElementById("status").textContent = `Could not load dataset: ${err.message}`;
    });
})();
