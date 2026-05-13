// Nearest Neighbors — in-browser demo
// JS port of k-NN over the live CSV dataset, with selectable metric and k.

(function () {
  const TARGET = "Committed Crime";
  const SEED = 42;

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

  function rowVector(row, features) { return features.map(f => row[f]); }

  function dist(a, b, method) {
    if (method === "Euclidean") {
      let s = 0;
      for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
      return Math.sqrt(s);
    }
    if (method === "Manhattan") {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
      return s;
    }
    if (method === "Minkowski3") {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** 3;
      return Math.cbrt(s);
    }
    if (method === "Cosine") {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      const denom = Math.sqrt(na) * Math.sqrt(nb);
      if (denom === 0) return 1;
      return 1 - (dot / denom);
    }
    throw new Error("Unknown metric: " + method);
  }

  function knnPredict(trainVecs, trainLabels, testVec, k, metric) {
    const distances = trainVecs.map((v, i) => ({ d: dist(testVec, v, metric), label: trainLabels[i] }));
    distances.sort((a, b) => a.d - b.d);
    const top = distances.slice(0, k);
    const counts = new Map();
    for (const t of top) counts.set(t.label, (counts.get(t.label) || 0) + 1);
    let bestLabel = null, bestCount = -1;
    for (const [label, count] of counts) if (count > bestCount) { bestCount = count; bestLabel = label; }
    return bestLabel;
  }

  function shuffleSplit(rows, testRatio, seed) {
    let s = seed >>> 0;
    const rand = () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const shuffled = rows.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const cut = Math.floor(shuffled.length * (1 - testRatio));
    return [shuffled.slice(0, cut), shuffled.slice(cut)];
  }

  let DATA = null, FEATURES = null;

  function run() {
    if (!DATA) return;
    const k = parseInt(document.getElementById("k").value, 10);
    const metric = document.getElementById("metric").value;
    const testN = parseInt(document.getElementById("testN").value, 10);
    document.getElementById("kval").textContent = k;

    const [trainRows, testRowsAll] = shuffleSplit(DATA, 0.2, SEED);
    const testRows = testRowsAll.slice(0, testN);

    const trainVecs = trainRows.map(r => rowVector(r, FEATURES));
    const trainLabels = trainRows.map(r => r[TARGET]);

    const t0 = performance.now();
    let correct = 0;
    const predictions = [];
    for (const testRow of testRows) {
      const pred = knnPredict(trainVecs, trainLabels, rowVector(testRow, FEATURES), k, metric);
      const actual = testRow[TARGET];
      if (pred === actual) correct++;
      predictions.push({ pred, actual });
    }
    const elapsed = performance.now() - t0;

    document.getElementById("trainN").textContent = trainRows.length;
    document.getElementById("testRows").textContent = testRows.length;
    document.getElementById("acc").textContent = (correct / testRows.length).toFixed(3);
    document.getElementById("time").textContent = `${elapsed.toFixed(0)} ms`;

    const tbody = document.querySelector("#predTable tbody");
    tbody.innerHTML = predictions.slice(0, 10).map((p, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${p.pred}</td>
        <td>${p.actual}</td>
        <td class="${p.pred === p.actual ? "ok" : "bad"}">${p.pred === p.actual ? "✓" : "✗"}</td>
      </tr>
    `).join("");

    document.getElementById("status").textContent =
      `Showing first 10 of ${testRows.length} test predictions · k=${k} · ${metric}`;
  }

  fetch("data/section3-dataset.csv")
    .then(r => r.text())
    .then(text => {
      DATA = parseCSV(text);
      FEATURES = Object.keys(DATA[0]).filter(k => k !== TARGET);
      ["k", "metric", "testN"].forEach(id =>
        document.getElementById(id).addEventListener("input", run)
      );
      run();
    })
    .catch(err => {
      document.getElementById("status").textContent = `Could not load dataset: ${err.message}`;
    });
})();
