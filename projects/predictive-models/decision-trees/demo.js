// Decision Trees — in-browser demo
// JS port of the ID3 algorithm (with Gini option) over the live CSV dataset.

(function () {
  const TARGET = "Committed Crime";

  // ---------- CSV parsing ----------
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

  // ---------- impurity functions ----------
  function counts(rows, col) {
    const c = new Map();
    for (const r of rows) c.set(r[col], (c.get(r[col]) || 0) + 1);
    return c;
  }

  function entropy(rows, col) {
    const c = counts(rows, col);
    const n = rows.length;
    let h = 0;
    for (const v of c.values()) {
      const p = v / n;
      if (p > 0) h -= p * Math.log2(p);
    }
    return h;
  }

  function gini(rows, col) {
    const c = counts(rows, col);
    const n = rows.length;
    let g = 0;
    for (const v of c.values()) {
      const p = v / n;
      g += p * p;
    }
    return 1 - g;
  }

  // ---------- ID3 with selectable criterion ----------
  function impurityFn(name) { return name === "gini" ? gini : entropy; }

  function bestSplit(rows, features, criterion) {
    const impurity = impurityFn(criterion);
    const base = impurity(rows, TARGET);
    let bestFeature = null, bestGain = -Infinity;
    for (const f of features) {
      const groups = new Map();
      for (const r of rows) {
        const k = r[f];
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(r);
      }
      let weighted = 0;
      for (const sub of groups.values()) {
        weighted += (sub.length / rows.length) * impurity(sub, TARGET);
      }
      const gain = base - weighted;
      if (gain > bestGain) { bestGain = gain; bestFeature = f; }
    }
    return bestFeature;
  }

  function majority(rows) {
    const c = counts(rows, TARGET);
    let best = null, max = -1;
    for (const [k, v] of c) if (v > max) { best = k; max = v; }
    return best;
  }

  function buildTree(rows, features, depth, maxDepth, criterion) {
    if (rows.length === 0) return { leaf: 0 };
    if (depth >= maxDepth || features.length === 0 || counts(rows, TARGET).size === 1) {
      return { leaf: majority(rows) };
    }
    const feat = bestSplit(rows, features, criterion);
    const groups = new Map();
    for (const r of rows) {
      const k = r[feat];
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(r);
    }
    const remaining = features.filter(f => f !== feat);
    const branches = {};
    for (const [val, sub] of groups) {
      branches[val] = buildTree(sub, remaining, depth + 1, maxDepth, criterion);
    }
    return { feature: feat, branches, fallback: majority(rows) };
  }

  function predict(tree, row) {
    if ("leaf" in tree) return tree.leaf;
    const v = row[tree.feature];
    const next = tree.branches[v];
    if (next === undefined) return tree.fallback;
    return predict(next, row);
  }

  // ---------- train/test split (deterministic) ----------
  function splitData(rows, testRatio, seed) {
    // Mulberry32 PRNG for reproducibility
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

  function accuracy(tree, rows) {
    let correct = 0;
    for (const r of rows) if (predict(tree, r) === r[TARGET]) correct++;
    return rows.length === 0 ? 0 : correct / rows.length;
  }

  function countLeaves(tree) {
    if ("leaf" in tree) return 1;
    return Object.values(tree.branches).reduce((s, n) => s + countLeaves(n), 0);
  }

  // ---------- Render tree as HTML ----------
  function renderTree(tree, indent = 0) {
    const pad = "  ".repeat(indent);
    if ("leaf" in tree) {
      return `${pad}<span class="leaf">→ ${tree.leaf}</span>`;
    }
    const lines = [`${pad}<span class="feature">${tree.feature}</span>`];
    const entries = Object.entries(tree.branches);
    for (const [val, child] of entries) {
      lines.push(`${pad}  <span class="branch">[ = ${val} ]</span>`);
      lines.push(renderTree(child, indent + 2));
    }
    return lines.join("\n");
  }

  // ---------- Driver ----------
  let DATA = null;
  let FEATURES = null;

  function run() {
    if (!DATA) return;
    const depth = parseInt(document.getElementById("depth").value, 10);
    const criterion = document.getElementById("criterion").value;
    const seed = parseInt(document.getElementById("seed").value, 10);

    const t0 = performance.now();
    const [train, test] = splitData(DATA, 0.3, seed);
    const tree = buildTree(train, FEATURES, 0, depth, criterion);
    const trainAcc = accuracy(tree, train);
    const testAcc = accuracy(tree, test);
    const elapsed = performance.now() - t0;

    document.getElementById("rows").textContent = DATA.length;
    document.getElementById("trainAcc").textContent = trainAcc.toFixed(3);
    document.getElementById("testAcc").textContent = testAcc.toFixed(3);
    document.getElementById("leaves").textContent = countLeaves(tree);
    document.getElementById("tree").innerHTML = renderTree(tree);
    document.getElementById("status").textContent =
      `Train ${train.length} · Test ${test.length} · Built in ${elapsed.toFixed(1)} ms`;
  }

  fetch("data/section2-dataset.csv")
    .then(r => r.text())
    .then(text => {
      DATA = parseCSV(text);
      FEATURES = Object.keys(DATA[0]).filter(k => k !== TARGET);
      ["depth", "criterion", "seed"].forEach(id =>
        document.getElementById(id).addEventListener("change", run)
      );
      run();
    })
    .catch(err => {
      document.getElementById("tree").textContent = `Could not load dataset: ${err.message}`;
    });
})();
