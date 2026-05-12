// Resume-to-Job Matching System — read-only static viewer
// Loads precomputed JSON artifacts and renders five interactive tabs.
// No backend, no build step.

'use strict';

const DATA_FILES = {
  parsed:        './data/parsed_documents.json',
  ranked:        './data/ranked_matches.json',
  profiles:      './data/calibration_profiles.json',
  changes:       './data/calibration_changes.json',
  disagreements: './data/disagreement_log.json',
  rationales:    './data/cached_rationales.json',
};

const STATE = {
  data: {},
  resumesById: {},     // full id -> resume record
  jobsById: {},        // full id -> job record
  shortIdToResume: {}, // 'R014' -> resume record (some logs use short ids)
  shortIdToJob: {},    // 'J004' -> job record
  companyMeta: {},     // company_id -> profile object
  weightChart: null,
};

// ---------------------------------------------------------------- load
async function loadAll() {
  const errs = [];
  const entries = await Promise.all(
    Object.entries(DATA_FILES).map(async ([key, url]) => {
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
        return [key, await r.json()];
      } catch (e) {
        errs.push(`${key}: ${e.message}`);
        return [key, null];
      }
    })
  );

  for (const [k, v] of entries) STATE.data[k] = v;

  if (errs.length) {
    const box = document.getElementById('error');
    box.hidden = false;
    box.innerHTML =
      '<strong>Some data files failed to load:</strong><br>' + errs.join('<br>') +
      '<br><br>This static viewer expects the JSON files inside <code>./data/</code>. ' +
      'If you are opening it from <code>file://</code>, your browser may block the fetches — ' +
      'try <code>python -m http.server</code> in this folder instead.';
  }

  // Index lookups.
  if (STATE.data.parsed) {
    for (const r of STATE.data.parsed.resumes || []) {
      STATE.resumesById[r.id] = r;
      const short = r.id.split('_')[0];
      STATE.shortIdToResume[short] = r;
    }
    for (const j of STATE.data.parsed.jobs || []) {
      STATE.jobsById[j.id] = j;
      const short = j.id.split('_')[0];
      STATE.shortIdToJob[short] = j;
    }
  }
  if (STATE.data.profiles) {
    for (const [cid, prof] of Object.entries(STATE.data.profiles)) {
      STATE.companyMeta[cid] = prof;
    }
  }

  document.getElementById('loading').style.display = 'none';
}

// ---------------------------------------------------------------- utils
const escapeHtml = (s) =>
  String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const fmt = (n, digits = 3) =>
  (n === null || n === undefined || Number.isNaN(n)) ? '—' : Number(n).toFixed(digits);

const phaseBadge = (phase) => {
  const cls = { 'cold_start': 'amber', 'industry_defaults': 'pink', 'company_specific': 'green' }[phase] || '';
  const label = (phase || 'unknown').replace(/_/g, ' ');
  return `<span class="badge ${cls}">${label}</span>`;
};

const lookupResume = (id) => STATE.resumesById[id] || STATE.shortIdToResume[id] || null;
const lookupJob = (id) => STATE.jobsById[id] || STATE.shortIdToJob[id] || null;

const resumeLabel = (r) => {
  const name = r?.resume?.contact?.name;
  return name ? `${r.id} — ${name}` : r.id;
};
const jobLabel = (j) => {
  const title = j?.job?.title;
  return title ? `${j.id} — ${title}` : j.id;
};

// ---------------------------------------------------------------- tab nav
function initTabs() {
  const buttons = document.querySelectorAll('#tabNav button');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      buttons.forEach((b) => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.tab-panel').forEach((p) => {
        p.classList.toggle('active', p.id === target);
      });
      history.replaceState(null, '', '#' + target);
      // Re-render charts when their tab becomes visible.
      if (target === 'calibration') renderCalibrationCharts();
    });
  });

  // Hash routing on load.
  const hash = location.hash.slice(1);
  if (hash) {
    const target = document.querySelector(`#tabNav button[data-tab="${hash}"]`);
    if (target) target.click();
  }
}

// ---------------------------------------------------------------- Tab 1: Match Explorer
function initMatchExplorer() {
  const sel = document.getElementById('meResume');
  if (!STATE.data.parsed) return;
  const opts = STATE.data.parsed.resumes
    .filter((r) => r.resume)
    .sort((a, b) => a.id.localeCompare(b.id));
  for (const r of opts) {
    const o = document.createElement('option');
    o.value = r.id;
    o.textContent = resumeLabel(r);
    sel.appendChild(o);
  }
  sel.addEventListener('change', renderMatchExplorer);
  renderMatchExplorer();
}

function renderMatchExplorer() {
  const sel = document.getElementById('meResume');
  const rid = sel.value;
  const rawBody = document.getElementById('meRawBody');
  const adjBody = document.getElementById('meAdjBody');

  const raw = (STATE.data.ranked?.raw?.[rid]) || [];
  const adj = (STATE.data.ranked?.adjusted?.[rid]) || [];

  const renderRows = (matches) => matches.length
    ? matches.map((m) => {
        const job = lookupJob(m.job_id);
        const title = job?.job?.title || '';
        return `<tr>
          <td class="rank-cell">${m.rank}</td>
          <td><code>${escapeHtml(m.job_id)}</code></td>
          <td>${escapeHtml(title)}</td>
          <td class="score-cell">${fmt(m.score)}</td>
        </tr>`;
      }).join('')
    : '<tr><td colspan="4" class="empty">No matches above threshold.</td></tr>';

  rawBody.innerHTML = renderRows(raw);
  adjBody.innerHTML = renderRows(adj);

  // Rationale + overlap for the top adjusted match.
  const ration = document.getElementById('meRationale');
  const overlap = document.getElementById('meOverlap');
  if (adj.length > 0) {
    const topJobId = adj[0].job_id;
    const key = `${rid}::${topJobId}`;
    const rationale = STATE.data.rationales?.[key]?.rationale;
    if (rationale) {
      ration.innerHTML = `
        <p class="subhead">Why the top match</p>
        <div class="rationale">${escapeHtml(rationale)}</div>`;
    } else {
      ration.innerHTML = '';
    }

    const candidate = STATE.resumesById[rid]?.resume || {};
    const job = lookupJob(topJobId)?.job || {};
    const candSkills = new Set((candidate.skills || []).map((s) => s.toLowerCase()));
    const jobSkills = new Set((job.skills || []).map((s) => s.toLowerCase()));
    const shared = [...candSkills].filter((s) => jobSkills.has(s));

    overlap.innerHTML = `
      <p class="subhead">Matched fields (top adjusted match: <code>${escapeHtml(topJobId)}</code>)</p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        <div>
          <strong>Shared skills:</strong>
          <div class="pill-row" style="margin-top:6px">
            ${shared.length
              ? shared.map((s) => `<span class="pill">${escapeHtml(s)}</span>`).join('')
              : '<span class="muted">No directly overlapping skills</span>'}
          </div>
        </div>
        <div>
          <strong>Geography:</strong>
          <div class="muted" style="margin-top:6px">
            Candidate is in <em>${escapeHtml(candidate.contact?.location || 'unknown')}</em>;
            role is in <em>${escapeHtml(job.location || 'unspecified')}</em>.
          </div>
        </div>
      </div>`;
  } else {
    ration.innerHTML = '';
    overlap.innerHTML = '';
  }
}

// ---------------------------------------------------------------- Tab 2: Document Inspector
function initDocumentInspector() {
  const sel = document.getElementById('diSelect');
  const typeButtons = document.querySelectorAll('#diType button');

  let mode = 'resume';
  const fillOptions = () => {
    sel.innerHTML = '';
    const pool = mode === 'resume'
      ? (STATE.data.parsed?.resumes || [])
      : (STATE.data.parsed?.jobs || []);
    for (const item of pool.slice().sort((a, b) => a.id.localeCompare(b.id))) {
      const o = document.createElement('option');
      o.value = item.id;
      o.textContent = mode === 'resume' ? resumeLabel(item) : jobLabel(item);
      sel.appendChild(o);
    }
    renderDocumentInspector(mode);
  };

  typeButtons.forEach((b) => {
    b.addEventListener('click', () => {
      mode = b.dataset.type;
      typeButtons.forEach((x) => x.classList.toggle('active', x === b));
      fillOptions();
    });
  });
  sel.addEventListener('change', () => renderDocumentInspector(mode));
  fillOptions();
}

function renderDocumentInspector(mode) {
  const sel = document.getElementById('diSelect');
  const id = sel.value;
  const pool = mode === 'resume' ? STATE.resumesById : STATE.jobsById;
  const doc = pool[id];
  if (!doc) return;

  const sourceName = (doc.source_path || '').split('/').pop() || doc.source_path;
  const confColor = doc.parse_confidence >= 0.7 ? 'green'
                  : doc.parse_confidence >= 0.4 ? 'amber' : 'red';
  document.getElementById('diMeta').innerHTML = JSON.stringify({
    id: doc.id,
    source_path: sourceName,
    parse_confidence: Number((doc.parse_confidence || 0).toFixed(3)),
  }, null, 2);

  const payload = mode === 'resume' ? doc.resume : doc.job;
  document.getElementById('diSchema').textContent = JSON.stringify(payload, null, 2);

  // Add a confidence badge inline above the schema.
  const schemaEl = document.getElementById('diSchema');
  schemaEl.insertAdjacentHTML('beforebegin', ''); // no-op; we keep simple
  schemaEl.dataset.confidence = confColor;
}

// ---------------------------------------------------------------- Tab 3: Feedback Loop
function renderFeedback() {
  const profiles = STATE.data.profiles || {};
  const body = document.getElementById('fbBody');
  const rows = Object.values(profiles).map((p) => `
    <tr>
      <td><strong>${escapeHtml(p.name)}</strong><br><span class="muted">${escapeHtml(p.location || '')}</span></td>
      <td>${escapeHtml(p.industry_vertical || '')}</td>
      <td>${phaseBadge(p.calibration_phase)}</td>
      <td class="score-cell">${p.hire_count}</td>
      <td>${Object.keys(p.feature_weights || {}).length}</td>
      <td class="score-cell">${fmt(p.model_accuracy, 2)}</td>
    </tr>`).join('');
  body.innerHTML = rows || '<tr><td colspan="6" class="empty">No company data loaded.</td></tr>';
}

// ---------------------------------------------------------------- Tab 4: Disagreements
function initDisagreements() {
  const events = STATE.data.disagreements || [];
  const companies = [...new Set(events.map((e) => e.company_id))].sort();
  const types = [...new Set(events.map((e) => e.disagreement_type).filter(Boolean))].sort();
  const recruiters = [...new Set(events.map((e) => e.recruiter_id))].sort();

  const compSel = document.getElementById('dgCompany');
  const typeSel = document.getElementById('dgType');
  const recSel = document.getElementById('dgRecruiter');

  for (const c of companies) {
    const o = document.createElement('option');
    o.value = c;
    o.textContent = STATE.companyMeta[c]?.name || c;
    compSel.appendChild(o);
  }
  for (const t of types) {
    const o = document.createElement('option');
    o.value = t;
    o.textContent = t.replace(/_/g, ' ');
    typeSel.appendChild(o);
  }
  for (const r of recruiters) {
    const o = document.createElement('option');
    o.value = r;
    o.textContent = r;
    recSel.appendChild(o);
  }

  [compSel, typeSel, recSel].forEach((s) => s.addEventListener('change', renderDisagreements));
  renderDisagreements();
}

function renderDisagreements() {
  const events = STATE.data.disagreements || [];
  const c = document.getElementById('dgCompany').value;
  const t = document.getElementById('dgType').value;
  const r = document.getElementById('dgRecruiter').value;

  const filtered = events.filter((e) =>
    (!c || e.company_id === c) &&
    (!t || e.disagreement_type === t) &&
    (!r || e.recruiter_id === r)
  );

  const typeBadge = (type) => {
    if (!type) return '<span class="muted">—</span>';
    const cls = type === 'positive_override' ? 'green'
              : type === 'negative_override' || type === 'bulk_dismissal' ? 'red'
              : type === 'explicit_flag' ? 'amber'
              : 'pink';
    return `<span class="badge ${cls}">${type.replace(/_/g, ' ')}</span>`;
  };

  const body = document.getElementById('dgBody');
  body.innerHTML = filtered.length
    ? filtered.map((e) => {
        const when = (e.action_timestamp || '').split('T')[0];
        return `<tr>
          <td class="muted">${escapeHtml(when)}</td>
          <td>${escapeHtml(e.recruiter_id || '')}</td>
          <td>${escapeHtml(STATE.companyMeta[e.company_id]?.name || e.company_id)}</td>
          <td><code>${escapeHtml(e.candidate_id)}</code></td>
          <td><code>${escapeHtml(e.job_id)}</code></td>
          <td>${escapeHtml(e.recruiter_action || '')}</td>
          <td>${typeBadge(e.disagreement_type)}</td>
          <td class="muted">${e.system_rank}</td>
          <td class="score-cell">${fmt(e.system_score)}</td>
          <td class="muted">${escapeHtml(e.override_reason || '—')}</td>
        </tr>`;
      }).join('')
    : '<tr><td colspan="10" class="empty">No events match the current filters.</td></tr>';
}

// ---------------------------------------------------------------- Tab 5: Calibration Profile
function initCalibrationProfile() {
  const sel = document.getElementById('cpCompany');
  for (const [cid, prof] of Object.entries(STATE.data.profiles || {})) {
    const o = document.createElement('option');
    o.value = cid;
    o.textContent = prof.name;
    sel.appendChild(o);
  }
  sel.addEventListener('change', renderCalibrationCharts);
  renderCalibrationCharts();
}

function renderCalibrationCharts() {
  const sel = document.getElementById('cpCompany');
  const cid = sel.value;
  const profile = STATE.data.profiles?.[cid];
  if (!profile) return;

  // Metrics.
  document.getElementById('cpMetrics').innerHTML = `
    <div class="metric"><div class="v">${phaseBadge(profile.calibration_phase)}</div><div class="l">Phase</div></div>
    <div class="metric"><div class="v">${profile.hire_count}</div><div class="l">Hires logged</div></div>
    <div class="metric"><div class="v">${fmt(profile.model_accuracy, 2)}</div><div class="l">90-day accuracy</div></div>
    <div class="metric"><div class="v">${Object.keys(profile.feature_weights || {}).length}</div><div class="l">Tracked features</div></div>`;

  // Explicit rules.
  document.getElementById('cpRules').textContent = JSON.stringify(profile.explicit_rules || {}, null, 2);

  // Weight bars (top 15 by absolute distance from 1.0).
  const weights = profile.feature_weights || {};
  const entries = Object.entries(weights)
    .sort((a, b) => Math.abs(b[1] - 1) - Math.abs(a[1] - 1))
    .slice(0, 15);
  const wrap = document.getElementById('cpWeights');
  wrap.innerHTML = entries.length
    ? entries.map(([name, w]) => {
        const above = w >= 1;
        // Map weight 0.2..2.5 onto a centered bar where 1.0 is the midline.
        // Half-track to each side; clamp.
        const halfMax = 1.5; // ceiling is 2.5; floor is 0.2 (~ -0.8 below)
        const offset = Math.min(halfMax, Math.abs(w - 1));
        const pct = (offset / halfMax) * 50;
        const fillStyle = above
          ? `left:50%;width:${pct}%`
          : `right:50%;width:${pct}%;left:auto`;
        return `<div class="weight-bar">
          <div class="name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
          <div class="track">
            <div class="midline"></div>
            <div class="fill ${above ? '' : 'below'}" style="${fillStyle}"></div>
          </div>
          <div class="v">${fmt(w, 2)}</div>
        </div>`;
      }).join('')
    : '<div class="empty">No learned weights yet for this company.</div>';

  // Audit log filtered to this company.
  const changes = (STATE.data.changes || []).filter((c) => c.company_id === cid);
  const auditBody = document.getElementById('cpAuditBody');
  auditBody.innerHTML = changes.length
    ? changes.map((c) => {
        const when = (c.changed_at || '').split('T')[0];
        const delta = (c.new_weight || 0) - (c.old_weight || 0);
        const deltaCls = delta > 0 ? 'green' : 'red';
        return `<tr>
          <td class="muted">${escapeHtml(when)}</td>
          <td><code>${escapeHtml(c.feature)}</code></td>
          <td class="muted">${fmt(c.old_weight, 2)}</td>
          <td class="score-cell">${fmt(c.new_weight, 2)} <span class="badge ${deltaCls}">${delta >= 0 ? '+' : ''}${fmt(delta, 2)}</span></td>
          <td>${escapeHtml(c.source)}</td>
          <td class="muted">${escapeHtml(c.rationale)}</td>
        </tr>`;
      }).join('')
    : '<tr><td colspan="6" class="empty">No calibration changes recorded for this company yet.</td></tr>';
}

// ---------------------------------------------------------------- boot
(async function main() {
  await loadAll();
  initTabs();
  initMatchExplorer();
  initDocumentInspector();
  renderFeedback();
  initDisagreements();
  initCalibrationProfile();
})();
