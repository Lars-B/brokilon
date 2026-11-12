import copy
import html
import json
from collections import Counter

import numpy as np
from tqdm import tqdm

from brokilon.ccd.domain.sranges import sranges


def smallest_common_superclade(s, clades, rare):
    """All smallest non-rare clades in the input clades whose taxa include all of s's taxa."""
    candidates = [
        c for c in clades
        if c not in rare and c != s and s.clade <= c.clade
    ]
    if not candidates:
        return []
    smallest = min(len(c.clade) for c in candidates)
    return [c for c in candidates if len(c.clade) == smallest]


def flag_outliers(scores, z_cutoff=1.0):
    """Indices of trees whose log score is more than z_cutoff robust deviations above
    the median, with the median and MAD taken from trees scoring above 0."""
    scores = np.asarray(scores, dtype=float)
    log_scores = np.log1p(scores)
    positive = log_scores[scores > 0]
    print(f"{len(positive)} of {len(scores)} trees have a score above 0")

    if len(positive) == 0:
        print("No tree has a score above 0, no trees flagged.")
        return np.array([], dtype=int)

    med = np.median(positive)
    mad = np.median(np.abs(positive - med)) * 1.4826
    if mad == 0:
        # Most scored trees share one value, so flag only the ones above it
        flagged = np.where(log_scores > med)[0]
    else:
        robust_z = (log_scores - med) / mad
        flagged = np.where(robust_z > z_cutoff)[0]

    print(f"{len(flagged)} of {len(scores)} trees flagged (z > {z_cutoff})")
    return flagged


def prune_outlier_trees(trees, taxon_map, k=1, z_cutoff=1.0):
    # Clades of each tree, each from its own fresh copy
    per_tree_clades = []
    for tree in tqdm(trees, desc="Collecting clades", unit="tree"):
        counts, _, _, _ = sranges.get_sranges_map([copy.deepcopy(tree)], taxon_map)
        per_tree_clades.append(set(counts))

    # Count across trees by combining the per-tree results
    clade_counts = Counter()
    for clades in per_tree_clades:
        clade_counts.update(clades)

    rare = {c for c, n in clade_counts.items() if n <= k}
    non_rare = [c for c in clade_counts if c not in rare]
    sup_cache = {
        s: smallest_common_superclade(s, non_rare, rare)
        for s in tqdm(rare, desc="Finding superclades", unit="clade")
    }

    # For each rare clade: (rare clade size, superclade size,
    #                       rare clade count, superclade count)
    rows = []
    for tc in per_tree_clades:
        tree_rows = []
        for s in tc:
            if s not in rare:
                continue
            # there can be multiple same sized superclades
            sups = sup_cache[s]
            tree_rows.append((
                len(s.clade),
                len(sups[0].clade) if sups else None,
                clade_counts[s],
                sum(clade_counts[c] for c in sups) if sups else None,
            ))
        rows.append(tree_rows)

    scores = []
    for tree_rows in rows:
        size_score = 0.0
        count_score = 0.0
        weighted_count_score = 0.0
        for len_s, len_sup, count_s, count_sup in tree_rows:
            if len_sup is None:
                continue
            size_score += len_sup / len_s
            count_score += count_sup / count_s
            weighted_count_score += (len_sup - len_s) * count_sup / count_s
        scores.append((size_score, count_score, weighted_count_score))

    weighted = [s[2] for s in scores]
    flagged = set(flag_outliers(weighted, z_cutoff=z_cutoff).tolist())

    kept = [
        copy.deepcopy(tree)
        for i, tree in enumerate(trees)
        if i not in flagged
    ]
    # print(f"Kept {len(kept)} of {len(trees)} trees.")
    return kept


def _clade_label(s):
    """Raw contents of s.clade, sorted so the same clade always reads the same."""
    try:
        items = sorted(s.clade)
    except TypeError:
        items = sorted(s.clade, key=str)
    return "({" + ", ".join(str(x) for x in items) + "}," + f"{s.ancestral_range})"


def report_outlier_trees(trees, taxon_map, out_path="outlier_report.html", k=1, z_cutoff=1.0):
    """Same scoring as prune_outlier_trees, but writes an HTML report instead of pruning.

    Table 1: one row per tree (its three scores, robust z, flagged or not).
    Table 2: one row per clade per tree. Every clade is listed; superclade and
    ratio columns are only filled in for rare clades (count <= k) that have a
    non-rare superclade, since only those contribute to the tree scores.
    """
    # Clades of each tree, each from its own fresh copy
    per_tree_clades = []
    for tree in tqdm(trees, desc="Collecting clades", unit="tree"):
        counts, _, _, _ = sranges.get_sranges_map([copy.deepcopy(tree)], taxon_map)
        per_tree_clades.append(set(counts))

    # Count across trees by combining the per-tree results
    clade_counts = Counter()
    for clades in per_tree_clades:
        clade_counts.update(clades)

    rare = {c for c, n in clade_counts.items() if n <= k}
    non_rare = [c for c in clade_counts if c not in rare]
    sup_cache = {
        s: smallest_common_superclade(s, non_rare, rare)
        for s in tqdm(rare, desc="Finding superclades", unit="clade")
    }

    clade_rows = []
    tree_scores = []
    for i, tc in enumerate(tqdm(per_tree_clades, desc="Scoring clades", unit="tree")):
        size_score = 0.0
        count_score = 0.0
        weighted_count_score = 0.0
        n_rare = 0
        n_scored = 0
        rows_this_tree = []
        for s in tc:
            len_s = len(s.clade)
            count_s = clade_counts[s]
            is_rare = s in rare
            if not is_rare:
                continue
            sup_label = len_sup = count_sup = n_sups = None
            size_ratio = count_ratio = weighted = None

            n_rare += 1
            # can have multiple superclades of same size
            sups = sup_cache[s]
            if sups:
                n_scored += 1
                n_sups = len(sups)  # keeping track of how many superclades
                sup_label = " | ".join(sorted(_clade_label(c) for c in sups))
                len_sup = len(sups[0].clade)
                count_sup = sum(clade_counts[c] for c in sups)
                # Above is accounting for multiple super clades possibility
                size_ratio = len_sup / len_s
                count_ratio = count_sup / count_s
                weighted = (len_sup - len_s) * count_sup / count_s
                size_score += size_ratio
                count_score += count_ratio
                weighted_count_score += weighted
            rows_this_tree.append([
                i, _clade_label(s), len_s, count_s, "yes" if is_rare else "no",
                sup_label, n_sups, len_sup, count_sup, size_ratio, count_ratio, weighted,
            ])
        rows_this_tree.sort(key=lambda r: (r[2], r[1]))
        clade_rows.extend(rows_this_tree)
        tree_scores.append((len(tc), n_rare, n_scored,
                            size_score, count_score, weighted_count_score))

    # Flagging exactly as in prune_outlier_trees
    weighted_scores = np.asarray([t[5] for t in tree_scores], dtype=float)
    flagged = set(flag_outliers(weighted_scores, z_cutoff=z_cutoff).tolist())

    # Robust z per tree, same median/MAD as flag_outliers (blank when MAD is 0)
    log_scores = np.log1p(weighted_scores)
    positive = log_scores[weighted_scores > 0]
    robust_z = [None] * len(trees)
    if len(positive) > 0:
        med = np.median(positive)
        mad = np.median(np.abs(positive - med)) * 1.4826
        if mad != 0:
            robust_z = ((log_scores - med) / mad).tolist()

    tree_rows = []
    for i, (n_clades, n_rare, n_scored, ss, cs, ws) in enumerate(tree_scores):
        tree_rows.append([
            i, n_clades, n_rare, n_scored, ss, cs, ws,
            float(log_scores[i]), robust_z[i], "yes" if i in flagged else "no",
        ])

    tree_columns = [
        {"label": "Tree", "type": "num", "width": 70},
        {"label": "Clades", "type": "num", "width": 80},
        {"label": "Rare clades", "type": "num", "width": 100},
        {"label": "Rare w/ superclade", "type": "num", "width": 140},
        {"label": "Size score", "type": "num", "width": 110},
        {"label": "Count score", "type": "num", "width": 110},
        {"label": "Weighted count score", "type": "num", "width": 160},
        {"label": "log1p(weighted)", "type": "num", "width": 130},
        {"label": "Robust z", "type": "num", "width": 100},
        {"label": "Flagged", "type": "str", "width": 80},
    ]
    clade_columns = [
        {"label": "Tree", "type": "num", "width": 70},
        {"label": "Clade", "type": "str", "width": 320},
        {"label": "Size", "type": "num", "width": 70},
        {"label": "Count", "type": "num", "width": 80},
        {"label": "Rare", "type": "str", "width": 70},
        {"label": "Superclade", "type": "str", "width": 320},
        {"label": "# superclades", "type": "num", "width": 110},
        {"label": "Sup. size", "type": "num", "width": 90},
        {"label": "Sup. count", "type": "num", "width": 90},
        {"label": "Size ratio", "type": "num", "width": 100},
        {"label": "Count ratio", "type": "num", "width": 100},
        {"label": "Weighted contrib.", "type": "num", "width": 130},
    ]

    data = {
        "meta": {
            "n_trees": len(trees),
            "n_flagged": len(flagged),
            "n_unique_clades": len(clade_counts),
            "n_rare_unique": len(rare),
            "k": k,
            "z_cutoff": z_cutoff,
        },
        "trees": {"columns": tree_columns, "rows": tree_rows},
        "clades": {"columns": clade_columns, "rows": clade_rows},
    }
    # "</" escaped so clade text can never close the <script> tag early
    payload = json.dumps(data, default=str).replace("</", "<\\/")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_REPORT_TEMPLATE.replace("__DATA__", payload)
                .replace("__TITLE__", html.escape(f"Outlier tree report ({len(trees)} trees)")))
    print(f"Report written to {out_path} "
          f"({len(tree_rows)} tree rows, {len(clade_rows)} clade rows)")

_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; color: #1d1d1f; background: #fafafa; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  h2 { font-size: 16px; margin: 26px 0 6px; }
  .meta { color: #555; font-size: 13px; margin-bottom: 8px; }
  .hint { color: #777; font-size: 12px; margin: 2px 0 6px; }
  .bar { display: flex; gap: 12px; align-items: center; font-size: 13px; margin-bottom: 6px; }
  .bar button { font-size: 12px; padding: 3px 8px; cursor: pointer; }
  .scroll { height: 60vh; overflow: auto; border: 1px solid #ccc; background: #fff; }
  table { border-collapse: collapse; table-layout: fixed; font-size: 12.5px; }
  th, td { border-bottom: 1px solid #eee; padding: 0 8px; height: 24px; line-height: 24px;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  thead th { position: sticky; background: #f0f0f2; z-index: 1; text-align: left; }
  thead tr.labels th { top: 0; cursor: pointer; user-select: none; border-bottom: 1px solid #ccc; }
  thead tr.filters th { top: 25px; padding: 2px 4px; }
  thead input { width: 100%; box-sizing: border-box; font-size: 12px; padding: 1px 4px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.flag td { background: #fdecea; }
  tr.rare td { background: #fff7e0; }
  #trees tbody tr.data { cursor: pointer; }
  #trees tbody tr.data:hover td { background: #e8f0fe; }
</style>
</head>
<body>
<h1>Outlier tree report</h1>
<div class="meta" id="meta"></div>

<h2>Trees</h2>
<div class="hint">Click a header to sort. Filters accept text, or for numbers: &gt;5, &lt;=2, =3, 1..10. Click a tree row to show its clades below.</div>
<div class="bar"><span id="trees-count"></span><button data-clear="trees">Clear filters</button></div>
<div class="scroll" id="trees-scroll"><table id="trees"></table></div>

<h2>Clades</h2>
<div class="hint">Hover a clade cell to see it in full. Rare clades are highlighted.</div>
<div class="bar"><span id="clades-count"></span><button data-clear="clades">Clear filters</button></div>
<div class="scroll" id="clades-scroll"><table id="clades"></table></div>

<script>
const DATA = __DATA__;
const ROW_H = 25, OVERSCAN = 20;

const m = DATA.meta;
document.getElementById("meta").textContent =
  `${m.n_trees} trees, ${m.n_flagged} flagged (z > ${m.z_cutoff}). ` +
  `${m.n_unique_clades} unique clades, ${m.n_rare_unique} rare (count <= ${m.k}).`;

function fmt(v, type) {
  if (v === null || v === undefined) return "";
  if (type === "num" && !Number.isInteger(v)) return v.toFixed(4);
  return String(v);
}

function makeFilter(text, type) {
  text = text.trim();
  if (!text) return null;
  if (type === "num") {
    let mm = text.match(/^(-?[\d.eE+-]+)\s*\.\.\s*(-?[\d.eE+-]+)$/);
    if (mm) { const a = +mm[1], b = +mm[2]; return v => v !== null && v >= a && v <= b; }
    mm = text.match(/^(<=|>=|<|>|=|==)\s*(-?[\d.eE+-]+)$/);
    if (mm) {
      const x = +mm[2];
      const ops = { "<": v => v < x, ">": v => v > x, "<=": v => v <= x, ">=": v => v >= x,
                    "=": v => v === x, "==": v => v === x };
      const f = ops[mm[1]];
      return v => v !== null && f(v);
    }
  }
  const terms = text.toLowerCase().split(";").map(s => s.trim()).filter(Boolean);
  return (v, type2) => {
    if (v === null) return false;
    const s = fmt(v, type2).toLowerCase();
    return terms.every(t => s.includes(t));
  };
}

function VTable(id, spec, opts) {
  const table = document.getElementById(id);
  const scroll = document.getElementById(id + "-scroll");
  const countEl = document.getElementById(id + "-count");
  const cols = spec.columns, all = spec.rows;
  let view = all.slice(), sortCol = null, sortDir = 1;
  const filters = cols.map(() => "");

  const colgroup = "<colgroup>" + cols.map(c => `<col style="width:${c.width}px">`).join("") + "</colgroup>";
  table.style.width = cols.reduce((s, c) => s + c.width, 0) + "px";
  table.innerHTML = colgroup +
    "<thead><tr class='labels'>" + cols.map((c, j) => `<th data-j="${j}">${c.label}</th>`).join("") + "</tr>" +
    "<tr class='filters'>" + cols.map((c, j) => `<th><input data-j="${j}" placeholder="filter"></th>`).join("") + "</tr></thead>" +
    "<tbody></tbody>";
  const tbody = table.querySelector("tbody");
  const inputs = table.querySelectorAll("thead input");

  function apply() {
    const fs = filters.map((t, j) => [j, makeFilter(t, cols[j].type)]).filter(p => p[1]);
    view = fs.length ? all.filter(r => fs.every(([j, f]) => f(r[j], cols[j].type))) : all.slice();
    if (sortCol !== null) {
      const j = sortCol;
      view.sort((a, b) => {
        const x = a[j], y = b[j];
        if (x === null && y === null) return 0;
        if (x === null) return 1;            // blanks always last
        if (y === null) return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
      });
    }
    table.querySelectorAll("thead tr.labels th").forEach((th, j) => {
      th.textContent = cols[j].label + (j === sortCol ? (sortDir > 0 ? " ▲" : " ▼") : "");
    });
    countEl.textContent = `Showing ${view.length.toLocaleString()} of ${all.length.toLocaleString()} rows`;
    scroll.scrollTop = 0;
    render();
  }

  function render() {
    const head = 2 * ROW_H;
    const start = Math.max(0, Math.floor((scroll.scrollTop - head) / ROW_H) - OVERSCAN);
    const visible = Math.ceil(scroll.clientHeight / ROW_H) + 2 * OVERSCAN;
    const end = Math.min(view.length, start + visible);
    let h = `<tr style="height:${start * ROW_H}px"></tr>`;
    for (let i = start; i < end; i++) {
      const r = view[i];
      const cls = opts.rowClass ? opts.rowClass(r) : "";
      h += `<tr class="data ${cls}" data-i="${i}">` + cols.map((c, j) => {
        const s = fmt(r[j], c.type).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
        return `<td class="${c.type}" title="${s}">${s}</td>`;
      }).join("") + "</tr>";
    }
    h += `<tr style="height:${(view.length - end) * ROW_H}px"></tr>`;
    tbody.innerHTML = h;
  }

  let raf = null;
  scroll.addEventListener("scroll", () => {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = null; render(); });
  });
  table.querySelectorAll("thead tr.labels th").forEach(th => th.addEventListener("click", () => {
    const j = +th.dataset.j;
    if (sortCol === j) sortDir = -sortDir; else { sortCol = j; sortDir = 1; }
    apply();
  }));
  let timer = null;
  inputs.forEach(inp => inp.addEventListener("input", () => {
    filters[+inp.dataset.j] = inp.value;
    clearTimeout(timer); timer = setTimeout(apply, 200);
  }));
  if (opts.onRowClick) tbody.addEventListener("click", e => {
    const tr = e.target.closest("tr.data");
    if (tr) opts.onRowClick(view[+tr.dataset.i]);
  });

  const api = {
    setFilter(j, text) { filters[j] = text; inputs[j].value = text; apply(); },
    clear() { filters.fill(""); inputs.forEach(i => i.value = ""); apply(); },
  };
  apply();
  return api;
}

const cladeTable = VTable("clades", DATA.clades, {
  rowClass: r => r[4] === "yes" ? "rare" : "",
});
const treeTable = VTable("trees", DATA.trees, {
  rowClass: r => r[9] === "yes" ? "flag" : "",
  onRowClick: r => {
    cladeTable.setFilter(0, "=" + r[0]);
    document.getElementById("clades-scroll").scrollIntoView({ behavior: "smooth" });
  },
});
document.querySelectorAll("button[data-clear]").forEach(b => b.addEventListener("click", () =>
  (b.dataset.clear === "trees" ? treeTable : cladeTable).clear()));
</script>
</body>
</html>
"""
