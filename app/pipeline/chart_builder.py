# FIX: Added rendering support for new chart types:
#      area, stacked_bar, grouped_bar, heatmap, bubble, funnel, treemap, waterfall

import logging
import math
import numpy as np
import pandas as pd
from app.pipeline.transformer import DataTransformer
from app.utils.column_utils import _meaningful_numeric_cols
from app.utils.column_utils import _is_id_col

logger = logging.getLogger(__name__)

#Backend does ALL statistical work before sending to the frontend:
#
#   [1] OUTLIER DETECTION & AXIS RANGE
#       IQR fence (Q1 - 3×IQR, Q3 + 3×IQR) defines the "core" range
#       that the x-axis is zoomed to. Outliers still exist in the data
#       (Plotly counts them) but don't stretch the axis. This directly
#       fixes the "Profit Distribution" screenshot — the -4000→+5000
#       axis range with 2 visible bars is caused by extreme outliers.
#
#   [2] FREEDMAN-DIACONIS BIN WIDTH
#       bin_width = 2 × IQR × n^(-1/3)
#       This is the gold-standard data-driven bin selector. It adapts
#       to both the spread of the data and the sample size. Plotly's
#       xbins.size is set to this width so bins are always meaningful.
#       Fallback: if IQR = 0 (constant column), use Sturges' rule.
#
#   [3] STATISTICAL CONTEXT PAYLOAD
#       Backend computes and sends: mean, median, std, skewness,
#       outlier_count, outlier_pct. Frontend uses these to render:
#         - Mean line  (dashed, labeled)
#         - Median line (dotted, labeled)
#         - Insight text (skew direction, outlier warning)
#       No computation happens in the browser.
#
def _compute_histogram_stats(series: pd.Series) -> dict:
    """
    Compute all statistical context needed for a well-formed histogram.
 
    Returns a dict with:
      x_range   — [lo, hi] axis zoom range (IQR fence), or None if no outliers
      bin_size  — Freedman-Diaconis bin width (falls back to Sturges)
      stats     — mean, median, std, skewness, outlier info, quartiles
    """
    vals = series.dropna().astype(float)
    n    = len(vals)
 
    if n == 0:
        return {"x_range": None, "bin_size": None, "stats": {}}
 
    arr = vals.to_numpy()
 
    # ── Quartiles & IQR ──────────────────────────────────────────────────────
    q1  = float(np.percentile(arr, 25))
    q3  = float(np.percentile(arr, 75))
    iqr = q3 - q1
 
    # ── Outlier detection (3×IQR fence — wider than Tukey's 1.5× to be lenient) ─
    # Using 3× keeps the fence wide enough to not over-clip genuine heavy-tail
    # distributions like sales revenue, while still excluding true extremes.
    if iqr > 0:
        lo_fence = q1 - 3.0 * iqr
        hi_fence = q3 + 3.0 * iqr
        outlier_mask  = (arr < lo_fence) | (arr > hi_fence)
        outlier_count = int(outlier_mask.sum())
        outlier_pct   = round(outlier_count / n * 100, 1)
 
        # Only apply axis zoom if outliers actually exist and distort the range
        if outlier_count > 0:
            # Pad the fence by 5% of the core range for visual breathing room
            core_range = hi_fence - lo_fence
            pad = core_range * 0.05
            x_range = [round(lo_fence - pad, 2), round(hi_fence + pad, 2)]
        else:
            x_range = None
    else:
        # IQR = 0: all values are identical (or nearly so) — no outlier concept
        lo_fence = float(arr.min())
        hi_fence = float(arr.max())
        outlier_count = 0
        outlier_pct   = 0.0
        x_range = None
 
    # ── Freedman-Diaconis bin width ───────────────────────────────────────────
    # bin_width = 2 × IQR × n^(-1/3)
    # This adapts to both data spread and sample size. It's the standard
    # data-driven bin width used by numpy, matplotlib, and seaborn by default.
    if iqr > 0 and n > 1:
        fd_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        # Clamp: never fewer than 5 bins or more than 150 bins in the core range
        core_span = hi_fence - lo_fence
        if fd_width > 0:
            n_bins = core_span / fd_width
            if n_bins < 5:
                fd_width = core_span / 5.0
            elif n_bins > 150:
                fd_width = core_span / 150.0
        bin_size = round(fd_width, 4) if fd_width > 0 else None
    else:
        # Sturges' rule fallback for zero-IQR or tiny datasets
        n_bins_sturges = max(5, int(math.ceil(1 + math.log2(n))) if n > 1 else 5)
        data_range = float(arr.max() - arr.min())
        bin_size = round(data_range / n_bins_sturges, 4) if data_range > 0 else None
 
    # ── Summary stats ─────────────────────────────────────────────────────────
    mean   = float(np.mean(arr))
    median = float(np.median(arr))
    std    = float(np.std(arr, ddof=1)) if n > 1 else 0.0
 
    # Pearson's moment skewness
    if std > 0:
        skewness = float(np.mean(((arr - mean) / std) ** 3))
    else:
        skewness = 0.0
 
    return {
        "x_range": x_range,
        "bin_size": bin_size,
        "stats": {
            "mean":          round(mean, 4),
            "median":        round(median, 4),
            "std":           round(std, 4),
            "skewness":      round(skewness, 4),
            "outlier_count": outlier_count,
            "outlier_pct":   outlier_pct,
            "q1":            round(q1, 4),
            "q3":            round(q3, 4),
            "iqr":           round(iqr, 4),
        },
    }

def _iqr_y_range(values_flat: list, fence_multiplier: float = 3.0) -> list | None:
    """
    Compute a y-axis zoom range based on IQR fencing.
 
    Uses `fence_multiplier` × IQR (default 3.0, same as histogram fix).
    Returns [lo, hi] with 5% padding when outliers are detected, else None.
 
    Returns None (no range override) when:
      - fewer than 4 values
      - IQR is 0 (all values identical)
      - no values fall outside the fence (no distortion present)
    """
    arr = np.array([v for v in values_flat if v is not None and not math.isnan(v)], dtype=float)
    if len(arr) < 4:
        return None
 
    q1  = float(np.percentile(arr, 25))
    q3  = float(np.percentile(arr, 75))
    iqr = q3 - q1
 
    if iqr <= 0:
        return None
 
    lo_fence = q1 - fence_multiplier * iqr
    hi_fence = q3 + fence_multiplier * iqr
 
    has_outliers = bool(np.any(arr < lo_fence) or np.any(arr > hi_fence))
    if not has_outliers:
        return None
 
    pad = (hi_fence - lo_fence) * 0.05
    return [round(lo_fence - pad, 4), round(hi_fence + pad, 4)]
 
 
def _box_stats(values_flat: list) -> dict:
    """Compute summary stats for the box InsightBadge."""
    arr = np.array([v for v in values_flat if v is not None and not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {}
    q1      = float(np.percentile(arr, 25))
    q3      = float(np.percentile(arr, 75))
    iqr     = q3 - q1
    median  = float(np.median(arr))
    lo_fence = q1 - 1.5 * iqr   # standard Tukey fence for stat reporting
    hi_fence = q3 + 1.5 * iqr
    outliers = arr[(arr < lo_fence) | (arr > hi_fence)]
    return {
        "median":        round(median, 4),
        "q1":            round(q1, 4),
        "q3":            round(q3, 4),
        "iqr":           round(iqr, 4),
        "outlier_count": int(len(outliers)),
        "outlier_pct":   round(len(outliers) / len(arr) * 100, 1),
    }

# ─────────────────────────────────────────────
# 5. CHART BUILDER
# ─────────────────────────────────────────────

class ChartBuilder:
    def __init__(self, transformer: DataTransformer):
        self._transformer = transformer

    def build(self, cfg: dict) -> dict | None:
        try:
            df = self._transformer.transform(cfg)
            if df.empty:
                return None

            chart_type = cfg["type"]
            x          = cfg["x"]
            y          = cfg["y"]
            color      = cfg.get("color")

            # ── box ──────────────────────────────────────────────────────────
            if chart_type == "box":
                # Check if x is a categorical column (distinct from y/numeric)
                x_is_categorical = (
                    x != y
                    and x in df.columns
                    and not pd.api.types.is_numeric_dtype(df[x])
                )
 
                if x_is_categorical and x in df.columns:
                    # ── Grouped box: one trace per category ──────────────────
                    groups = []
                    all_values = []
                    for cat, grp in df.groupby(x, observed=True):
                        vals = grp[y].dropna().tolist()
                        if len(vals) >= 4:   # skip trivially small groups
                            groups.append({"name": str(cat), "values": vals})
                            all_values.extend(vals)
 
                    if not groups:
                        # Fallback: treat as ungrouped if all groups were too small
                        all_values = df[y].dropna().tolist()
                        groups = [{"name": str(x), "values": all_values}]
                else:
                    # ── Single-column box ─────────────────────────────────────
                    # x == y (LLM set both to the numeric column), no category split
                    all_values = df[y].dropna().tolist()
                    groups = [{"name": str(y), "values": all_values}]
 
                return {
                    "type":        "box",
                    "title":       cfg["title"],
                    "groups":      groups,          # ← NEW: list of {name, values}
                    "x_label":     x,
                    "y_label":     y,
                    "layout_size": cfg["layout_size"],
                    # ── new scale/stats fields ───────────────────────────────
                    "y_range":     _iqr_y_range(all_values),   # None if no distortion
                    "stats":       _box_stats(all_values),
                    # ── legacy field for backward compat (kept for safety) ───
                    "values":      all_values,
                }

             # ── histogram ────────────────────────────────────────────────────
            # FIXED: compute all stats on the backend so the frontend can render
            # a properly scaled, binned, and annotated histogram without any
            # client-side number-crunching.
            if chart_type == "histogram":
                raw_values = df[x].dropna()
                hist_stats = _compute_histogram_stats(raw_values)
 
                return {
                    "type":        "histogram",
                    "title":       cfg["title"],
                    "values":      raw_values.tolist(),   # all values (outliers included)
                    "x_label":     x,
                    "layout_size": cfg["layout_size"],
                    # ── new fields ──────────────────
                    "x_range":     hist_stats["x_range"],   # [lo, hi] or None
                    "bin_size":    hist_stats["bin_size"],   # FD bin width or None
                    "stats":       hist_stats["stats"],      # mean, median, std, skew, ...
                }

            # ── heatmap ──────────────────────────────────────────────────────
            if chart_type == "heatmap":
                z_col = cfg.get("z")

                # Step 1: basic validation
                if not z_col or z_col not in df.columns:
                    return None

                # Step 2: create safe df FIRST
                safe_df = df[[x, y, z_col]].copy()

                # Force numeric BEFORE ANY CHECK
                safe_df[z_col] = pd.to_numeric(safe_df[z_col], errors="coerce")

                # Drop bad rows
                safe_df = safe_df.dropna(subset=[z_col])

                if safe_df.empty:
                    logger.error("Heatmap failed: z column invalid after conversion")
                    return None

                logger.info("Heatmap dtype AFTER FIX: %s", safe_df[z_col].dtype)

                # Step 3: Limit dimensions to keep the grid readable.
                # Cap at 20 rows × 20 columns — beyond this the cells become too
                # small to read and color differences become imperceptible.
                MAX_DIM = 20
                top_x_vals = (
                    safe_df.groupby(x)[z_col].sum()
                    .nlargest(MAX_DIM).index.tolist()
                )
                top_y_vals = (
                    safe_df.groupby(y)[z_col].sum()
                    .nlargest(MAX_DIM).index.tolist()
                )
                safe_df = safe_df[
                    safe_df[x].isin(top_x_vals) & safe_df[y].isin(top_y_vals)
                ]

                # Step 4: pivot
                try:
                    pivot = safe_df.pivot_table(
                        index=y,
                        columns=x,
                        values=z_col,
                        aggfunc="mean"
                    )
                except Exception as exc:
                    logger.error("Heatmap pivot failed: %s", exc)
                    return None

                # Step 5: Sort rows and columns by their totals (highest → lowest)
                # so the most significant patterns appear in the top-left corner.
                row_order = pivot.sum(axis=1).sort_values(ascending=False).index
                col_order = pivot.sum(axis=0).sort_values(ascending=False).index
                pivot = pivot.loc[row_order, col_order]

                # Step 6: Detect scale distortion.
                # If the max value is more than 10× the median, a single cell will
                # dominate the color scale and wash out everything else.
                # In that case, apply log1p normalization and flag it so the
                # frontend can label the color bar correctly.
                flat_vals = pivot.values.flatten()
                flat_vals = flat_vals[~np.isnan(flat_vals)]
                use_log_scale = False
                if len(flat_vals) > 1:
                    med = float(np.median(flat_vals[flat_vals > 0])) if np.any(flat_vals > 0) else 0
                    max_val = float(np.max(flat_vals))
                    if med > 0 and max_val / med > 10:
                        use_log_scale = True
                        pivot = pivot.apply(lambda col: np.log1p(col.clip(lower=0)))

                z_values = pivot.values.tolist()

                # Step 7: Annotate cells when the grid is small enough to be readable.
                # For grids up to 10×10, pass the raw (un-logged) cell values for
                # display inside each cell. The frontend shows them as formatted numbers.
                n_rows, n_cols = pivot.shape
                cell_annotations: list | None = None
                if n_rows <= 10 and n_cols <= 10:
                    # Re-pivot original (non-log) values for the text layer
                    raw_pivot = safe_df.pivot_table(
                        index=y, columns=x, values=z_col, aggfunc="mean"
                    ).loc[row_order[:n_rows], col_order[:n_cols]]
                    cell_annotations = raw_pivot.values.tolist()

                return {
                    "type":             "heatmap",
                    "title":            cfg["title"],
                    "x":                pivot.columns.tolist(),
                    "y":                pivot.index.tolist(),
                    "z":                z_values,
                    "cell_annotations": cell_annotations,  # None when grid is large
                    "use_log_scale":    use_log_scale,      # True → color bar is log1p
                    "x_label":          x,
                    "y_label":          y,
                    "z_label":          z_col,
                    "layout_size":      cfg["layout_size"],
                }
            # ── bubble ───────────────────────────────────────────────────────
            # Size distortion handled client-side via min-max normalization
            if chart_type == "bubble":
                size_col = cfg.get("size")
                sizes = df[size_col].tolist() if size_col and size_col in df.columns else None
                return {
                    "type": "bubble", "title": cfg["title"],
                    "x": df[x].tolist(), "y": df[y].tolist(),
                    "sizes": sizes,
                    "x_label": x, "y_label": y,
                    "size_label": size_col,
                    "layout_size": cfg["layout_size"],
                }

            # ── funnel ───────────────────────────────────────────────────────
            if chart_type == "funnel":
                # ── Semantic pre-flight: detect bad column pairings BEFORE any
                # data work, even if the LLM produced the config.
                #
                # A funnel x-axis must be a genuine stage/step label column.
                # Date columns, geographic columns, and high-cardinality IDs are
                # never valid funnel stages. When a bad pairing is detected, we
                # redirect to the semantically correct chart type immediately —
                # no "not_possible" dead end, and no misleading fallback bar.

                _DATE_HINTS  = ("date", "time", "month", "year", "day", "week",
                                "period", "created", "updated", "order_date", "ship")
                _GEO_HINTS   = ("region", "country", "state", "city", "zip",
                                "postal", "location", "territory", "market")
                _STAGE_HINTS = ("stage", "step", "phase", "funnel", "pipeline",
                                "status", "level", "tier", "bucket", "segment")

                x_lower = x.lower()

                def _is_date_col(col: str) -> bool:
                    """True if col looks like a date/time axis."""
                    if pd.api.types.is_datetime64_any_dtype(self._transformer._source[col]):
                        return True
                    col_l = col.lower()
                    return any(h in col_l for h in _DATE_HINTS)

                def _is_geo_col(col: str) -> bool:
                    col_l = col.lower()
                    return any(h in col_l for h in _GEO_HINTS)

                def _has_stage_hints(col: str) -> bool:
                    col_l = col.lower()
                    return any(h in col_l for h in _STAGE_HINTS)

                # Check 1: date column as x → redirect to line chart
                if _is_date_col(x):
                    logger.warning(
                        "Funnel rejected: x='%s' is a date/time column, not a stage label. "
                        "Redirecting to line chart.", x
                    )
                    # Build a proper line chart payload directly
                    line_df = df[[x, y]].copy()
                    line_df[y] = pd.to_numeric(line_df[y], errors="coerce")
                    line_df = line_df.dropna(subset=[y])
                    return {
                        "type":          "line",
                        "fallback_from": "funnel",
                        "fallback_reason": f"'{x}' is a time column — showing trend instead",
                        "title":         cfg["title"].replace("Funnel", "Trend").replace("funnel", "Trend"),
                        "x":             line_df[x].astype(str).tolist(),
                        "y":             line_df[y].tolist(),
                        "x_label":       x,
                        "y_label":       y,
                        "layout_size":   "large",
                        "y_range":       None,
                    }

                # Check 2: geographic / regional column as x → redirect to bar
                if _is_geo_col(x) and not _has_stage_hints(x):
                    logger.warning(
                        "Funnel rejected: x='%s' is a geographic column, not a stage label. "
                        "Redirecting to bar chart.", x
                    )
                    bar_df = df[[x, y]].copy()
                    bar_df[y] = pd.to_numeric(bar_df[y], errors="coerce")
                    bar_df = bar_df.dropna(subset=[y]).nlargest(20, y)
                    return {
                        "type":          "bar",
                        "fallback_from": "funnel",
                        "fallback_reason": f"'{x}' is a geographic column — showing bar comparison instead",
                        "title":         cfg["title"],
                        "x":             bar_df[x].tolist(),
                        "y":             bar_df[y].tolist(),
                        "x_label":       x,
                        "y_label":       y,
                        "layout_size":   cfg["layout_size"],
                        "y_range":       None,
                    }

                # Check 3: high-cardinality x with no stage hints → redirect to bar
                x_unique = self._transformer._source[x].nunique()
                if x_unique > 10 and not _has_stage_hints(x):
                    logger.warning(
                        "Funnel rejected: x='%s' has %d unique values and no stage-name hints. "
                        "Redirecting to bar chart.", x, x_unique
                    )
                    bar_df = df[[x, y]].copy()
                    bar_df[y] = pd.to_numeric(bar_df[y], errors="coerce")
                    bar_df = bar_df.dropna(subset=[y]).nlargest(20, y)
                    return {
                        "type":          "bar",
                        "fallback_from": "funnel",
                        "fallback_reason": f"'{x}' has too many unique values for a funnel — showing top categories instead",
                        "title":         cfg["title"],
                        "x":             bar_df[x].tolist(),
                        "y":             bar_df[y].tolist(),
                        "x_label":       x,
                        "y_label":       y,
                        "layout_size":   cfg["layout_size"],
                        "y_range":       None,
                    }

                # Helper: build a bar-chart fallback payload so the user still
                # gets a useful visualisation with a clear "why no funnel" note.
                def _funnel_fallback(reason: str) -> dict:
                    logger.warning("Funnel → bar fallback: %s", reason)
                    bar_df = df[[x, y]].copy()
                    bar_df[y] = pd.to_numeric(bar_df[y], errors="coerce")
                    bar_df = bar_df.dropna(subset=[y]).nlargest(20, y)
                    return {
                        "type":          "bar",
                        "fallback_from": "funnel",
                        "fallback_reason": reason,
                        "title":         cfg["title"],
                        "x":             bar_df[x].tolist(),
                        "y":             bar_df[y].tolist(),
                        "x_label":       x,
                        "y_label":       y,
                        "layout_size":   cfg["layout_size"],
                        "y_range":       None,
                    }

                # FIX 5: Validate y is numeric before doing anything else.
                if not pd.api.types.is_numeric_dtype(df[y]):
                    coerced = pd.to_numeric(df[y], errors="coerce")
                    if coerced.notna().sum() / max(len(df), 1) < 0.5:
                        return _funnel_fallback("y column is not numeric")
                    df = df.copy()
                    df[y] = coerced

                stages = df[x].tolist()
                values = pd.to_numeric(df[y], errors="coerce").tolist()
                n_steps = len(stages)

                if n_steps < 2:
                    return _funnel_fallback("fewer than 2 stages")

                # FIX 1: Auto-sort descending so the user doesn't need to
                # pre-order their data — the funnel shape will always be correct.
                pairs  = sorted(
                    zip(stages, values),
                    key=lambda t: (t[1] is None or math.isnan(t[1]), -(t[1] or 0)),
                )
                # pairs is ascending; reverse for descending (largest first)
                pairs  = list(reversed(pairs))
                stages = [p[0] for p in pairs]
                values = [p[1] for p in pairs]

                # After sorting, cap at 6 stages (ideal funnel range).
                if len(stages) > 6:
                    logger.warning("Funnel: truncating %d stages to 6.", len(stages))
                    stages = stages[:6]
                    values = values[:6]

                non_null = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]

                # FIX 2: Relaxed monotonic check — allow up to 30% of transitions
                # to be out-of-order (noise tolerance) rather than a hard limit of 1.
                # Only reject when the data is clearly non-funnel (majority non-decreasing).
                inversions = sum(
                    1 for i in range(len(non_null) - 1)
                    if non_null[i] < non_null[i + 1]
                )
                if inversions > len(non_null) * 0.3:
                    return _funnel_fallback(
                        f"values are not monotonically decreasing "
                        f"({inversions} inversions out of {len(non_null) - 1} transitions)"
                    )

                # Compute per-step conversion rates relative to the previous step.
                conversion_pcts: list[float | None] = [None]
                for i in range(1, len(non_null)):
                    prev = non_null[i - 1]
                    curr = non_null[i]
                    pct  = round((curr / prev) * 100, 1) if prev else None
                    conversion_pcts.append(pct)

                # Identify the single biggest drop-off stage index.
                drops = [
                    (i, (non_null[i - 1] - non_null[i]) / non_null[i - 1])
                    for i in range(1, len(non_null))
                    if non_null[i - 1]
                ]
                biggest_drop_idx = max(drops, key=lambda t: t[1])[0] if drops else None

                return {
                    "type":              "funnel",
                    "title":             cfg["title"],
                    "x":                 stages,
                    "y":                 values,
                    "conversion_pcts":   conversion_pcts,
                    "biggest_drop_idx":  biggest_drop_idx,
                    "x_label":           x,
                    "y_label":           y,
                    "layout_size":       cfg["layout_size"],
                }

            # ── treemap ──────────────────────────────────────────────────────
            if chart_type == "treemap":
                return {
                    "type": "treemap", "title": cfg["title"],
                    "labels": df[x].tolist(), "values": df[y].tolist(),
                    "x_label": x, "y_label": y,
                    "layout_size": cfg["layout_size"],
                }

            # ── waterfall ────────────────────────────────────────────────────
            if chart_type == "waterfall":
                return {
                    "type": "waterfall", "title": cfg["title"],
                    "x": df[x].tolist(), "y": df[y].tolist(),
                    "x_label": x, "y_label": y,
                    "layout_size": cfg["layout_size"],
                }

            # ── multi-series types (line, area, bar, stacked_bar, grouped_bar) ──
             # FIXED: compute y_range for scale distortion prevention
            if color and color in df.columns:
                payload = self._multi_series(df, cfg)
                # Flatten all y values across all series to compute a single range
                all_y = [v for s in payload["series"] for v in s["y"]
                         if v is not None and not math.isnan(float(v))]
                payload["y_range"] = _iqr_y_range(all_y)
                return payload
 
            # ── single-series fallthrough ─────────────────────────────────────
            y_vals = df[y].tolist()
            y_range = _iqr_y_range(y_vals) if chart_type not in ("pie",) else None
 
            return {
                "type":        chart_type,
                "title":       cfg["title"],
                "x":           df[x].tolist(),
                "y":           y_vals,
                "x_label":     x,
                "y_label":     y,
                "layout_size": cfg["layout_size"],
                "y_range":     y_range,   # None if no distortion detected
            }

        except Exception as exc:
            logger.error("Chart build failed for %r: %s", cfg.get("title"), exc)
            return None

    @staticmethod
    def _multi_series(df: pd.DataFrame, cfg: dict) -> dict:
        x, y, color = cfg["x"], cfg["y"], cfg["color"]
        return {
            "type": cfg["type"], "title": cfg["title"],
            "series": [
                {"name": str(n), "x": g[x].tolist(), "y": g[y].tolist()}
                for n, g in df.groupby(color)
            ],
            "x_label": x, "y_label": y,
            "layout_size": cfg["layout_size"],
        }