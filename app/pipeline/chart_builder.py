# FIX: Added rendering support for new chart types:
#      area, stacked_bar, grouped_bar, heatmap, bubble, funnel, treemap, waterfall

import logging
import math
import numpy as np
import pandas as pd
from app.pipeline.transformer import DataTransformer
 
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
            # No scale distortion risk (color scale auto-normalizes)
            if chart_type == "heatmap":
                z_col = cfg.get("z") or y
                if z_col not in df.columns:
                    logger.warning("Heatmap: z column %r not found", z_col)
                    return None
                try:
                    pivot = (
                        pd.pivot_table(df, index=y, columns=x, values=z_col,
                                       aggfunc="mean", observed=True)
                        .fillna(0)
                    )
                    return {
                        "type": "heatmap", "title": cfg["title"],
                        "x": pivot.columns.tolist(),
                        "y": pivot.index.tolist(),
                        "z": pivot.values.tolist(),
                        "x_label": x, "y_label": y, "z_label": z_col,
                        "layout_size": cfg["layout_size"],
                    }
                except Exception as exc:
                    logger.error("Heatmap pivot failed: %s", exc)
                    return None

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
                return {
                    "type": "funnel", "title": cfg["title"],
                    "x": df[x].tolist(), "y": df[y].tolist(),
                    "x_label": x, "y_label": y,
                    "layout_size": cfg["layout_size"],
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
