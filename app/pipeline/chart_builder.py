import logging
logger = logging.getLogger(__name__)
import pandas as pd
from app.pipeline.transformer import DataTransformer

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
            x, y       = cfg["x"], cfg["y"]
            color      = cfg["color"]

            if chart_type == "histogram":
                return {
                    "type": "histogram", "title": cfg["title"],
                    "values": df[x].tolist(), "x_label": x,
                    "layout_size": cfg["layout_size"],
                }

            if color and color in df.columns:
                return self._multi_series(df, cfg)

            if chart_type == "scatter" and (df[x].nunique() < 5 or df[y].nunique() < 5):
                return None

            return {
                "type": chart_type, "title": cfg["title"],
                "x": df[x].tolist(), "y": df[y].tolist(),
                "x_label": x, "y_label": y,
                "layout_size": cfg["layout_size"],
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
