import logging
logger = logging.getLogger(__name__)
import pandas as pd

# ─────────────────────────────────────────────
# 6. TABLE BUILDER
# ─────────────────────────────────────────────

class TableBuilder:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def build_all(self, table_configs: list) -> list:
        """
        Accepts either list[TableConfigSchema] (from LLM path) or
        list[dict] (from _default_tables fallback path).
        Plain dicts are passed through as-is since they are already rendered.
        """
        results = []
        for cfg in table_configs:
            if isinstance(cfg, dict):
                # already a rendered table from _default_tables — pass through
                results.append(cfg)
            elif (r := self._build_one(cfg)):
                results.append(r)
        return results

    def _build_one(self, cfg) -> dict | None:
        """cfg is a TableConfigSchema instance."""
        try:
            df    = self._df
            title = cfg.title
            t     = cfg.type
            vals  = cfg.values
            agg   = cfg.aggregation

            # column guard — log and skip rather than raise
            if vals not in df.columns:
                logger.warning("Table skipped: values column %r not in DataFrame", vals)
                return None

            if t == "pivot":
                idx  = cfg.index
                cols = cfg.columns

                if idx not in df.columns:
                    logger.warning("Table skipped: index column %r not in DataFrame", idx)
                    return None
                # cols is optional for pivot — None means no column grouping
                if cols and cols not in df.columns:
                    logger.warning("Pivot: columns %r not found, building without it", cols)
                    cols = None

                pivot = (
                    pd.pivot_table(
                        df,
                        index=idx,
                        columns=cols if cols else None,
                        values=vals,
                        aggfunc=agg,
                        observed=True,       # silence pandas FutureWarning
                    )
                    .fillna(0)
                    .reset_index()
                    .head(20)
                )
                # flatten MultiIndex columns produced when cols is set
                pivot.columns = [
                    str(c) if not isinstance(c, tuple) else "_".join(str(x) for x in c if x)
                    for c in pivot.columns
                ]
                return {"title": title, "data": pivot.to_dict(orient="records")}

            if t == "summary":
                if not pd.api.types.is_numeric_dtype(df[vals]):
                    logger.warning("Summary skipped: %r is not numeric", vals)
                    return None
                return {"title": title, "data": [{vals: float(df[vals].agg(agg))}]}

        except Exception as exc:
            logger.error("Table build failed (%s / %s): %s", cfg.type, cfg.values, exc)

        return None
