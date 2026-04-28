# app/pipeline/table_builder.py
#
# FIX 3: Removed hard .head(20) cap from pivot tables.
#         Tables now return all rows by default.
#         A configurable TABLE_MAX_ROWS constant is provided (default 500)
#         so operators can tune it without changing logic.

import logging
logger = logging.getLogger(__name__)
import pandas as pd

# ─────────────────────────────────────────────
# 6. TABLE BUILDER
# ─────────────────────────────────────────────

# FIX 3: was previously hard-coded as .head(20) inside _build_one.
# Increase this constant or set to None to remove the limit entirely.
TABLE_MAX_ROWS: int | None = 500   # None = no limit


class TableBuilder:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def build_all(self, table_configs: list) -> list:
        results = []
        for cfg in table_configs:
            if isinstance(cfg, dict):
                results.append(cfg)
            elif (r := self._build_one(cfg)):
                results.append(r)
        return results

    def _build_one(self, cfg) -> dict | None:
        try:
            df    = self._df
            title = cfg.title
            t     = cfg.type
            vals  = cfg.values
            agg   = cfg.aggregation

            if vals not in df.columns:
                logger.warning("Table skipped: values column %r not in DataFrame", vals)
                return None

            if t == "pivot":
                idx  = cfg.index
                cols = cfg.columns

                if idx not in df.columns:
                    logger.warning("Table skipped: index column %r not in DataFrame", idx)
                    return None
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
                        observed=True,
                    )
                    .fillna(0)
                    .reset_index()
                )
                # FIX 3: apply configurable cap instead of hard head(20)
                if TABLE_MAX_ROWS is not None:
                    pivot = pivot.head(TABLE_MAX_ROWS)

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
