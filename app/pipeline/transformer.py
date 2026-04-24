import pandas as pd

# ─────────────────────────────────────────────
# 4. DATA TRANSFORMER  (one aggregation block, minimal copies)
# ─────────────────────────────────────────────

_TIME_FREQ = {"month": "MS", "year": "YS", "week": "W-MON", "day": "D"}


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self._source = df
        # time-column detection is O(n) — cache per column name
        self._time_cache: dict[str, bool] = {}

    def transform(self, cfg: dict) -> pd.DataFrame:
        x, y    = cfg["x"], cfg["y"]
        agg     = cfg["aggregation"]
        color   = cfg["color"]
        is_time = self._is_time(x, cfg["type"])

        # histogram / scatter: select only needed columns — no full copy
        if cfg["type"] == "histogram":
            return self._source[[x]].dropna()

        if cfg["type"] == "scatter":
            return self._source[[x, y]].drop_duplicates()

        # copy only the real columns this chart needs, then attach any synthetic
        real_cols = [c for c in ({x, y} | ({color} if color else set()))
                     if c in self._source.columns]
        df = self._source[real_cols].copy()

        synthetic: pd.Series | None = cfg.get("_synthetic")
        if synthetic is not None and synthetic.name not in df.columns:
            df[synthetic.name] = synthetic.values

        if is_time:
            self._parse_time_inplace(df, x)
            self._granularise_inplace(df, x, cfg["time_granularity"])

        # ── single aggregation block ──────────────────────────────────
        group_keys = [x] if not color else [x, color]
        if agg != "none":
            df = df.groupby(group_keys)[y].agg(agg).reset_index()
        # ─────────────────────────────────────────────────────────────

        if is_time:
            df = self._fill_gaps(df, x, y, cfg["time_granularity"], color=color)
            if cfg["type"] == "line":
                # rolling smoothing: for multi-series apply per-group so series
                # don't bleed into each other across the color boundary
                if color and color in df.columns:
                    df[y] = (
                        df.groupby(color)[y]
                        .transform(lambda s: s.rolling(2, min_periods=1).mean())
                    )
                else:
                    df[y] = df[y].rolling(2, min_periods=1).mean()
        else:
            df = self._limit_categories(df, cfg, x, y)

        df = self._safe_sort(df, x)

        if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]):
            df[x] = df[x].astype(str)

        return df

    # ── helpers ──────────────────────────────────────────────────────

    def _is_time(self, col: str, chart_type: str) -> bool:
        if chart_type == "scatter":
            return False
        if col in self._time_cache:
            return self._time_cache[col]

        series = self._source[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            self._time_cache[col] = True
            return True
        if series.dtype != object:
            self._time_cache[col] = False
            return False

        parsed = pd.to_datetime(series, errors="coerce")
        result = parsed.notna().sum() / max(len(series), 1) >= 0.80
        self._time_cache[col] = result
        return result

    @staticmethod
    def _parse_time_inplace(df: pd.DataFrame, col: str) -> None:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    @staticmethod
    def _granularise_inplace(df: pd.DataFrame, col: str, granularity: str) -> None:
        freq_map = {"month": "M", "year": "Y", "week": "W", "day": "D"}
        freq = freq_map.get(granularity)
        if freq:
            df[col] = df[col].dt.to_period(freq).dt.to_timestamp()

    @staticmethod
    def _fill_gaps(
        df: pd.DataFrame, x: str, y: str, granularity: str,
        color: str | None = None,
    ) -> pd.DataFrame:
        """
        Fill time-series gaps with zeros.

        Single-series: straightforward asfreq on the x index.
        Multi-series:  asfreq must be applied per-group, otherwise the
                       (x, color) composite index breaks asfreq entirely.
                       Each group is reindexed against the GLOBAL date
                       range so all series share the same x-axis.
        """
        freq = _TIME_FREQ.get(granularity)
        if not freq:
            return df

        if color and color in df.columns:
            # build the global date range from the full aggregated data
            try:
                full_range = pd.date_range(
                    start=df[x].min(), end=df[x].max(), freq=freq
                )
            except Exception:
                return df

            filled_groups = []
            for group_name, group in df.groupby(color):
                try:
                    group = (
                        group.set_index(x)[y]
                        .reindex(full_range, fill_value=0)
                        .reset_index()
                        .rename(columns={"index": x})
                    )
                    group[color] = group_name
                    filled_groups.append(group)
                except Exception:
                    filled_groups.append(group)

            return pd.concat(filled_groups, ignore_index=True) if filled_groups else df

        # single-series path
        try:
            df = df.set_index(x).asfreq(freq).fillna(0).reset_index()
        except Exception:
            pass
        return df

    @staticmethod
    def _limit_categories(df: pd.DataFrame, cfg: dict, x: str, y: str) -> pd.DataFrame:
        if cfg.get("limit_top"):
            return df.nlargest(10, y)
        if df[x].nunique() > 20:
            return df.nlargest(20, y)
        return df

    @staticmethod
    def _safe_sort(df: pd.DataFrame, col: str) -> pd.DataFrame:
        try:
            return df.sort_values(by=col)
        except Exception:
            return df
