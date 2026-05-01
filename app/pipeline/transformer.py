# app/pipeline/transformer.py

import warnings
import pandas as pd

_TIME_FREQ = {"month": "MS", "year": "YS", "week": "W-MON", "day": "D"}

# Column-name substrings that strongly hint at a date/time column
_DATE_NAME_HINTS = ("date", "time", "period", "order_date", "ship", "created", "updated")


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self._source = df
        self._time_cache: dict[str, bool] = {}

    def transform(self, cfg: dict) -> pd.DataFrame:
        x, y    = cfg["x"], cfg["y"]
        agg     = cfg["aggregation"]
        color   = cfg["color"]
        is_time = cfg["type"] != "heatmap" and self._is_time(x, cfg["type"])

        if cfg["type"] == "histogram":
            return self._source[[x]].dropna()

        if cfg["type"] == "box":
            cols = [c for c in [x, y] if c and c in self._source.columns]
            return self._source[list(dict.fromkeys(cols))].dropna()

        if cfg["type"] == "scatter":
            return self._source[[x, y]].drop_duplicates()

        z = cfg.get("z")

        real_cols = [c for c in ({x, y, z} | ({color} if color else set()))
                    if c and c in self._source.columns]
        df = self._source[real_cols].copy()

        synthetic: pd.Series | None = cfg.get("_synthetic")
        if synthetic is not None and synthetic.name not in df.columns:
            df[synthetic.name] = synthetic.values

        if is_time:
            self._parse_time_inplace(df, x)
            self._granularise_inplace(df, x, cfg["time_granularity"])

        if cfg["type"] == "heatmap":
            group_keys = [x, y]
        else:
            group_keys = [x] if not color else [x, color]

        if agg != "none":
            target = z if cfg["type"] == "heatmap" else y
            df = df.groupby(group_keys)[target].agg(agg).reset_index()

        if is_time:
            df = self._fill_gaps(df, x, y, cfg["time_granularity"], color=color)
            if cfg["type"] == "line":
                if color and color in df.columns:
                    df[y] = (
                        df.groupby(color)[y]
                        .transform(lambda s: s.rolling(2, min_periods=1).mean())
                    )
                else:
                    df[y] = df[y].rolling(2, min_periods=1).mean()
        else:
            if cfg["type"] != "heatmap":
                df = self._limit_categories(df, cfg, x, y)

        # FIX A: sort while column is still datetime (before string conversion)
        df = self._sort_chronologically(df, x)

        # FIX C: convert datetime → readable date string AFTER sorting
        if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]):
            df[x] = df[x].dt.strftime("%Y-%m-%d")

        return df

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_time(self, col: str, chart_type: str) -> bool:
        if chart_type == "scatter":
            return False
        if col in self._time_cache:
            return self._time_cache[col]

        series = self._source[col]

        # Already a proper datetime dtype — done
        if pd.api.types.is_datetime64_any_dtype(series):
            self._time_cache[col] = True
            return True

        # Non-object dtypes that are not datetime → not a time column
        if series.dtype != object:
            self._time_cache[col] = False
            return False

        # FIX 1: suppress dateutil warning during the probe parse
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce")

        total      = max(len(series), 1)
        parse_rate = parsed.notna().sum() / total

        # FIX D: name-hinted columns get a relaxed threshold (0.5 instead of 0.8)
        col_lower = col.lower()
        name_hint = any(hint in col_lower for hint in _DATE_NAME_HINTS)
        threshold = 0.5 if name_hint else 0.80

        result = parse_rate >= threshold
        self._time_cache[col] = result
        return result

    @staticmethod
    def _parse_time_inplace(df: pd.DataFrame, col: str) -> None:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return  # already parsed

        # FIX 3: probe the full column directly (no astype(str) conversion)
        # and reuse the parsed result — avoids parsing the column twice.
        FORMATS_TO_TRY = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%b %d, %Y",   # "Jan 06, 2014"
            "%B %d, %Y",   # "January 06, 2014"
            "%Y%m%d",      # "20140106"
        ]

        non_null_count = max(df[col].notna().sum(), 1)

        for fmt in FORMATS_TO_TRY:
            try:
                parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
                if parsed.notna().sum() / non_null_count >= 0.80:
                    df[col] = parsed  # reuse — no second parse needed
                    return
            except Exception:
                continue

        # No single format matched well — fall back to mixed/dateutil inference.
        # FIX 1 (shared): suppress the UserWarning that pandas emits when it
        # cannot infer a single format and falls back to dateutil per-element.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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
        FIX B: always sort by x at the end so the returned DataFrame is
        in chronological order regardless of how reindex/concat ordered rows.
        """
        freq = _TIME_FREQ.get(granularity)
        if not freq:
            # Even without gap-filling, ensure chronological order
            return DataTransformer._sort_df_by_col(df, x)

        if color and color in df.columns:
            try:
                full_range = pd.date_range(
                    start=df[x].min(), end=df[x].max(), freq=freq
                )
            except Exception:
                return DataTransformer._sort_df_by_col(df, x)

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

            if filled_groups:
                result = pd.concat(filled_groups, ignore_index=True)
            else:
                result = df

            # FIX B: sort after concat
            return DataTransformer._sort_df_by_col(result, x)

        # single-series path
        try:
            df = df.set_index(x).asfreq(freq).fillna(0).reset_index()
        except Exception:
            pass

        # FIX B: sort after asfreq (asfreq should preserve order, but be safe)
        return DataTransformer._sort_df_by_col(df, x)

    @staticmethod
    def _sort_df_by_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Sort df by col chronologically.  Works whether col is datetime or string.
        If col is a string that looks like dates, parse temporarily for sorting.

        FIX 2: the original version reassigned series.values (original order)
        back onto the sorted DataFrame — misaligning dates and row data.
        Now uses a temporary __sort_key__ column instead, same pattern as
        _sort_chronologically(), so original column values are never touched.
        """
        if col not in df.columns:
            return df
        try:
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                return df.sort_values(by=col, ignore_index=True)
            # FIX 1 (shared): suppress dateutil warning during the probe parse
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.5:
                # FIX 2: sort via a temp key — never overwrite the original values
                tmp = df.copy()
                tmp["__sort_key__"] = parsed
                return tmp.sort_values("__sort_key__", ignore_index=True).drop(columns="__sort_key__")
            return df.sort_values(by=col, ignore_index=True)
        except Exception:
            return df

    @staticmethod
    def _sort_chronologically(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        FIX A: Primary sort entry point called from transform().
        At this point the column should already be datetime (after
        _parse_time_inplace + optional _granularise_inplace).
        Falls back gracefully if not.
        """
        if col not in df.columns:
            return df
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return df.sort_values(by=col, ignore_index=True)
            # Fallback: try parsing on the fly
            # FIX 1 (shared): suppress dateutil warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.5:
                tmp = df.copy()
                tmp["__sort_key__"] = parsed
                tmp = tmp.sort_values("__sort_key__", ignore_index=True).drop(columns="__sort_key__")
                return tmp
            # Last resort: lexicographic sort (better than nothing for non-date cols)
            return df.sort_values(by=col, ignore_index=True)
        except Exception:
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
        """
        Legacy entry point kept for any external callers.
        Delegates to _sort_chronologically for correct behaviour.
        """
        return DataTransformer._sort_chronologically(df, col)