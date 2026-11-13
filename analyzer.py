import numpy as np
import pandas as pd

LABELS = {
    "integer": "Numeric", "float": "Numeric",
    "category": "Categorical", "boolean": "Categorical",
    "datetime": "Datetime", "string": "Text", "text": "Text",
}

def fmt(x):
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"


def read(path, delimiter, header):
    kw = dict(sep=delimiter, header=0 if header else None, low_memory=False)
    try:
        df = pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", **kw)
    df.columns = [str(c) for c in df.columns]
    return df.replace([np.inf, -np.inf], np.nan)


def infer_type(s):
    """Returns (type, stored_as_text)."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean", False
    if pd.api.types.is_integer_dtype(s):
        return "integer", False
    if pd.api.types.is_float_dtype(s):
        return "float", False
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", False
    values = s.dropna().astype(str)
    if values.empty:
        return "string", False
    parsed = pd.to_datetime(values.head(500), errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.9:
        return "datetime", True
    if values.nunique() <= max(20, 0.05 * len(values)):
        return "category", False
    return ("text" if values.str.len().mean() > 50 else "string"), False


def profile_column(s):
    ctype, as_text = infer_type(s)
    non_null = s.dropna()
    count = len(non_null)
    missing_pct = round(float(s.isna().mean() * 100), 1)
    unique = int(non_null.nunique())
    unique_pct = round(unique / max(count, 1) * 100, 2)
    issues, stats = [], None

    if missing_pct >= 40:
        issues.append(("Mostly empty", "bad"))
    elif missing_pct > 5:
        issues.append((f"{missing_pct}% missing", "warn"))
    if unique <= 1 and count:
        issues.append(("Constant column", "warn"))
    if as_text:
        issues.append(("Stored as text", "warn"))

    if ctype in ("integer", "float") and count:
        skew = non_null.skew() if count > 2 else float("nan")
        stats = {
            "mean": fmt(non_null.mean()), "median": fmt(non_null.median()),
            "std": fmt(non_null.std()), "min": fmt(non_null.min()),
            "max": fmt(non_null.max()), "skew": fmt(skew),
        }
        q1, q3 = non_null.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            out = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum())
            if out:
                issues.append((f"{out:,} outliers", "bad" if out / count > 0.01 else "warn"))
        if pd.notna(skew) and abs(skew) > 2:
            issues.append(("Right-skewed" if skew > 0 else "Left-skewed", "warn"))

    if ctype in ("category", "string") and count:
        as_str = non_null.astype(str)
        if as_str.str.strip().str.lower().nunique() < as_str.nunique():
            issues.append(("Mixed casing", "warn"))
    if ctype in ("integer", "string") and unique_pct == 100 and count > 1:
        issues.append(("ID-like", "ok"))

    return ctype, {
        "name": s.name, "type": ctype, "unique_pct": unique_pct, "missing_pct": missing_pct,
        "issues": [{"label": l, "severity": v} for l, v in issues], "stats": stats,
    }


def histogram(s):
    counts, edges = np.histogram(s.dropna(), bins=14)
    return [{"label": fmt(e), "count": int(c)} for c, e in zip(counts, edges[:-1])]


def profile_csv(path, delimiter=",", header=True):
    df = read(path, delimiter, header)
    if df.empty:
        raise ValueError("the file has no data rows")
    n, m = df.shape

    types, columns = {}, []
    for name in df.columns:
        t, col = profile_column(df[name])
        types[name] = t
        columns.append(col)

    num_cols = [c for c in df.columns if types[c] in ("integer", "float")]
    corr = df[num_cols[:12]].corr().round(2).fillna(0) if len(num_cols) > 1 else None

    missing_cells = int(df.isna().sum().sum())
    missing_pct = round(missing_cells / (n * m) * 100, 1)
    dups = int(df.duplicated().sum())
    dup_pct = round(dups / n * 100, 2)

    flagged = [(c["name"], i) for c in columns for i in c["issues"] if i["severity"] != "ok"]
    critical = sum(1 for _, i in flagged if i["severity"] == "bad")
    
    penalty = critical * 6 + (len(flagged) - critical) * 2 + min(15, missing_pct) + min(10, dup_pct * 2)
    score = max(0, round(100 - penalty))
    checks_total = m * 4 + 2
    top = set(df.isna().sum().nlargest(30).index)
    shown_cols = [c for c in df.columns if c in top]
    
    return {
        "row_count": n,
        "column_count": m,
        "health_score": score,
        "profile": {
            "overview": {"missing_cells_pct": missing_pct, "duplicate_rows": dups, "duplicate_rows_pct": dup_pct},
            "column_types": [
                {"label": label, "count": sum(1 for t in types.values() if LABELS[t] == label)}
                for label in ("Numeric", "Categorical", "Datetime", "Text")
                if any(LABELS[t] == label for t in types.values())
            ],
            "columns": columns,
            "histograms": {c: histogram(df[c]) for c in num_cols[:30]},
            "correlation": {
                "columns": num_cols[:12],
                "values": corr.values.tolist() if corr is not None else [],
            },
            "missing": {
                "cells": missing_cells,
                "columns_with_gaps": int((df.isna().sum() > 0).sum()),
                "rows_with_gaps": int(df.isna().any(axis=1).sum()),
                "rows_with_gaps_pct": round(float(df.isna().any(axis=1).mean() * 100), 1),
                "matrix": {
                    "columns": shown_cols,
                    "rows": df.head(40)[shown_cols].isna().values.tolist(),
                },
            },
            "health": {
                "score": score, "issues": len(flagged), "critical_issues": critical,
                "columns_affected": len({name for name, _ in flagged}),
                "checks_passed": max(checks_total - len(flagged), 0), "checks_total": checks_total,
            },
        },
    }