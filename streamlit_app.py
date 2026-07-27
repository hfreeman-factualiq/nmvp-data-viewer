"""NMVP SharePoint Discovery — Streamlit Community Cloud entrypoint.

Loads precomputed parquet from ./data (produced by export_sharepoint_discovery_cloud.py).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FILES_PARQUET = DATA / "files.parquet"
UNITS_PARQUET = DATA / "fs_statement_units.parquet"
META_JSON = DATA / "meta.json"

HEATMAP_MAX_ROWS = 80

KIND_ORDER = [
    "financial_statement",
    "budget",
    "model",
    "board_deck",
    "investor_update",
    "cap_table",
    "valuation",
    "diligence",
    "legal",
    "financing",
    "other",
]
CORE_KINDS = {
    "financial_statement",
    "budget",
    "model",
    "board_deck",
    "investor_update",
    "cap_table",
}
FIN_KINDS = CORE_KINDS

STATEMENT_FAMILIES = [
    "balance_sheet",
    "income_statement",
    "cash_flow",
    "shareholders_equity",
    "other_financial",
    "unclassified",
]
STATEMENT_FAMILY_LABELS = {
    "balance_sheet": "Balance sheet",
    "income_statement": "Income statement",
    "cash_flow": "Cash flow",
    "shareholders_equity": "Shareholders' equity",
    "other_financial": "Other financial",
    "unclassified": "Unclassified",
}

CATEGORICAL_PALETTE = [
    "#6366F1", "#F43F5E", "#14B8A6", "#F59E0B", "#8B5CF6",
    "#06B6D4", "#EC4899", "#22C55E", "#E879F9", "#3B82F6",
    "#FB923C", "#A3E635", "#F472B6", "#2DD4BF", "#FACC15",
]
HEATMAP_COLORSCALE = [
    [0.0, "#0F172A"],
    [0.15, "#312E81"],
    [0.35, "#6366F1"],
    [0.55, "#14B8A6"],
    [0.75, "#F59E0B"],
    [1.0, "#F43F5E"],
]
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0"),
    title_font=dict(size=16, color="#F8FAFC"),
)


def apply_chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, height=height, margin=dict(t=48, b=24, l=24, r=24))
    return fig


def chart_type_selector(key: str) -> str:
    return st.radio("Chart type", options=["Bar", "Pie"], horizontal=True, key=key)


def render_count_chart(
    counts: pd.DataFrame,
    label_col: str,
    value_col: str,
    chart_type: str,
    title: str,
    *,
    key: str,
) -> None:
    if counts.empty:
        st.info("No data to chart.")
        return
    if chart_type == "Pie":
        fig = px.pie(
            counts,
            names=label_col,
            values=value_col,
            title=title,
            hole=0.35,
            color_discrete_sequence=CATEGORICAL_PALETTE,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#0F172A", width=1.5)),
        )
        st.plotly_chart(apply_chart_layout(fig), use_container_width=True, key=key)
    else:
        fig = px.bar(
            counts,
            x=label_col,
            y=value_col,
            title=title,
            text=value_col,
            color=label_col,
            color_discrete_sequence=CATEGORICAL_PALETTE,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title=None, yaxis_title=value_col, showlegend=False)
        st.plotly_chart(apply_chart_layout(fig), use_container_width=True, key=key)


def build_count_matrix(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    matrix = df.groupby([row_col, col_col]).size().unstack(fill_value=0)
    if matrix.empty:
        return matrix
    matrix = matrix[matrix.sum().sort_values(ascending=False).index]
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    matrix["Total"] = matrix.sum(axis=1)
    totals_row = matrix.sum(axis=0).to_frame().T
    totals_row.index = ["Total"]
    return pd.concat([matrix, totals_row])


def render_matrix_heatmap(matrix: pd.DataFrame, title: str, *, key: str) -> None:
    if matrix.empty:
        st.info("No data for heatmap.")
        return
    heat = matrix.drop(columns=["Total"], errors="ignore").drop(index=["Total"], errors="ignore")
    if heat.empty:
        st.info("No data for heatmap.")
        return
    if len(heat.index) > HEATMAP_MAX_ROWS:
        heat = heat.iloc[:HEATMAP_MAX_ROWS]
        st.caption(f"Showing top {HEATMAP_MAX_ROWS} rows by total.")
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=heat.columns.astype(str).tolist(),
            y=heat.index.astype(str).tolist(),
            colorscale=HEATMAP_COLORSCALE,
            text=heat.values,
            texttemplate="%{text}",
            textfont=dict(color="#F8FAFC", size=12),
            hovertemplate="%{y}<br>%{x}<br>count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#F8FAFC")),
        xaxis=dict(tickangle=-35, side="bottom", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    fig = apply_chart_layout(fig, height=max(420, 28 * len(heat.index) + 140))
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_matrix_section(matrix: pd.DataFrame, title: str, *, key: str) -> None:
    st.markdown(f"**{title}**")
    render_matrix_heatmap(matrix, title, key=f"{key}_heat")
    st.dataframe(matrix, use_container_width=True, key=f"{key}_table")


def table_and_chart(
    counts: pd.DataFrame,
    label_col: str,
    value_col: str,
    chart_type: str,
    title: str,
    *,
    key: str,
) -> None:
    left, right = st.columns([1, 1.4])
    with left:
        st.dataframe(counts, use_container_width=True, hide_index=True, key=f"{key}_tbl")
    with right:
        render_count_chart(
            counts, label_col, value_col, chart_type, title, key=f"{key}_plotly"
        )


@st.cache_data(show_spinner="Loading precomputed inventory…")
def load_files() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(FILES_PARQUET)
    meta: dict[str, Any] = {}
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    if "last_modified" not in df.columns and "lastModifiedDateTime" in df.columns:
        df = df.copy()
        df["last_modified"] = pd.to_datetime(
            df["lastModifiedDateTime"], errors="coerce", utc=True
        )
    return df, meta


@st.cache_data(show_spinner="Loading statement units…")
def load_units() -> pd.DataFrame:
    if not UNITS_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(UNITS_PARQUET)


def filter_statement_units(units: pd.DataFrame, filtered_files: pd.DataFrame) -> pd.DataFrame:
    if units.empty or filtered_files.empty:
        return units.iloc[0:0].copy() if not units.empty else units
    paths = set(filtered_files["file_path"].astype(str))
    return units[units["file_path"].astype(str).isin(paths)].copy()


def _normalize_periods_list(ps: Any) -> list[str]:
    if ps is None:
        return []
    if isinstance(ps, str):
        return [ps] if ps else []
    if isinstance(ps, (list, tuple)):
        return [str(p) for p in ps if p]
    try:
        import numpy as np

        if isinstance(ps, np.ndarray):
            return [str(p) for p in ps.tolist() if p]
    except Exception:
        pass
    try:
        return [str(p) for p in list(ps) if p]
    except TypeError:
        return []


def explode_unit_periods(units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return units
    out = units.copy()
    if "periods" not in out.columns:
        out["periods"] = out.get("period", pd.Series([""] * len(out))).map(
            _normalize_periods_list
        )
    out["periods"] = out["periods"].map(_normalize_periods_list)
    out["periods"] = out["periods"].apply(lambda ps: ps if ps else [""])
    if "period" in out.columns:
        out = out.drop(columns=["period"])
    return out.explode("periods", ignore_index=True).rename(columns={"periods": "period"})


def month_range_ending(end: date | None = None, years: int = 4) -> list[str]:
    end = end or date.today()
    months: list[str] = []
    y, m = end.year, end.month
    for _ in range(years * 12):
        months.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    months.reverse()
    return months


def company_availability(df: pd.DataFrame) -> pd.DataFrame:
    portcos = df[df["company_id"] != "(unassigned)"]
    if portcos.empty:
        return pd.DataFrame()
    core_keys = [
        "financial_statement",
        "budget",
        "model",
        "board_deck",
        "investor_update",
        "cap_table",
    ]
    label_map = {
        0: "none",
        1: "thin",
        2: "partial",
        3: "partial",
        4: "strong",
        5: "strong",
        6: "full",
    }
    rows: list[dict[str, Any]] = []
    for company, g in portcos.groupby("company_id"):
        fin_g = g[g["is_excel_pdf"]] if "is_excel_pdf" in g.columns else g
        kinds = fin_g["content_kind"].value_counts()
        counts = {k: int(kinds.get(k, 0)) for k in KIND_ORDER}
        flags = {f"has_{k}": counts[k] > 0 for k in core_keys}
        score = sum(int(flags[f"has_{k}"]) for k in core_keys)
        rows.append(
            {
                "company_id": company,
                "total_files": len(g),
                "excel_pdf": len(fin_g),
                "excel": int((fin_g["format"] == "excel").sum()) if "format" in fin_g else 0,
                "pdf": int((fin_g["format"] == "pdf").sum()) if "format" in fin_g else 0,
                "size_mb": round(float(g["size"].sum()) / (1024 * 1024), 1)
                if "size" in g.columns
                else 0.0,
                **counts,
                **flags,
                "coverage_score": score,
                "coverage_label": label_map.get(score, "partial"),
                "funds": ", ".join(sorted(g["fund"].dropna().unique())),
                "site_copies": ", ".join(sorted(g["site_copy"].dropna().unique())),
                "top_folders": int(g["top_folder"].nunique()) if "top_folder" in g else 0,
            }
        )
    avail = pd.DataFrame(rows)
    return avail.sort_values(
        ["coverage_score", "financial_statement", "budget", "excel_pdf"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        copies = sorted(df["site_copy"].dropna().unique())
        default_copies = [c for c in ["live", "final"] if c in copies] or copies
        selected_copies = st.multiselect(
            "Site copy / area",
            options=copies,
            default=default_copies,
        )
        funds = sorted(df["fund"].dropna().unique())
        selected_funds = st.multiselect("Fund / area", options=funds, default=funds)
        companies = sorted(
            c for c in df["company_id"].dropna().unique() if c != "(unassigned)"
        )
        include_unassigned = st.checkbox("Include unassigned (non-portco) files", value=False)
        selected_companies = st.multiselect(
            "Company", options=companies, default=companies
        )
        excel_pdf_only = st.checkbox("Excel / PDF only", value=False)
        fin_kinds_only = st.checkbox(
            "Core discovery kinds only (FS / budget / model / board / investor / cap table)",
            value=False,
        )
        if st.button("Clear cache & reload"):
            load_files.clear()
            load_units.clear()
            st.rerun()
        st.caption("Data: precomputed parquet (no live SharePoint).")
        if META_JSON.exists():
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            if meta.get("exported_at"):
                st.caption(f"Exported: `{meta['exported_at']}`")

    out = df[df["site_copy"].isin(selected_copies) & df["fund"].isin(selected_funds)]
    if include_unassigned:
        company_mask = out["company_id"].isin(selected_companies) | (
            out["company_id"] == "(unassigned)"
        )
    else:
        company_mask = out["company_id"].isin(selected_companies)
    out = out[company_mask]
    if excel_pdf_only and "is_excel_pdf" in out.columns:
        out = out[out["is_excel_pdf"]]
    if fin_kinds_only and "is_core_kind" in out.columns:
        out = out[out["is_core_kind"]]
    return out


def render_overview(
    df: pd.DataFrame, meta: dict[str, Any], units: pd.DataFrame | None = None
) -> None:
    st.caption(
        "Precomputed SharePoint inventory analytics (AINMVP1). "
        "Categories: S3 document types + SharePoint folder fallbacks."
    )
    if meta.get("exported_at"):
        st.caption(f"Snapshot exported: `{meta['exported_at']}`")

    chart_type = chart_type_selector("overview_chart")
    total = len(df)
    companies = df.loc[df["company_id"] != "(unassigned)", "company_id"].nunique()
    fin = df[df["is_excel_pdf"]] if "is_excel_pdf" in df.columns else df
    core = fin[fin["is_core_kind"]] if "is_core_kind" in fin.columns else fin
    other_n = int((fin["content_kind"] == "other").sum()) if len(fin) else 0
    size_mb = round(float(df["size"].sum()) / (1024 * 1024), 1) if "size" in df.columns else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Files in filter", f"{total:,}")
    m2.metric("Companies", f"{companies:,}")
    m3.metric("Excel / PDF", f"{len(fin):,}")
    m4.metric("Core discovery kinds", f"{len(core):,}")
    m5.metric("Still other", f"{other_n:,}")

    st.subheader("Category mix (excel/pdf)")
    kind_df = (
        fin.groupby("content_kind", as_index=False)
        .size()
        .rename(columns={"size": "file_count"})
    )
    if not kind_df.empty:
        kind_df["content_kind"] = pd.Categorical(
            kind_df["content_kind"], categories=KIND_ORDER, ordered=True
        )
        kind_df = kind_df.sort_values("content_kind")
    table_and_chart(
        kind_df, "content_kind", "file_count", chart_type, "By content kind", key="ov_kind"
    )

    if "kind_source" in fin.columns:
        src_df = (
            fin.groupby("kind_source", as_index=False)
            .size()
            .rename(columns={"size": "file_count"})
            .sort_values("file_count", ascending=False)
        )
        st.caption("Classification source: filename vs folder/path.")
        table_and_chart(
            src_df, "kind_source", "file_count", chart_type, "Kind source", key="ov_kind_src"
        )

    st.subheader("Download-style scopes (excel/pdf)")
    statements = fin[fin["content_kind"] == "financial_statement"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Financial statements", f"{len(statements):,}")
    s2.metric("Core discovery kinds", f"{len(core):,}")
    s3.metric("All excel/pdf", f"{len(fin):,}")
    st.caption(f"Total size in filter: **{size_mb:,.0f} MB**")

    if units is not None and not units.empty:
        st.subheader("Statement units in filter")
        u1, u2 = st.columns(2)
        u1.metric("Statement units", f"{len(units):,}")
        if "statement_family" in units.columns:
            fam = (
                units.groupby("statement_family", as_index=False)
                .size()
                .rename(columns={"size": "unit_count"})
                .sort_values("unit_count", ascending=False)
            )
            table_and_chart(
                fam,
                "statement_family",
                "unit_count",
                chart_type,
                "Units by statement family",
                key="ov_units_fam",
            )

    st.subheader("Files by company")
    by_co = (
        df[df["company_id"] != "(unassigned)"]
        .groupby("company_id", as_index=False)
        .size()
        .rename(columns={"size": "file_count"})
        .sort_values("file_count", ascending=False)
    )
    table_and_chart(
        by_co, "company_id", "file_count", chart_type, "Files by company", key="ov_by_co"
    )

    st.subheader("Files by fund / site copy")
    c1, c2 = st.columns(2)
    with c1:
        by_fund = (
            df.groupby("fund", as_index=False)
            .size()
            .rename(columns={"size": "file_count"})
            .sort_values("file_count", ascending=False)
        )
        table_and_chart(
            by_fund, "fund", "file_count", chart_type, "Files by fund", key="ov_by_fund"
        )
    with c2:
        by_copy = (
            df.groupby("site_copy", as_index=False)
            .size()
            .rename(columns={"size": "file_count"})
            .sort_values("file_count", ascending=False)
        )
        table_and_chart(
            by_copy, "site_copy", "file_count", chart_type, "Files by site copy", key="ov_by_copy"
        )


def render_by_company(df: pd.DataFrame) -> None:
    st.caption(
        "Coverage score = presence of core kinds among excel/pdf: "
        "financial_statement, budget, model, board_deck, investor_update, cap_table (0–6)."
    )
    avail = company_availability(df)
    if avail.empty:
        st.warning("No portfolio companies in the current filter.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(avail):,}")
    c2.metric("Full coverage (6/6)", int((avail["coverage_score"] == 6).sum()))
    c3.metric("Missing financials", int((~avail["has_financial_statement"]).sum()))
    c4.metric("Missing budgets", int((~avail["has_budget"]).sum()))

    score_counts = (
        avail.groupby("coverage_label", as_index=False)
        .size()
        .rename(columns={"size": "companies"})
    )
    order = ["full", "strong", "partial", "thin", "none"]
    score_counts["coverage_label"] = pd.Categorical(
        score_counts["coverage_label"], categories=order, ordered=True
    )
    score_counts = score_counts.sort_values("coverage_label")
    table_and_chart(
        score_counts,
        "coverage_label",
        "companies",
        chart_type_selector("co_score_chart_type"),
        "Companies by coverage",
        key="co_avail_score",
    )

    show_cols = [
        "company_id",
        "coverage_score",
        "coverage_label",
        "has_financial_statement",
        "has_budget",
        "has_model",
        "has_board_deck",
        "has_investor_update",
        "has_cap_table",
        "financial_statement",
        "budget",
        "model",
        "board_deck",
        "investor_update",
        "cap_table",
        "valuation",
        "diligence",
        "legal",
        "financing",
        "other",
        "excel",
        "pdf",
        "excel_pdf",
        "total_files",
        "size_mb",
        "funds",
        "site_copies",
    ]
    st.dataframe(
        avail[[c for c in show_cols if c in avail.columns]],
        use_container_width=True,
        hide_index=True,
    )

    if "is_excel_pdf" in df.columns:
        fin = df[(df["company_id"] != "(unassigned)") & df["is_excel_pdf"]].copy()
    else:
        fin = df[df["company_id"] != "(unassigned)"].copy()
    if not fin.empty:
        hide_other = st.checkbox("Hide content_kind=other in heatmap", value=True)
        core_only = st.checkbox("Core discovery kinds only in heatmap", value=False)
        fin_h = fin.copy()
        if core_only and "is_core_kind" in fin_h.columns:
            fin_h = fin_h[fin_h["is_core_kind"]]
        elif hide_other:
            fin_h = fin_h[fin_h["content_kind"] != "other"]
        matrix = build_count_matrix(fin_h, "company_id", "content_kind")
        preferred = [c for c in KIND_ORDER if c in matrix.columns] + [
            c for c in matrix.columns if c not in KIND_ORDER and c != "Total"
        ]
        if "Total" in matrix.columns:
            preferred = preferred + ["Total"]
        matrix = matrix.reindex(columns=preferred)
        render_matrix_section(matrix, "Content kind × company", key="co_kind_matrix")


def render_content_mix(df: pd.DataFrame, units: pd.DataFrame | None = None) -> None:
    st.caption("Content kinds from local precompute (filename + path heuristics).")
    fin = df[df["is_excel_pdf"]].copy() if "is_excel_pdf" in df.columns else df.copy()
    if fin.empty:
        st.warning("No excel/pdf in filter.")
        return
    scope = st.radio(
        "Corpus",
        options=["All excel/pdf", "Core discovery kinds only", "Financial statements only"],
        horizontal=True,
        key="sp_mix_scope",
    )
    if scope == "Core discovery kinds only" and "is_core_kind" in fin.columns:
        fin = fin[fin["is_core_kind"]]
    elif scope == "Financial statements only":
        fin = fin[fin["content_kind"] == "financial_statement"]
    if fin.empty:
        st.info("No files for this corpus scope.")
        return

    kind_counts = fin["content_kind"].astype(str).value_counts()
    metrics = [
        ("Financial stmt", "financial_statement"),
        ("Budget", "budget"),
        ("Model", "model"),
        ("Board", "board_deck"),
        ("Investor", "investor_update"),
        ("Cap table", "cap_table"),
        ("Other", "other"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, key) in zip(cols, metrics):
        col.metric(label, int(kind_counts.get(key, 0)))

    chart_type = chart_type_selector("mix_chart")
    kind_df = (
        fin.groupby("content_kind", as_index=False)
        .size()
        .rename(columns={"size": "file_count"})
    )
    kind_df["content_kind"] = pd.Categorical(
        kind_df["content_kind"], categories=KIND_ORDER, ordered=True
    )
    kind_df = kind_df.sort_values("content_kind")
    table_and_chart(
        kind_df, "content_kind", "file_count", chart_type, "By content kind", key="mix_kind"
    )

    company_kind = build_count_matrix(
        fin[fin["company_id"] != "(unassigned)"], "company_id", "content_kind"
    )
    preferred_c = [c for c in KIND_ORDER if c in company_kind.columns] + [
        c for c in company_kind.columns if c not in KIND_ORDER and c != "Total"
    ]
    if "Total" in company_kind.columns:
        preferred_c = preferred_c + ["Total"]
    company_kind = company_kind.reindex(columns=preferred_c)
    render_matrix_section(company_kind, "Content kind × company", key="mix_co_kind")

    if units is not None and not units.empty and "statement_family" in units.columns:
        st.subheader("Statement family mix (units)")
        fam = (
            units.groupby("statement_family", as_index=False)
            .size()
            .rename(columns={"size": "unit_count"})
        )
        table_and_chart(
            fam,
            "statement_family",
            "unit_count",
            chart_type,
            "Units by family",
            key="mix_units_fam",
        )


def render_period_coverage(df: pd.DataFrame, units: pd.DataFrame | None = None) -> None:
    st.caption("Period coverage from precomputed statement units (YYYY-MM).")
    if units is None or units.empty:
        st.warning("No statement units in this filter.")
        return
    exploded = explode_unit_periods(units)
    exploded = exploded[exploded["period"].astype(str).str.match(r"^\d{4}-\d{2}$", na=False)]
    if exploded.empty:
        st.warning("No YYYY-MM periods found on units in this filter.")
        return

    years = st.slider("Lookback years", min_value=1, max_value=8, value=4, key="cov_years")
    months = month_range_ending(years=years)
    companies = sorted(
        c for c in exploded["company_id"].dropna().unique() if c != "(unassigned)"
    )
    if not companies:
        st.warning("No companies with period-tagged units.")
        return

    # Presence matrix: company × month
    present = (
        exploded[exploded["company_id"].isin(companies)]
        .groupby(["company_id", "period"])
        .size()
        .unstack(fill_value=0)
    )
    for m in months:
        if m not in present.columns:
            present[m] = 0
    present = present.reindex(columns=months, fill_value=0)
    present = present.loc[present.sum(axis=1).sort_values(ascending=False).index]
    # Binary heatmap
    heat = (present > 0).astype(int)
    render_matrix_section(heat, "Company × month presence (1 = has unit)", key="cov_heat")

    st.subheader("Units by period")
    by_period = (
        exploded.groupby("period", as_index=False)
        .size()
        .rename(columns={"size": "unit_count"})
        .sort_values("period")
    )
    by_period = by_period[by_period["period"].isin(months)]
    table_and_chart(
        by_period,
        "period",
        "unit_count",
        chart_type_selector("cov_period_chart_type"),
        "Units by period",
        key="cov_by_period",
    )


def render_browser(df: pd.DataFrame) -> None:
    st.caption("Search precomputed inventory rows.")
    q = st.text_input("Search name / path", value="")
    kinds = [k for k in KIND_ORDER if k in set(df["content_kind"].dropna().unique())]
    kinds += sorted(set(df["content_kind"].dropna().unique()) - set(kinds))
    formats = sorted(df["format"].dropna().unique()) if "format" in df.columns else []
    c1, c2 = st.columns(2)
    with c1:
        sel_kinds = st.multiselect("Content kind", options=kinds, default=kinds)
    with c2:
        sel_fmt = st.multiselect("Format", options=formats, default=formats) if formats else []

    view = df[df["content_kind"].isin(sel_kinds)]
    if sel_fmt:
        view = view[view["format"].isin(sel_fmt)]
    if q.strip():
        ql = q.strip().lower()
        view = view[
            view["file_name"].str.lower().str.contains(ql, na=False)
            | view["file_path"].str.lower().str.contains(ql, na=False)
        ]
    st.metric("Matching files", f"{len(view):,}")
    cols = [
        c
        for c in [
            "company_id",
            "fund",
            "site_copy",
            "content_kind",
            "kind_source",
            "format",
            "file_name",
            "top_folder",
            "size_mb",
            "file_path",
            "webUrl",
        ]
        if c in view.columns
    ]
    st.dataframe(
        view[cols].sort_values(["company_id", "content_kind", "file_name"]).head(5000),
        use_container_width=True,
        hide_index=True,
        column_config={
            "webUrl": st.column_config.LinkColumn("Open", display_text="SharePoint"),
        }
        if "webUrl" in cols
        else None,
    )


def main() -> None:
    st.set_page_config(
        page_title="NMVP SharePoint Discovery",
        page_icon="📂",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(160deg, #0B1020 0%, #111827 45%, #1E1B4B 100%);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(20,184,166,0.18));
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 14px;
            padding: 0.6rem 0.9rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("NMVP SharePoint Discovery")
    st.caption("Precomputed analytics snapshot — SharePoint site inventory availability.")

    if not FILES_PARQUET.exists():
        st.error("Missing data/files.parquet. Run export_sharepoint_discovery_cloud.py locally.")
        return

    try:
        df, meta = load_files()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df.empty:
        st.warning("No files in snapshot.")
        return

    filtered = filter_sidebar(df)
    all_units = load_units()
    units = filter_statement_units(all_units, filtered)

    tab_overview, tab_company, tab_mix, tab_coverage, tab_browser = st.tabs(
        ["Overview", "By Company", "Content Mix", "Period coverage", "Browser"]
    )
    with tab_overview:
        render_overview(filtered, meta, units=units)
    with tab_company:
        render_by_company(filtered)
    with tab_mix:
        render_content_mix(filtered, units=units)
    with tab_coverage:
        render_period_coverage(filtered, units=units)
    with tab_browser:
        render_browser(filtered)


if __name__ == "__main__":
    main()
