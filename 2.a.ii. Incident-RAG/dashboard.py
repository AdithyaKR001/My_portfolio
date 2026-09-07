"""RAG evaluation metrics dashboard.

Reads eval/report.json (produced by `python -m eval.run_eval`) and visualizes
retrieval + generation quality, per-category breakdowns, latency, and a per-case
explorer.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPORT = Path(__file__).resolve().parent / "eval" / "report.json"

st.set_page_config(page_title="Incident-RAG — Eval Dashboard", page_icon="📊", layout="wide")
st.title("📊 Incident-RAG — Evaluation Dashboard")

if not REPORT.exists():
    st.warning("No eval/report.json yet. Generate it with:  `python -m eval.run_eval`")
    st.stop()

report = json.loads(REPORT.read_text())
s = report["summary"]
m = s["metrics"]

st.caption(f"Mode: **{s['mode']}**  •  k = **{s['k']}**  •  cases = **{s['cases']}**  •  "
           f"gate: hit_rate@{s['k']} ≥ {s['threshold']}")

# ---- headline metric cards ----
cols = st.columns(len(m) + 1)
labels = {
    "hit_rate": "Hit-rate@k", "recall": "Recall@k", "precision": "Precision@k",
    "mrr": "MRR", "answer_match": "Answer key-facts", "citation_validity": "Citation validity",
}
for col, (key, val) in zip(cols, m.items()):
    col.metric(labels.get(key, key), f"{val:.2f}")
cols[-1].metric("Avg latency", f"{s['avg_latency_sec']}s")

# ---- aggregate bar chart ----
st.subheader("Aggregate metrics")
st.bar_chart(pd.DataFrame({"score": m}).sort_values("score", ascending=False))

# ---- per-category breakdown ----
st.subheader("By failure category")
by_cat = pd.DataFrame(s["by_category"]).T
if not by_cat.empty:
    st.dataframe(by_cat.style.format("{:.2f}"), use_container_width=True)
    if "hit_rate" in by_cat.columns:
        st.bar_chart(by_cat[["hit_rate"]])

# ---- latency distribution ----
st.subheader("Latency per case")
cases = pd.DataFrame(report["cases"])
st.bar_chart(cases.set_index("id")["latency_sec"])

# ---- per-case explorer ----
st.subheader("Case explorer")
show_fails = st.checkbox("Only show cases that missed all relevant docs (hit_rate = 0)", value=False)
view = cases.copy()
view["hit_rate"] = view["metrics"].apply(lambda d: d.get("hit_rate", 0))
if show_fails:
    view = view[view["hit_rate"] == 0]
for _, row in view.iterrows():
    with st.expander(f"{row['id']} — {row['question']}  (hit_rate={row['hit_rate']})"):
        st.write("**Relevant:**", ", ".join(row["relevant"]))
        st.write("**Retrieved:**", ", ".join(row["retrieved"]))
        st.write("**Metrics:**", row["metrics"])
        if row.get("answer"):
            st.write("**Answer:**")
            st.info(row["answer"])
