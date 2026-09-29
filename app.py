"""
app.py
------
🤖 AI Model Comparison & Performance Analytics Dashboard

Run with:  double-click run_dashboard.bat
       or: python -m streamlit run app.py

This version uses SAMPLE / DEMO benchmark data (data/model_comparison.csv).
It does NOT call the ChatGPT, Claude, Gemini or Perplexity APIs.
"""

import html as htmllib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Import check: show helpful instructions if a package is missing
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    import plotly.express as px
    import streamlit as st
except ImportError as error:
    print("\n[ERROR] A required package is missing:", error)
    print("Fix: double-click run_dashboard.bat")
    print("  or run:  python -m pip install -r requirements.txt\n")
    sys.exit(1)

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="AI Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "model_comparison.csv"  # created by data_generator.py

REQUIRED_COLUMNS = [
    "Prompt_ID", "Model", "Category", "Prompt", "Response",
    "Response_Time", "Response_Length", "Accuracy", "Clarity", "Timestamp",
]
NUMERIC_COLUMNS = ["Response_Time", "Response_Length", "Accuracy", "Clarity"]
TEXT_COLUMNS = ["Prompt_ID", "Model", "Category", "Prompt", "Response"]

# Neutral descriptions - no model is claimed to be universally better
MODEL_INFO = {
    "ChatGPT": {
        "category": "General AI",
        "icon": "💬",
        "desc": "A general-purpose AI assistant for writing, explaining, brainstorming and coding.",
    },
    "Claude": {
        "category": "Documents & Analysis",
        "icon": "📄",
        "desc": "An AI assistant often used for reading, summarising and analysing longer documents.",
    },
    "Gemini": {
        "category": "Multimodal AI",
        "icon": "🖼️",
        "desc": "An AI assistant that works with several input types such as text and images.",
    },
    "Perplexity": {
        "category": "Web + Sources",
        "icon": "🌐",
        "desc": "An AI answer engine designed around web search and showing sources.",
    },
}
ALL_MODELS = list(MODEL_INFO.keys())

MODEL_COLORS = {
    "ChatGPT": "#6366f1",
    "Claude": "#a855f7",
    "Gemini": "#3b82f6",
    "Perplexity": "#14b8a6",
}

# ---------------------------------------------------------------------------
# 3. Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.stApp { background: #f5f7fb; color: #1f2937; }
section[data-testid="stSidebar"] { background: #eef2ff; }
.dashboard-title {
    font-size: 2.1rem; font-weight: 800; color: #1e1b4b;
    margin: 0 0 4px 0; line-height: 1.2;
}
.dashboard-subtitle { font-size: 1.05rem; color: #4b5563; margin-bottom: 8px; }
.demo-banner {
    background: #fef3c7; border-left: 5px solid #f59e0b; color: #78350f;
    padding: 10px 14px; border-radius: 8px; font-size: 0.92rem; margin: 10px 0 18px 0;
}
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #312e81;
    border-bottom: 3px solid #c7d2fe; padding-bottom: 6px; margin: 30px 0 14px 0;
}
.kpi-card {
    background: #ffffff; border-radius: 14px; padding: 16px 12px; text-align: center;
    box-shadow: 0 2px 10px rgba(79,70,229,0.10); border-top: 4px solid #6366f1;
}
.kpi-icon { font-size: 1.4rem; }
.kpi-value { font-size: 1.6rem; font-weight: 800; color: #1e1b4b; }
.kpi-label { font-size: 0.85rem; color: #6b7280; }
.model-card {
    background: #ffffff; border-radius: 14px; padding: 16px; min-height: 150px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.model-name { font-size: 1.2rem; font-weight: 700; color: #1e1b4b; }
.model-category {
    display: inline-block; background: #ede9fe; color: #5b21b6; font-size: 0.78rem;
    padding: 2px 10px; border-radius: 999px; margin: 6px 0;
}
.model-desc { font-size: 0.9rem; color: #4b5563; }
.prompt-box {
    background: #ffffff; border-left: 5px solid #6366f1; padding: 14px 16px;
    border-radius: 8px; color: #1f2937; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
div[data-testid="stDataFrame"] {
    background: #ffffff; border-radius: 10px; padding: 4px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.footer {
    text-align: center; color: #6b7280; font-size: 0.85rem;
    margin-top: 40px; padding: 16px 0; border-top: 1px solid #e5e7eb;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------
def esc(value) -> str:
    """Escape text so it is safe to place inside HTML."""
    return htmllib.escape(str(value))


def fmt_num(value, decimals: int = 1, suffix: str = "") -> str:
    """Format a number, or return 'N/A' if it is missing."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def safe_mean(series: pd.Series) -> float:
    """Average of numeric values only. Returns NaN if there are none."""
    numbers = pd.to_numeric(series, errors="coerce").dropna()
    return numbers.mean() if not numbers.empty else float("nan")


def fmt_timestamp(value) -> str:
    """Format a timestamp, or 'N/A' if it is missing/invalid."""
    if pd.isna(value):
        return "N/A"
    return value.strftime("%Y-%m-%d %H:%M")


def section(title: str) -> None:
    """Draw a styled section header."""
    st.markdown(f'<div class="section-header">{esc(title)}</div>', unsafe_allow_html=True)


def kpi_card(icon: str, value: str, label: str) -> None:
    """Draw one KPI card (single-line HTML so Markdown does not break it)."""
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-label">{esc(label)}</div></div>',
        unsafe_allow_html=True,
    )


def load_data():
    """
    Load and clean data/model_comparison.csv.
    Returns (DataFrame or None, list of warning messages, error message or None).
    """
    warnings = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # create data folder if needed

    if not CSV_PATH.exists():
        return None, warnings, "missing"

    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
    except Exception as exc:  # unreadable / broken CSV
        return None, warnings, f"The dataset could not be read: {exc}"

    # Add any missing columns so the app never crashes
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for col in missing_cols:
        df[col] = pd.NA
    if missing_cols:
        warnings.append("Some columns were missing in the CSV and are shown as N/A: " + ", ".join(missing_cols))

    # Safe numeric conversion ("N/A", text, blanks all become NaN - never 0)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text columns: blanks become empty strings
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Fill a blank category from the known model list
    for model, info in MODEL_INFO.items():
        mask = (df["Model"] == model) & (df["Category"] == "")
        df.loc[mask, "Category"] = info["category"]
    df.loc[df["Prompt_ID"] == "", "Prompt_ID"] = "Unknown"

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Drop rows without a model name (cannot be used)
    df = df[df["Model"] != ""].reset_index(drop=True)
    if df.empty:
        return None, warnings, "The dataset has no usable rows."
    return df, warnings, None


def bar_chart(data: pd.DataFrame, metric: str, title: str, y_label: str,
              decimals: int = 1, y_max=None) -> None:
    """Bar chart of one metric. Only numeric values are plotted (no N/A -> 0)."""
    numeric = data[["Model", metric]].copy()
    numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    plotted = numeric.dropna(subset=[metric])
    missing_models = sorted(set(numeric["Model"]) - set(plotted["Model"]))

    if plotted.empty:
        st.info(f"No numeric data available for **{title}**.")
        return

    fig = px.bar(
        plotted, x="Model", y=metric, color="Model",
        color_discrete_map=MODEL_COLORS, text=metric,
        template="plotly_white", title=title,
    )
    fig.update_traces(texttemplate=f"%{{text:.{decimals}f}}", textposition="outside")
    fig.update_layout(
        showlegend=False, height=340, xaxis_title=None, yaxis_title=y_label,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])
    st.plotly_chart(fig, use_container_width=True)
    if missing_models:
        st.caption("Not plotted (value is N/A): " + ", ".join(missing_models))


# ---------------------------------------------------------------------------
# 5. Header
# ---------------------------------------------------------------------------
st.markdown('<div class="dashboard-title">🤖 AI Model Comparison & Performance Analytics Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Compare AI models using response quality, clarity, '
            'length, performance and feature analysis.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="demo-banner">⚠️ <b>DEMO / SAMPLE DATA.</b> Every model answers the <b>same prompt</b>, '
    'but the responses and scores here are sample values, not real benchmark results. '
    'This version does not make real-time API calls to ChatGPT, Claude, Gemini, or Perplexity. '
    'Real benchmarking requires collecting responses to the same prompts under the same conditions.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 6. Load data (friendly error if the CSV is missing)
# ---------------------------------------------------------------------------
df, load_warnings, load_error = load_data()

if load_error == "missing":
    st.error("📂 The dataset file **data/model_comparison.csv** was not found.")
    st.write("Fix it in one of these ways:")
    st.markdown("- Double-click **run_dashboard.bat** (it creates the dataset for you), or\n"
                "- Click the button below to create the demo dataset now.")
    if st.button("Create demo dataset"):
        try:
            import data_generator
            data_generator.create_dataset()
            st.success("Dataset created! Reloading...")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create the dataset: {exc}")
    st.stop()
elif load_error:
    st.error(f"⚠️ {load_error}")
    st.info("Delete data/model_comparison.csv and double-click run_dashboard.bat to recreate it.")
    st.stop()

for message in load_warnings:
    st.warning(message)

# ---------------------------------------------------------------------------
# 7. Sidebar: model, category, prompt, Compare / Reset
# ---------------------------------------------------------------------------
present_models = [m for m in ALL_MODELS if m in set(df["Model"])]
extra_models = sorted(set(df["Model"]) - set(ALL_MODELS))
model_options = present_models + extra_models

category_options = ["All"] + sorted(set(df["Category"]) - {""})

prompt_table = df.drop_duplicates("Prompt_ID")[["Prompt_ID", "Prompt"]]
prompt_map = dict(zip(prompt_table["Prompt_ID"], prompt_table["Prompt"]))
prompt_ids = list(prompt_map.keys())


def default_filters() -> dict:
    return {"models": list(model_options), "category": "All", "prompt": prompt_ids[0]}


def apply_filters() -> None:
    """Called when 'Compare' is clicked."""
    st.session_state["applied"] = {
        "models": list(st.session_state["w_models"]),
        "category": st.session_state["w_category"],
        "prompt": st.session_state["w_prompt"],
    }


def reset_filters() -> None:
    """Called when 'Reset' is clicked."""
    d = default_filters()
    st.session_state["w_models"] = d["models"]
    st.session_state["w_category"] = d["category"]
    st.session_state["w_prompt"] = d["prompt"]
    st.session_state["applied"] = d
    st.session_state["hist_models"] = list(model_options)


# First run (or dataset changed): set safe defaults
if "applied" not in st.session_state:
    reset_filters()
if st.session_state.get("w_prompt") not in prompt_ids:
    st.session_state["w_prompt"] = prompt_ids[0]
if st.session_state.get("w_category") not in category_options:
    st.session_state["w_category"] = "All"
st.session_state["w_models"] = [m for m in st.session_state.get("w_models", []) if m in model_options]
if st.session_state["applied"]["prompt"] not in prompt_ids:
    st.session_state["applied"] = default_filters()

with st.sidebar:
    st.header("⚙️ Controls")
    st.multiselect("1. Select models", model_options, key="w_models")
    st.selectbox("2. Category filter", category_options, key="w_category")
    st.selectbox(
        "3. Prompt", prompt_ids, key="w_prompt",
        format_func=lambda pid: f"{pid}: {prompt_map[pid][:55]}...",
    )
    st.caption("Demo mode: you can choose from the prompts stored in the dataset. "
               "Typing a brand-new prompt would need real API calls (future version).")
    st.button("🔍 Compare", on_click=apply_filters, use_container_width=True, type="primary")
    st.button("🔄 Reset", on_click=reset_filters, use_container_width=True)

applied = st.session_state["applied"]

# ---------------------------------------------------------------------------
# 8. Apply filters
# ---------------------------------------------------------------------------
if not applied["models"]:
    st.info("👈 Please select at least one model in the sidebar, then click **Compare**.")
    st.stop()

filtered = df[(df["Prompt_ID"] == applied["prompt"]) & (df["Model"].isin(applied["models"]))]
if applied["category"] != "All":
    filtered = filtered[filtered["Category"] == applied["category"]]

# Keep the model order consistent
order = {m: i for i, m in enumerate(model_options)}
filtered = filtered.sort_values("Model", key=lambda s: s.map(order)).reset_index(drop=True)

if filtered.empty:
    st.warning("No records match the selected models / category / prompt. "
               "Try another combination or click **Reset**.")
    st.stop()

# ---------------------------------------------------------------------------
# 9. KPI cards
# ---------------------------------------------------------------------------
section("📊 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("🧩", str(filtered["Model"].nunique()), "Models Compared")
with k2:
    kpi_card("🎯", fmt_num(safe_mean(filtered["Accuracy"]), 1, " / 10"), "Average Accuracy")
with k3:
    kpi_card("✨", fmt_num(safe_mean(filtered["Clarity"]), 1, " / 10"), "Average Clarity")
with k4:
    kpi_card("📏", fmt_num(safe_mean(filtered["Response_Length"]), 0, " words"), "Avg Response Length")
with k5:
    kpi_card("⏱️", fmt_num(safe_mean(filtered["Response_Time"]), 1, " s"), "Avg Response Time")

# ---------------------------------------------------------------------------
# 10. Model overview
# ---------------------------------------------------------------------------
section("🧭 Model Overview")
overview_cols = st.columns(len(filtered))
for col, (_, row) in zip(overview_cols, filtered.iterrows()):
    info = MODEL_INFO.get(row["Model"], {"icon": "🤖", "desc": "No description available."})
    category = row["Category"] or "N/A"
    with col:
        st.markdown(
            f'<div class="model-card"><div class="model-name">{info["icon"]} {esc(row["Model"])}</div>'
            f'<div class="model-category">{esc(category)}</div>'
            f'<div class="model-desc">{esc(info["desc"])}</div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# 11. Prompt comparison
# ---------------------------------------------------------------------------
section("📝 Prompt Comparison")
st.markdown("**Selected prompt (the same for every model):**")
st.markdown(f'<div class="prompt-box">{esc(prompt_map[applied["prompt"]])}</div>', unsafe_allow_html=True)
st.write("")
for _, row in filtered.iterrows():
    with st.expander(f"{MODEL_INFO.get(row['Model'], {}).get('icon', '🤖')} {row['Model']} response", expanded=False):
        st.write(row["Response"] if row["Response"] else "N/A")

# ---------------------------------------------------------------------------
# 12. Comparison table
# ---------------------------------------------------------------------------
section("📋 Comparison Table")
table = pd.DataFrame({
    "Model": filtered["Model"],
    "Category": filtered["Category"].replace("", "N/A"),
    "Response": filtered["Response"].replace("", "N/A"),
    "Accuracy": filtered["Accuracy"].apply(lambda v: fmt_num(v, 1, " / 10")),
    "Clarity": filtered["Clarity"].apply(lambda v: fmt_num(v, 1, " / 10")),
    "Response Length": filtered["Response_Length"].apply(lambda v: fmt_num(v, 0, " words")),
    "Response Time": filtered["Response_Time"].apply(lambda v: fmt_num(v, 1, " s")),
})
st.dataframe(table, hide_index=True)

# ---------------------------------------------------------------------------
# 13. Performance charts (numeric values only)
# ---------------------------------------------------------------------------
section("📈 Performance Charts")
c1, c2 = st.columns(2)
with c1:
    bar_chart(filtered, "Accuracy", "Accuracy by Model", "Score (1-10)", 1, 10)
with c2:
    bar_chart(filtered, "Clarity", "Clarity by Model", "Score (1-10)", 1, 10)
c3, c4 = st.columns(2)
with c3:
    bar_chart(filtered, "Response_Length", "Response Length by Model", "Words", 0)
with c4:
    bar_chart(filtered, "Response_Time", "Response Time by Model", "Seconds", 1)

# ---------------------------------------------------------------------------
# 14. Feature comparison (neutral wording; availability varies by product/plan)
# ---------------------------------------------------------------------------
section("🧰 Feature Comparison")
features = pd.DataFrame(
    [
        ["General AI", "Core feature", "Core feature", "Core feature", "Available"],
        ["Coding", "Available", "Available", "Available", "Depends on mode/product"],
        ["Document Analysis", "Available", "Core feature", "Available", "Depends on mode/product"],
        ["Multimodal", "Available", "Available", "Core feature", "Depends on mode/product"],
        ["Web Search", "Depends on mode/product", "Depends on mode/product",
         "Depends on mode/product", "Core feature"],
        ["Citations", "Depends on mode/product", "Depends on mode/product",
         "Depends on mode/product", "Core feature"],
    ],
    columns=["Feature", "ChatGPT", "Claude", "Gemini", "Perplexity"],
)
st.dataframe(features, hide_index=True)
st.caption("Features and availability change over time and differ by plan, version and product. "
           "Check each provider's official website for current details. This table does not rank the models.")

# ---------------------------------------------------------------------------
# 15. Detailed response analysis
# ---------------------------------------------------------------------------
section("🔍 Detailed Response Analysis")
for _, row in filtered.iterrows():
    with st.container(border=True):
        st.subheader(f"{MODEL_INFO.get(row['Model'], {}).get('icon', '🤖')} {row['Model']}")
        st.caption(f"Category: {row['Category'] or 'N/A'}")
        st.markdown("**Prompt:**")
        st.write(row["Prompt"] or "N/A")
        st.markdown("**Response:**")
        st.write(row["Response"] or "N/A")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", fmt_num(row["Accuracy"], 1, " / 10"))
        m2.metric("Clarity", fmt_num(row["Clarity"], 1, " / 10"))
        m3.metric("Response Length", fmt_num(row["Response_Length"], 0, " words"))
        m4.metric("Response Time", fmt_num(row["Response_Time"], 1, " s"))

# ---------------------------------------------------------------------------
# 16. Response history (all records in the CSV, filter by model)
# ---------------------------------------------------------------------------
section("📚 Response History")
if "hist_models" not in st.session_state:
    st.session_state["hist_models"] = list(model_options)
st.session_state["hist_models"] = [m for m in st.session_state["hist_models"] if m in model_options]
st.multiselect("Filter history by model", model_options, key="hist_models")

history = df[df["Model"].isin(st.session_state["hist_models"])].sort_values("Timestamp", ascending=False)
if history.empty:
    st.info("No history records for the selected models.")
else:
    history_table = pd.DataFrame({
        "Timestamp": history["Timestamp"].apply(fmt_timestamp),
        "Prompt ID": history["Prompt_ID"],
        "Model": history["Model"],
        "Category": history["Category"].replace("", "N/A"),
        "Accuracy": history["Accuracy"].apply(lambda v: fmt_num(v, 1, " / 10")),
        "Clarity": history["Clarity"].apply(lambda v: fmt_num(v, 1, " / 10")),
        "Length": history["Response_Length"].apply(lambda v: fmt_num(v, 0, " words")),
        "Time": history["Response_Time"].apply(lambda v: fmt_num(v, 1, " s")),
        "Response": history["Response"].replace("", "N/A"),
    })
    st.dataframe(history_table, hide_index=True)
    st.caption(f"{len(history_table)} demo record(s) shown.")

# ---------------------------------------------------------------------------
# 17. Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">🤖 AI Model Comparison & Performance Analytics Dashboard · Built with Python, '
    'Streamlit, Pandas &amp; Plotly · Sample/demo data only, no real API calls</div>',
    unsafe_allow_html=True,
)
