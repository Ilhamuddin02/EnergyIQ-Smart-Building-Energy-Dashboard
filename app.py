"""
⚡ EnergyIQ — Smart Building Energy Dashboard
Kelompok 14 · Kelas A
Rafif Titan Athallah (2043231082) · Muhammad Ilhamuddin (2043231098)
Proyek EAS · Mata Kuliah Pembelajaran Mesin · 2025
"""

import os, json, math, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyIQ · Smart Building Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IDR_PER_KWH  = 1_699.53
BASELINE_KWH = 320.0
DATA_PATH     = "processed_data.csv"
META_PATH     = "model_metadata.json"

# ── Colour palette ────────────────────────────────────────────────────────────
INDIGO   = "#6366f1"
EMERALD  = "#10b981"
AMBER    = "#f59e0b"
RED      = "#ef4444"
SLATE    = "#94a3b8"
BG_DARK  = "#0f172a"
BG_CARD  = "#1e293b"
BG_LIGHT = "#334155"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #e2e8f0;
}
.stApp { background-color: #0f172a; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0f1729; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.82rem; font-weight: 500; }
[data-testid="stSidebar"] .stSelectbox > div > div { background-color: #1e293b; border: 1px solid #334155; color: #e2e8f0; }
[data-testid="stSidebar"] .stSlider > div { color: #e2e8f0; }
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #162035 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: transform .15s;
}
[data-testid="metric-container"]:hover { transform: translateY(-2px); }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Section headers */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #e2e8f0;
    border-left: 3px solid #6366f1; padding-left: .65rem;
    margin: 1.6rem 0 .8rem 0; letter-spacing: .02em;
}

/* Info / alert boxes */
.alert-box {
    border-radius: 10px; padding: .75rem 1rem; margin: .5rem 0;
    font-size: .85rem; font-weight: 500;
    border: 1px solid;
}
.alert-red    { background: #450a0a33; border-color: #ef4444; color: #fca5a5; }
.alert-green  { background: #052e1633; border-color: #10b981; color: #6ee7b7; }
.alert-yellow { background: #43200033; border-color: #f59e0b; color: #fcd34d; }
.alert-blue   { background: #172554aa; border-color: #6366f1; color: #a5b4fc; }

/* Tab headers */
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px 8px 0 0;
    color: #94a3b8;
    font-weight: 600;
    font-size: .85rem;
}
.stTabs [aria-selected="true"] { color: #6366f1 !important; border-bottom: 2px solid #6366f1; }

/* Plotly frames */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Footer */
.footer { 
    text-align: center; color: #475569; font-size: .75rem; 
    padding: 1.5rem 0; margin-top: 2rem;
    border-top: 1px solid #1e293b;
}

/* Expander */
[data-testid="stExpander"] { background: #1e293b; border: 1px solid #334155; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data & model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df.sort_values("Timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_resource(show_spinner="Memuat model…")
def load_models():
    rf  = joblib.load("rf_model.pkl")
    le_b = joblib.load("le_building.pkl")
    le_o = joblib.load("le_occupancy.pkl")
    le_t = joblib.load("le_type.pkl")
    return rf, le_b, le_o, le_t

# ─────────────────────────────────────────────────────────────────────────────
# Helper: prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_energy(hour, day_of_week, temperature, humidity,
                   occupancy_label, building_type_label, building_id_label,
                   month, rf_model, le_building, le_occupancy, le_type):
    is_weekend   = 1 if day_of_week >= 5 else 0
    is_work_hour = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    hour_sin  = math.sin(2 * math.pi * hour / 24)
    hour_cos  = math.cos(2 * math.pi * hour / 24)
    day_sin   = math.sin(2 * math.pi * day_of_week / 7)
    day_cos   = math.cos(2 * math.pi * day_of_week / 7)
    occ_enc   = int(le_occupancy.transform([occupancy_label])[0])
    type_enc  = int(le_type.transform([building_type_label])[0])
    bid_enc   = int(le_building.transform([building_id_label])[0])

    X_pred = pd.DataFrame([{
        "Hour": hour, "DayOfWeek": day_of_week, "IsWeekend": is_weekend,
        "IsWorkingHour": is_work_hour, "Month": month,
        "Hour_sin": hour_sin, "Hour_cos": hour_cos,
        "Day_sin": day_sin, "Day_cos": day_cos,
        "Temperature (°C)": temperature, "Humidity (%)": humidity,
        "Occupancy_enc": occ_enc, "BuildingType_enc": type_enc,
        "Building_ID_enc": bid_enc,
    }])
    return max(50.0, float(rf_model.predict(X_pred)[0]))

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e293b",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(t=45, b=35, l=45, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="#334155", linecolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", linecolor="#334155", zerolinecolor="#334155"),
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(df, le_building, le_occupancy, le_type):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:.5rem 0 1rem">
            <span style="font-size:2rem">⚡</span>
            <div style="font-size:1.15rem;font-weight:800;color:#e2e8f0;letter-spacing:.03em">EnergyIQ</div>
            <div style="font-size:.7rem;color:#6366f1;font-weight:600;letter-spacing:.08em">SMART BUILDING DASHBOARD</div>
        </div>
        <hr style="border-color:#1e293b;margin:.5rem 0 1rem">
        """, unsafe_allow_html=True)

        st.markdown("**🏢 Filter Data**")
        btype_opts = ["Semua"] + sorted(df["Building_Type"].unique().tolist())
        selected_type = st.selectbox("Tipe Gedung", btype_opts)

        bid_opts = ["Semua"] + sorted(df["Building_ID"].unique().tolist())
        selected_bid = st.selectbox("ID Gedung", bid_opts)

        date_min = df["Timestamp"].min().date()
        date_max = df["Timestamp"].max().date()
        date_range = st.date_input("Rentang Tanggal",
                                   value=(date_min, date_max),
                                   min_value=date_min, max_value=date_max)

        st.markdown("<hr style='border-color:#1e293b;margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown("**🔮 Prediksi Energi**")

        pred_hour = st.slider("Jam", 0, 23, datetime.now().hour)
        pred_day  = st.selectbox("Hari", ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"])
        pred_month = st.slider("Bulan", 1, 12, datetime.now().month)
        pred_temp = st.slider("Suhu (°C)", 10, 45, 28)
        pred_hum  = st.slider("Kelembaban (%)", 20, 100, 65)
        pred_occ  = st.selectbox("Tingkat Hunian", list(le_occupancy.classes_))
        pred_btype = st.selectbox("Tipe Gedung (Prediksi)", list(le_type.classes_))
        pred_bid   = st.selectbox("ID Gedung (Prediksi)", list(le_building.classes_))

        st.markdown("<hr style='border-color:#1e293b;margin:.8rem 0'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.65rem;color:#475569;text-align:center;line-height:1.6">
            Kelompok 14 · Kelas A<br>
            Rafif Titan Athallah · Muhammad Ilhamuddin<br>
            EAS Pembelajaran Mesin · 2025
        </div>""", unsafe_allow_html=True)

    day_map = {"Senin":0,"Selasa":1,"Rabu":2,"Kamis":3,"Jumat":4,"Sabtu":5,"Minggu":6}
    return {
        "type": selected_type, "bid": selected_bid,
        "date_range": date_range,
        "pred": dict(hour=pred_hour, day=day_map[pred_day], month=pred_month,
                     temp=pred_temp, hum=pred_hum, occ=pred_occ,
                     btype=pred_btype, bid=pred_bid, day_label=pred_day)
    }

# ─────────────────────────────────────────────────────────────────────────────
# FILTER helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(df, filters):
    dff = df.copy()
    if filters["type"] != "Semua":
        dff = dff[dff["Building_Type"] == filters["type"]]
    if filters["bid"] != "Semua":
        dff = dff[dff["Building_ID"] == filters["bid"]]
    dr = filters["date_range"]
    if len(dr) == 2:
        dff = dff[(dff["Timestamp"].dt.date >= dr[0]) & (dff["Timestamp"].dt.date <= dr[1])]
    return dff

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
def tab_overview(dff, meta, kwh_pred, pct):
    # ── KPI row ──
    total_kwh   = dff["Energy_Usage (kWh)"].sum()
    avg_kwh     = dff["Energy_Usage (kWh)"].mean()
    total_cost  = total_kwh * IDR_PER_KWH
    total_saving = max(0, (BASELINE_KWH - avg_kwh) * IDR_PER_KWH)
    buildings   = dff["Building_ID"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⚡ Total Konsumsi", f"{total_kwh:,.0f} kWh", f"{len(dff):,} rekaman")
    c2.metric("📊 Rata-rata / Jam", f"{avg_kwh:.1f} kWh", f"Baseline {BASELINE_KWH} kWh")
    c3.metric("💰 Total Biaya", f"Rp {total_cost/1e6:.1f} Jt")
    c4.metric("🏢 Gedung Aktif", str(buildings))
    c5.metric("🔮 Prediksi Sekarang", f"{kwh_pred:.1f} kWh", f"{pct:+.1f}% vs baseline",
              delta_color="inverse")

    # ── Prediction alert ──
    if pct > 10:
        cls, emoji, msg = "alert-red", "🔴", f"Konsumsi TINGGI ({kwh_pred:.1f} kWh). Tunda beban berat ke off-peak."
    elif pct < -5:
        cls, emoji, msg = "alert-green", "🟢", f"Konsumsi HEMAT ({kwh_pred:.1f} kWh). Efisiensi energi tercapai!"
    else:
        cls, emoji, msg = "alert-yellow", "🟡", f"Konsumsi NORMAL ({kwh_pred:.1f} kWh). Dalam batas wajar."
    st.markdown(f'<div class="alert-box {cls}">{emoji} {msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Tren Konsumsi Energi</div>', unsafe_allow_html=True)

    # ── Time-series chart ──
    daily = (dff.set_index("Timestamp")["Energy_Usage (kWh)"]
               .resample("D").mean().reset_index()
               .rename(columns={"Timestamp": "Tanggal", "Energy_Usage (kWh)": "kWh"}))
    fig_ts = go.Figure()
    fig_ts.add_scatter(x=daily["Tanggal"], y=daily["kWh"],
                       mode="lines", name="Rata-rata harian",
                       line=dict(color=INDIGO, width=2.2),
                       fill="tozeroy",
                       fillcolor="rgba(99,102,241,.12)")
    fig_ts.add_hline(y=BASELINE_KWH, line=dict(color=AMBER, dash="dash", width=1.2),
                     annotation_text=f"Baseline {BASELINE_KWH} kWh",
                     annotation_font_color=AMBER)
    fig_ts.update_layout(**LAYOUT_BASE, title="Tren Rata-rata Konsumsi Harian", height=320)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Hourly + weekday bars ──
    c1, c2 = st.columns(2)
    with c1:
        hourly = dff.groupby("Hour")["Energy_Usage (kWh)"].mean().reset_index()
        hourly["Tipe"] = hourly["Hour"].apply(lambda h: "Jam Kerja" if 8 <= h <= 18 else "Off-peak")
        fig_h = px.bar(hourly, x="Hour", y="Energy_Usage (kWh)", color="Tipe",
                       color_discrete_map={"Jam Kerja": AMBER, "Off-peak": INDIGO},
                       labels={"Energy_Usage (kWh)": "kWh rata-rata"},
                       title="Konsumsi Rata-rata per Jam")
        fig_h.update_layout(**LAYOUT_BASE, height=310, showlegend=True)
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        day_names = ["Sen","Sel","Rab","Kam","Jum","Sab","Min"]
        daily_bar = dff.groupby("DayOfWeek")["Energy_Usage (kWh)"].mean().reset_index()
        daily_bar["Hari"] = daily_bar["DayOfWeek"].map(lambda d: day_names[d])
        daily_bar["Tipe"] = daily_bar["DayOfWeek"].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
        fig_d = px.bar(daily_bar, x="Hari", y="Energy_Usage (kWh)", color="Tipe",
                       color_discrete_map={"Weekday": INDIGO, "Weekend": EMERALD},
                       labels={"Energy_Usage (kWh)": "kWh rata-rata"},
                       title="Konsumsi Rata-rata per Hari",
                       category_orders={"Hari": day_names})
        fig_d.update_layout(**LAYOUT_BASE, height=310)
        st.plotly_chart(fig_d, use_container_width=True)


# ── Tab 2: EDA ────────────────────────────────────────────────────────────────
def tab_eda(dff):
    st.markdown('<div class="section-header">Distribusi & Korelasi</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=dff["Energy_Usage (kWh)"], nbinsx=50,
                                        marker_color=INDIGO, opacity=.8, name="kWh"))
        mean_v = dff["Energy_Usage (kWh)"].mean()
        fig_hist.add_vline(x=mean_v, line=dict(color="white", dash="dash", width=1.4),
                           annotation_text=f"μ={mean_v:.1f}", annotation_font_color="white")
        fig_hist.update_layout(**LAYOUT_BASE, title="Distribusi Energy_Usage (kWh)", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_scatter = px.scatter(dff.sample(min(3000, len(dff))),
                                 x="Temperature (°C)", y="Energy_Usage (kWh)",
                                 color="Occupancy_Level", opacity=.5, size_max=5,
                                 color_discrete_map={"Low": EMERALD, "Medium": AMBER, "High": RED},
                                 title="Suhu vs Konsumsi per Hunian",
                                 labels={"Energy_Usage (kWh)": "kWh"})
        fig_scatter.update_layout(**LAYOUT_BASE, height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-header">Boxplot per Kategori</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_box = px.box(dff, x="Building_Type", y="Energy_Usage (kWh)",
                         color="Building_Type",
                         color_discrete_map={"Commercial": INDIGO, "Industrial": RED,
                                             "Educational": EMERALD, "Residential": AMBER},
                         title="Distribusi per Tipe Gedung")
        fig_box.update_layout(**LAYOUT_BASE, height=340, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        fig_box2 = px.box(dff, x="Occupancy_Level", y="Energy_Usage (kWh)",
                          color="Occupancy_Level",
                          category_orders={"Occupancy_Level": ["Low", "Medium", "High"]},
                          color_discrete_map={"Low": EMERALD, "Medium": AMBER, "High": RED},
                          title="Distribusi per Tingkat Hunian")
        fig_box2.update_layout(**LAYOUT_BASE, height=340, showlegend=False)
        st.plotly_chart(fig_box2, use_container_width=True)

    st.markdown('<div class="section-header">Heatmap Korelasi</div>', unsafe_allow_html=True)
    num_cols = ["Energy_Usage (kWh)", "Temperature (°C)", "Humidity (%)",
                "Hour", "DayOfWeek", "IsWeekend", "IsWorkingHour", "Month"]
    existing = [c for c in num_cols if c in dff.columns]
    corr = dff[existing].corr().round(2)
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Heatmap Korelasi Antar Fitur")
    fig_heat.update_layout(**LAYOUT_BASE, height=450)
    st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 3: Model Performance ──────────────────────────────────────────────────
def tab_model(meta, dff, rf_model, le_building, le_occupancy, le_type):
    st.markdown('<div class="section-header">Metrik Evaluasi Model</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📉 RMSE", f"{meta['rmse']:.2f} kWh", help="Root Mean Squared Error")
    c2.metric("📏 MAE",  f"{meta['mae']:.2f} kWh",  help="Mean Absolute Error")
    c3.metric("📐 R²",   f"{meta['r2']:.4f}",        help="Koefisien Determinasi")
    c4.metric("🎯 MAPE", f"{meta['mape']:.2f}%",      help="Mean Absolute Percentage Error")

    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

    fi_data = meta.get("feature_importance", [])
    if fi_data:
        fi_df = pd.DataFrame(fi_data).sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#4f46e5"], [0.5, "#6366f1"], [1, "#818cf8"]],
            ),
            text=fi_df["Importance"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
        ))
        fig_fi.update_layout(**LAYOUT_BASE, title="Feature Importance — Random Forest", height=420)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Prediction simulation table ──
    st.markdown('<div class="section-header">Simulasi Skenario Prediksi</div>', unsafe_allow_html=True)
    scenarios = [
        dict(nama="Senin 09:00 · Panas · Hunian Padat",  hour=9,  dow=0, temp=33, hum=65, occ="High",   btype="Commercial",  bid="B001"),
        dict(nama="Rabu 14:00 · Sejuk · Hunian Sedang",  hour=14, dow=2, temp=22, hum=55, occ="Medium", btype="Educational", bid="B005"),
        dict(nama="Sabtu 22:00 · Normal · Hunian Sepi",  hour=22, dow=5, temp=27, hum=70, occ="Low",    btype="Residential", bid="B010"),
        dict(nama="Jumat 17:00 · Terik · Hunian Padat",  hour=17, dow=4, temp=35, hum=80, occ="High",   btype="Industrial",  bid="B015"),
        dict(nama="Minggu 03:00 · Dingin · Hunian Sepi", hour=3,  dow=6, temp=10, hum=40, occ="Low",    btype="Commercial",  bid="B020"),
    ]
    rows = []
    for s in scenarios:
        kwh  = predict_energy(s["hour"], s["dow"], s["temp"], s["hum"],
                              s["occ"], s["btype"], s["bid"], 6,
                              rf_model, le_building, le_occupancy, le_type)
        pct  = (kwh - BASELINE_KWH) / BASELINE_KWH * 100
        cost = kwh * IDR_PER_KWH
        flag = "🔴 Tinggi" if pct > 10 else ("🟢 Hemat" if pct < -5 else "🟡 Normal")
        rows.append({"Skenario": s["nama"], "Prediksi (kWh)": round(kwh, 1),
                     "vs Baseline": f"{pct:+.1f}%", "Biaya/jam": f"Rp {cost:,.0f}", "Status": flag})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model info expander ──
    with st.expander("ℹ️ Detail Hyperparameter Model"):
        info = {
            "Algoritma": "Random Forest Regressor",
            "n_estimators": 300, "max_depth": 15,
            "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42,
            "Split Strategy": "Kronologis 80/20 (bukan acak — cegah data leakage)",
            "Fitur Masukan": len(meta.get("features", [])),
        }
        for k, v in info.items():
            st.markdown(f"- **{k}:** {v}")


# ── Tab 4: Gedung ─────────────────────────────────────────────────────────────
def tab_buildings(dff):
    st.markdown('<div class="section-header">Konsumsi per Gedung</div>', unsafe_allow_html=True)

    bld_avg = (dff.groupby(["Building_ID", "Building_Type"])["Energy_Usage (kWh)"]
                  .agg(["mean", "sum", "count"])
                  .reset_index()
                  .rename(columns={"mean": "Avg kWh", "sum": "Total kWh", "count": "Rekaman"}))
    bld_avg["Biaya Total (Jt)"] = bld_avg["Total kWh"] * IDR_PER_KWH / 1e6
    bld_avg = bld_avg.sort_values("Avg kWh", ascending=False)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_bld = px.bar(bld_avg.head(20), x="Building_ID", y="Avg kWh",
                         color="Building_Type",
                         color_discrete_map={"Commercial": INDIGO, "Industrial": RED,
                                             "Educational": EMERALD, "Residential": AMBER},
                         title="Rata-rata Konsumsi per Gedung (Top 20)",
                         labels={"Avg kWh": "kWh rata-rata"})
        fig_bld.add_hline(y=BASELINE_KWH, line=dict(color=AMBER, dash="dash", width=1),
                          annotation_text="Baseline")
        fig_bld.update_layout(**LAYOUT_BASE, height=380)
        st.plotly_chart(fig_bld, use_container_width=True)

    with c2:
        type_avg = dff.groupby("Building_Type")["Energy_Usage (kWh)"].mean().reset_index()
        fig_pie = px.pie(type_avg, names="Building_Type", values="Energy_Usage (kWh)",
                         color="Building_Type",
                         color_discrete_map={"Commercial": INDIGO, "Industrial": RED,
                                             "Educational": EMERALD, "Residential": AMBER},
                         title="Proporsi Konsumsi per Tipe")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                              margin=dict(t=45, b=20), height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Tabel Detail Gedung</div>', unsafe_allow_html=True)
    show_df = bld_avg.copy()
    show_df["Avg kWh"] = show_df["Avg kWh"].round(2)
    show_df["Total kWh"] = show_df["Total kWh"].round(0).astype(int)
    show_df["Biaya Total (Jt)"] = show_df["Biaya Total (Jt)"].round(2)
    st.dataframe(show_df, use_container_width=True, hide_index=True)


# ── Tab 5: Prediksi ───────────────────────────────────────────────────────────
def tab_prediction(filters, rf_model, le_building, le_occupancy, le_type):
    p = filters["pred"]
    st.markdown('<div class="section-header">Detail Prediksi Real-Time</div>', unsafe_allow_html=True)

    kwh = predict_energy(p["hour"], p["day"], p["temp"], p["hum"],
                         p["occ"], p["btype"], p["bid"], p["month"],
                         rf_model, le_building, le_occupancy, le_type)
    cost     = kwh * IDR_PER_KWH
    pct      = (kwh - BASELINE_KWH) / BASELINE_KWH * 100
    saving   = max(0, BASELINE_KWH - kwh) * IDR_PER_KWH

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Prediksi Konsumsi", f"{kwh:.1f} kWh")
    c2.metric("💰 Estimasi Biaya/jam", f"Rp {cost:,.0f}")
    c3.metric("📉 vs Baseline", f"{pct:+.1f}%", delta_color="inverse")
    c4.metric("💚 Estimasi Penghematan", f"Rp {saving:,.0f}/jam")

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=kwh,
        delta={"reference": BASELINE_KWH, "valueformat": ".1f",
               "increasing": {"color": RED}, "decreasing": {"color": EMERALD}},
        gauge={
            "axis": {"range": [50, 600], "tickcolor": SLATE, "tickfont": {"color": SLATE}},
            "bar": {"color": INDIGO, "thickness": .25},
            "bgcolor": BG_CARD,
            "borderwidth": 0,
            "steps": [
                {"range": [50, 280], "color": "rgba(16,185,129,.15)"},
                {"range": [280, 360], "color": "rgba(245,158,11,.15)"},
                {"range": [360, 600], "color": "rgba(239,68,68,.15)"},
            ],
            "threshold": {"line": {"color": AMBER, "width": 2}, "thickness": .75, "value": BASELINE_KWH},
        },
        number={"suffix": " kWh", "font": {"color": "#e2e8f0", "size": 36}},
        title={"text": "Prediksi vs Baseline", "font": {"color": SLATE, "size": 13}},
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=SLATE),
                             margin=dict(t=30, b=10), height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Hourly projection
    st.markdown('<div class="section-header">Proyeksi 24 Jam</div>', unsafe_allow_html=True)
    hours  = list(range(24))
    kwhs_h = [predict_energy(h, p["day"], p["temp"], p["hum"],
                              p["occ"], p["btype"], p["bid"], p["month"],
                              rf_model, le_building, le_occupancy, le_type)
               for h in hours]
    fig_24 = go.Figure()
    fig_24.add_scatter(x=hours, y=kwhs_h, mode="lines+markers",
                       line=dict(color=INDIGO, width=2.5),
                       marker=dict(size=5, color=INDIGO),
                       fill="tozeroy", fillcolor="rgba(99,102,241,.1)")
    fig_24.add_hline(y=BASELINE_KWH, line=dict(color=AMBER, dash="dash", width=1.2),
                     annotation_text="Baseline", annotation_font_color=AMBER)
    fig_24.update_layout(**LAYOUT_BASE,
                          title=f"Proyeksi Konsumsi 24 Jam — {p['day_label']} · {p['btype']} · {p['bid']}",
                          xaxis_title="Jam", yaxis_title="kWh", height=350,
                          xaxis=dict(tickmode="array", tickvals=list(range(0,24,2)),
                                     **{k: v for k, v in LAYOUT_BASE.get("xaxis", {}).items()
                                        if k not in ["tickmode","tickvals"]}))
    st.plotly_chart(fig_24, use_container_width=True)

    # Recommendations
    st.markdown('<div class="section-header">💡 Rekomendasi Operasional</div>', unsafe_allow_html=True)
    recs = []
    if p["hour"] >= 8 and p["hour"] <= 18 and p["day"] < 5:
        recs.append(("alert-yellow", "⚡ Jam Kerja Aktif",
                      "Tunda beban berat ke off-peak (sebelum 08:00 atau setelah 18:00)."))
    if p["temp"] > 30:
        recs.append(("alert-red", "🌡️ Suhu Tinggi",
                      "Kurangi kapasitas AC 25% di zona hunian rendah, aktifkan mode eco."))
    if p["temp"] < 20:
        recs.append(("alert-blue", "❄️ Suhu Sejuk",
                      "Manfaatkan ventilasi alami, matikan AC zona non-kritis."))
    if p["day"] >= 5:
        recs.append(("alert-green", "📅 Weekend Mode",
                      "Aktifkan Weekend Energy Mode, target hemat energi 30%."))
    if p["occ"] == "Low":
        recs.append(("alert-green", "👥 Hunian Rendah",
                      "Matikan pencahayaan dan HVAC zona tidak terpakai."))
    if not recs:
        recs.append(("alert-yellow", "✅ Kondisi Normal",
                      "Semua sistem beroperasi dalam batas wajar. Lanjutkan monitoring."))
    for cls, title, msg in recs:
        st.markdown(f'<div class="alert-box {cls}"><b>{title}</b> — {msg}</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Check files
    missing = [f for f in [DATA_PATH, META_PATH, "rf_model.pkl",
                            "le_building.pkl", "le_occupancy.pkl", "le_type.pkl"]
               if not os.path.exists(f)]
    if missing:
        st.error(f"⚠️ File berikut tidak ditemukan: **{', '.join(missing)}**\n\n"
                 "Pastikan semua file model (`.pkl`) dan data (`processed_data.csv`, "
                 "`model_metadata.json`) ada di direktori yang sama dengan `app.py`.")
        st.stop()

    df              = load_data()
    meta            = load_meta()
    rf_m, le_b, le_o, le_t = load_models()

    # Sidebar
    filters = render_sidebar(df, le_b, le_o, le_t)

    # Title
    st.markdown("""
    <div style="padding:.2rem 0 .6rem">
        <span style="font-size:1.85rem;font-weight:800;color:#e2e8f0">⚡ EnergyIQ</span>
        <span style="margin-left:.5rem;font-size:.85rem;color:#6366f1;font-weight:600;
                     letter-spacing:.08em;vertical-align:middle">SMART BUILDING ENERGY DASHBOARD</span>
    </div>
    """, unsafe_allow_html=True)

    # Filter data
    dff = apply_filters(df, filters)
    if dff.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih. Sesuaikan filter di sidebar.")
        st.stop()

    # Pre-compute prediction (for overview KPI)
    p = filters["pred"]
    kwh_pred = predict_energy(p["hour"], p["day"], p["temp"], p["hum"],
                               p["occ"], p["btype"], p["bid"], p["month"],
                               rf_m, le_b, le_o, le_t)
    pct = (kwh_pred - BASELINE_KWH) / BASELINE_KWH * 100

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 EDA", "🤖 Model", "🏢 Gedung", "🔮 Prediksi"
    ])
    with tab1: tab_overview(dff, meta, kwh_pred, pct)
    with tab2: tab_eda(dff)
    with tab3: tab_model(meta, dff, rf_m, le_b, le_o, le_t)
    with tab4: tab_buildings(dff)
    with tab5: tab_prediction(filters, rf_m, le_b, le_o, le_t)

    # Footer
    st.markdown("""
    <div class="footer">
        ⚡ EnergyIQ · Smart Building Energy Dashboard &nbsp;|&nbsp;
        Kelompok 14 · Kelas A &nbsp;|&nbsp;
        Rafif Titan Athallah &amp; Muhammad Ilhamuddin &nbsp;|&nbsp;
        EAS Pembelajaran Mesin · 2025
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
