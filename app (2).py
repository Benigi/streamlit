# =============================================================================
# app.py  —  Toyota Corolla Resale Price Predictor
# Self-contained Streamlit application — no other local imports needed
#
# Run locally : streamlit run app.py
# Deploy      : push app.py + ToyotaCorolla.csv + requirements.txt to GitHub
#               On share.streamlit.io → set Main file path to: app.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Must be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="Corolla Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
h1, h2, h3                  { font-family: 'Syne', sans-serif !important; }

section[data-testid="stSidebar"]   { background: #0f1117; }
section[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

.price-card {
    background: linear-gradient(135deg,#1a1f35 0%,#0d1426 100%);
    border: 1px solid #2a3050; border-radius: 16px;
    padding: 2rem 2.4rem; text-align: center; margin-bottom: 1rem;
}
.price-label {
    font-family:'Syne',sans-serif; font-size:.75rem; font-weight:700;
    letter-spacing:.15em; text-transform:uppercase; color:#6c7494; margin-bottom:.4rem;
}
.price-value {
    font-family:'Syne',sans-serif; font-size:3.2rem; font-weight:800;
    color:#4fc3f7; line-height:1; margin-bottom:.5rem;
}
.price-range { font-size:.8rem; color:#555e7a; }

.section-title {
    font-family:'Syne',sans-serif; font-size:.7rem; font-weight:700;
    letter-spacing:.18em; text-transform:uppercase; color:#4fc3f7;
    margin:1.4rem 0 .6rem 0; padding-bottom:.4rem;
    border-bottom:1px solid #1e2540;
}
.influence-row    { display:flex; align-items:center; gap:.8rem; margin-bottom:.45rem; }
.influence-name   { font-size:.78rem; color:#8891b0; width:150px; flex-shrink:0; }
.influence-bar-bg { flex:1; height:6px; background:#1a2035; border-radius:3px; overflow:hidden; }
.influence-bar-fill { height:100%; border-radius:3px;
                      background:linear-gradient(90deg,#1565c0,#4fc3f7); }
.influence-pct    { font-size:.75rem; color:#4fc3f7; width:38px;
                    text-align:right; flex-shrink:0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1426",
    plot_bgcolor="#0d1426",
    font_color="#c8cfe8",
)
GRID = dict(gridcolor="#1a2035", linecolor="#1e2540")


def _chart(fig, height=400):
    """Render Plotly figure — compatible with both old and new Streamlit APIs."""
    fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")           # Streamlit ≥ 1.45
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)  # fallback


def _trendline(x_vals, y_vals):
    """Numpy-only OLS trendline — avoids statsmodels dependency entirely."""
    mask   = np.isfinite(x_vals) & np.isfinite(y_vals)
    coeffs = np.polyfit(x_vals[mask], y_vals[mask], 1)
    xs     = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 300)
    return xs, np.poly1d(coeffs)(xs)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv("ToyotaCorolla.csv")
    df.drop(columns=[c for c in ["Id", "Model", "Fuel_Type"] if c in df.columns],
            inplace=True)
    df = df.select_dtypes(include=[np.number])
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


# =============================================================================
# MODEL TRAINING
# =============================================================================
@st.cache_resource(show_spinner="Training model…")
def train(_df):
    TARGET   = "Price"
    features = [c for c in _df.columns if c != TARGET]
    X, y     = _df[features], _df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    best_d, best_mae = 1, float("inf")
    for d in range(1, 16):
        s = cross_val_score(
            DecisionTreeRegressor(max_depth=d, random_state=42),
            X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error",
        )
        if -s.mean() < best_mae:
            best_mae, best_d = -s.mean(), d

    model = DecisionTreeRegressor(max_depth=best_d, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    metrics = dict(
        depth     = best_d,
        train_mae = mean_absolute_error(y_tr, model.predict(X_tr)),
        test_mae  = mean_absolute_error(y_te, y_pred),
        test_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred))),
        test_r2   = float(r2_score(y_te, y_pred)),
        y_test    = y_te,
        y_pred    = y_pred,
    )
    importances = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    return model, features, metrics, importances


# =============================================================================
# LOAD & TRAIN
# =============================================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "**`ToyotaCorolla.csv` not found.**  "
        "Make sure it is at the root of your GitHub repo alongside `app.py`."
    )
    st.stop()

model, features, metrics, importances = train(df)

PRICE_MIN = int(df["Price"].min())
PRICE_MAX = int(df["Price"].max())
PRICE_MED = int(df["Price"].median())

# =============================================================================
# SIDEBAR — adjustable filters
# =============================================================================
with st.sidebar:
    st.markdown("## 🚗 Vehicle Specifications")
    st.caption("Adjust the filters · price updates instantly")
    st.divider()

    st.markdown("**Core features**")
    age    = st.slider("Age (months)",        1,    80,   36)
    km     = st.slider("Mileage (km)",        0, 250_000, 60_000, step=1_000)
    hp     = st.slider("Horsepower (HP)",    60,   192,   90)
    weight = st.slider("Weight (kg)",       900,  1615, 1050)
    cc     = st.slider("Engine (cc)",      1300,  2000, 1600, step=100)
    tax    = st.slider("Quarterly tax (€)",   0,   300,   85)

    st.divider()
    st.markdown("**Comfort & Safety**")
    airco        = st.toggle("Air conditioning",           value=True)
    auto_airco   = st.toggle("Automatic air conditioning", value=False)
    abs_b        = st.toggle("ABS",                        value=True)
    airbag1      = st.toggle("Driver airbag",              value=True)
    airbag2      = st.toggle("Passenger airbag",           value=True)
    pwr_steer    = st.toggle("Power steering",             value=True)
    central_lock = st.toggle("Central lock",               value=True)
    pwr_windows  = st.toggle("Powered windows",            value=False)
    automatic    = st.toggle("Automatic gearbox",          value=False)

    st.divider()
    st.markdown("**Extras**")
    cd_player  = st.toggle("CD player",         value=False)
    met_color  = st.toggle("Metallic paint",    value=True)
    sport      = st.toggle("Sport model",       value=False)
    tow_bar    = st.toggle("Tow bar",           value=False)
    mistlamps  = st.toggle("Fog lamps",         value=False)
    met_rim    = st.toggle("Metallic rims",     value=False)
    boardcomp  = st.toggle("On-board computer", value=False)
    backseat   = st.toggle("Backseat divider",  value=True)


# =============================================================================
# BUILD PREDICTION INPUT VECTOR
# =============================================================================
car = df[features].median().to_dict()
car.update({
    "Age_08_04":        age,
    "KM":               km,
    "HP":               hp,
    "Weight":           weight,
    "cc":               cc,
    "Quarterly_Tax":    tax,
    "Airco":            int(airco),
    "Automatic_airco":  int(auto_airco),
    "ABS":              int(abs_b),
    "Airbag_1":         int(airbag1),
    "Airbag_2":         int(airbag2),
    "Power_Steering":   int(pwr_steer),
    "Central_Lock":     int(central_lock),
    "Powered_Windows":  int(pwr_windows),
    "Automatic":        int(automatic),
    "CD_Player":        int(cd_player),
    "Met_Color":        int(met_color),
    "Sport_Model":      int(sport),
    "Tow_Bar":          int(tow_bar),
    "Mistlamps":        int(mistlamps),
    "Metallic_Rim":     int(met_rim),
    "Boardcomputer":    int(boardcomp),
    "Backseat_Divider": int(backseat),
})

pred = float(model.predict(pd.DataFrame([car])[features])[0])
diff = pred - PRICE_MED

# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown(
    "<h1 style='font-family:Syne;font-size:2rem;margin-bottom:0'>"
    "🚗 Toyota Corolla — Resale Price Predictor</h1>",
    unsafe_allow_html=True,
)
st.caption("Used-car dealership pricing tool · Decision Tree · 1,436 vehicles")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["💰 Price Predictor", "📊 Data Explorer", "🌳 Model Insights"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRICE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Resale Price</div>
            <div class="price-value">€{pred:,.0f}</div>
            <div class="price-range">
                Range: €{PRICE_MIN:,} – €{PRICE_MAX:,} &nbsp;·&nbsp; Median: €{PRICE_MED:,}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = pred,
            delta = {
                "reference": PRICE_MED, "valueformat": ",.0f", "prefix": "€",
                "increasing": {"color": "#4fc3f7"},
                "decreasing": {"color": "#ef5350"},
            },
            number = {"prefix": "€", "valueformat": ",.0f",
                      "font": {"size": 28, "color": "#c8cfe8", "family": "Syne"}},
            gauge  = {
                "axis": {
                    "range": [PRICE_MIN, PRICE_MAX],
                    "tickformat": ",.0f", "tickprefix": "€",
                    "tickfont": {"color": "#555e7a", "size": 9},
                },
                "bar": {"color": "#4fc3f7", "thickness": 0.28},
                "bgcolor": "#0d1426", "borderwidth": 0,
                "steps": [
                    {"range": [PRICE_MIN,
                               PRICE_MIN + (PRICE_MAX - PRICE_MIN) * .33],
                     "color": "#131a2e"},
                    {"range": [PRICE_MIN + (PRICE_MAX - PRICE_MIN) * .33,
                               PRICE_MIN + (PRICE_MAX - PRICE_MIN) * .66],
                     "color": "#162035"},
                    {"range": [PRICE_MIN + (PRICE_MAX - PRICE_MIN) * .66,
                               PRICE_MAX],
                     "color": "#1a2640"},
                ],
                "threshold": {
                    "line": {"color": "#ff8a65", "width": 3},
                    "thickness": .75, "value": PRICE_MED,
                },
            },
            title = {"text": "Market position · orange line = median",
                     "font": {"color": "#8891b0", "size": 11}},
        ))
        fig_g.update_layout(**DARK_LAYOUT, margin=dict(t=40, b=10, l=20, r=20))
        _chart(fig_g, height=270)

        pct  = (pred - PRICE_MIN) / (PRICE_MAX - PRICE_MIN) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("vs Median",
                  f"+€{diff:,.0f}" if diff >= 0 else f"-€{abs(diff):,.0f}",
                  delta_color="normal" if diff >= 0 else "inverse")
        c2.metric("Market %ile", f"{pct:.0f}th")
        c3.metric("R² (model)",  f"{metrics['test_r2']:.2f}")

    with right:
        st.markdown('<p class="section-title">Feature Influence (decision tree)</p>',
                    unsafe_allow_html=True)
        for feat, imp in importances.head(10).items():
            pct_imp = imp / importances.sum() * 100
            width   = imp / importances.iloc[0] * 100
            label   = feat.replace("_", " ").title()
            st.markdown(f"""
            <div class="influence-row">
                <span class="influence-name">{label}</span>
                <div class="influence-bar-bg">
                    <div class="influence-bar-fill" style="width:{width:.1f}%"></div>
                </div>
                <span class="influence-pct">{pct_imp:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown('<p class="section-title">Your Configuration</p>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        items = [
            ("Age",        f"{age} months"),
            ("Mileage",    f"{km:,} km"),
            ("Horsepower", f"{hp} HP"),
            ("Weight",     f"{weight} kg"),
            ("Engine",     f"{cc} cc"),
            ("Tax / qtr",  f"€{tax}"),
            ("Airco",      "✅" if airco else "❌"),
            ("Auto Airco", "✅" if auto_airco else "❌"),
            ("ABS",        "✅" if abs_b else "❌"),
            ("Airbags",    f"{'✅' if airbag1 else '❌'} / {'✅' if airbag2 else '❌'}"),
        ]
        for i, (lbl, val) in enumerate(items):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(
                    f"<div style='font-size:.75rem;color:#555e7a'>{lbl}</div>"
                    f"<div style='font-size:.92rem;color:#c8cfe8;margin-bottom:8px'>{val}</div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Explore Feature Relationships")
    continuous = [f for f in features if df[f].nunique() > 10]

    ea, eb = st.columns([1, 3], gap="large")
    with ea:
        x_feat = st.selectbox(
            "X-axis", continuous,
            index=continuous.index("Age_08_04") if "Age_08_04" in continuous else 0,
        )
        y_feat     = st.selectbox("Y-axis", ["Price"] + continuous, index=0)
        show_trend = st.checkbox("Show trend line", value=True)

        r         = df[[x_feat, y_feat]].corr().iloc[0, 1]
        strength  = "strong" if abs(r) > .5 else ("moderate" if abs(r) > .3 else "weak")
        direction = "positive" if r > 0 else "negative"
        st.markdown(f"""
        <div style="background:#131722;border:1px solid #1e2540;border-radius:10px;
                    padding:1rem;margin-top:1rem">
            <div style="font-size:.68rem;font-weight:700;letter-spacing:.1em;
                        text-transform:uppercase;color:#555e7a;margin-bottom:.3rem">
                Pearson r
            </div>
            <div style="font-family:Syne;font-size:1.8rem;font-weight:800;
                        color:{'#4fc3f7' if r > 0 else '#ef5350'}">
                {r:.3f}
            </div>
            <div style="font-size:.75rem;color:#6c7494;margin-top:.3rem">
                {strength.capitalize()} {direction} correlation
            </div>
        </div>""", unsafe_allow_html=True)

    with eb:
        x_vals = df[x_feat].to_numpy(dtype=float)
        y_vals = df[y_feat].to_numpy(dtype=float)

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(
                color=df["Price"].values, colorscale="Blues",
                size=5, opacity=.55,
                colorbar=dict(title="Price (€)",
                              tickfont=dict(color="#8891b0")),
            ),
            name="Data points",
        ))
        if show_trend:
            xs_t, ys_t = _trendline(x_vals, y_vals)
            fig_s.add_trace(go.Scatter(
                x=xs_t, y=ys_t, mode="lines",
                line=dict(color="#ff8a65", width=2, dash="dash"),
                name="Trend",
            ))
        fig_s.update_layout(
            **DARK_LAYOUT,
            xaxis=dict(title=x_feat.replace("_", " "), **GRID),
            yaxis=dict(title=y_feat.replace("_", " "), **GRID),
            legend=dict(bgcolor="#131722", bordercolor="#1e2540", borderwidth=1),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        _chart(fig_s, height=400)

    st.divider()
    st.markdown("### Price Distribution by Feature")

    ba, bb = st.columns([1, 3], gap="large")
    binary_feats = [c for c in features if df[c].nunique() == 2]
    with ba:
        box_f = st.selectbox(
            "Group by", binary_feats,
            index=binary_feats.index("Airco") if "Airco" in binary_feats else 0,
        )
    with bb:
        fig_b = px.box(
            df, x=box_f, y="Price",
            color=box_f,
            color_discrete_sequence=["#1565c0", "#4fc3f7"],
            points="outliers",
            labels={box_f: box_f.replace("_", " "), "Price": "Price (€)"},
        )
        fig_b.update_layout(
            **DARK_LAYOUT, showlegend=False,
            xaxis=dict(**GRID, tickvals=[0, 1],
                       ticktext=["Absent (0)", "Present (1)"]),
            yaxis=dict(**GRID),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        _chart(fig_b, height=360)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE",   f"€{metrics['test_mae']:,.0f}")
    m2.metric("Test RMSE",  f"€{metrics['test_rmse']:,.0f}")
    m3.metric("R² Score",   f"{metrics['test_r2']:.4f}")
    m4.metric("Tree Depth", str(metrics["depth"]))

    ia, ib = st.columns(2, gap="large")

    with ia:
        st.markdown("#### Feature Importances (Top 15)")
        top15 = importances.head(15).reset_index()
        top15.columns = ["Feature", "Importance"]
        top15["Feature"] = top15["Feature"].str.replace("_", " ").str.title()
        fig_i = px.bar(
            top15[::-1], x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale=["#1565c0", "#4fc3f7"],
        )
        fig_i.update_layout(
            **DARK_LAYOUT,
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(**GRID), yaxis=dict(**GRID),
            margin=dict(l=5, r=5, t=5, b=5),
        )
        _chart(fig_i, height=460)

    with ib:
        st.markdown("#### Predicted vs Actual (Test Set)")
        y_te = metrics["y_test"].values
        y_pr = metrics["y_pred"]

        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(
            x=y_te, y=y_pr, mode="markers",
            marker=dict(color="#4fc3f7", size=5, opacity=.5),
            name="Predictions",
        ))
        lm = [PRICE_MIN - 500, PRICE_MAX + 500]
        fig_pa.add_trace(go.Scatter(
            x=lm, y=lm, mode="lines",
            line=dict(color="#ef5350", dash="dash", width=1.5),
            name="Perfect prediction",
        ))
        fig_pa.update_layout(
            **DARK_LAYOUT,
            xaxis=dict(title="Actual Price (€)",    tickformat=",.0f", **GRID),
            yaxis=dict(title="Predicted Price (€)", tickformat=",.0f", **GRID),
            legend=dict(bgcolor="#131722", bordercolor="#1e2540", borderwidth=1),
            margin=dict(l=5, r=5, t=5, b=5),
        )
        _chart(fig_pa, height=460)

    st.divider()
    st.markdown("#### Residuals Distribution")
    residuals = y_te - y_pr
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(
        x=residuals, nbinsx=35,
        marker_color="#4fc3f7", opacity=.8,
    ))
    fig_r.add_vline(x=0, line_color="#ef5350", line_dash="dash", line_width=2)
    fig_r.update_layout(
        **DARK_LAYOUT,
        xaxis=dict(title="Residual (€)  [Actual − Predicted]",
                   tickformat=",.0f", **GRID),
        yaxis=dict(title="Count", **GRID),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    _chart(fig_r, height=280)
    st.caption(
        f"Mean residual: €{residuals.mean():,.0f} (ideal = 0)  ·  "
        f"Std: €{residuals.std():,.0f}  ·  "
        f"Train MAE: €{metrics['train_mae']:,.0f}  ·  "
        f"Test MAE: €{metrics['test_mae']:,.0f}"
    )
