# =============================================================================
# app.py  —  Toyota Corolla Resale Price Predictor
# Streamlit application for used-car dealerships
#
# Run locally : streamlit run app.py
# Deploy      : push to GitHub → connect on share.streamlit.io
#               set main module to "app.py"
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================================
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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
}
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stToggle > label {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7c84a0 !important;
}

/* Price card */
.price-card {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1426 100%);
    border: 1px solid #2a3050;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    text-align: center;
    margin-bottom: 1rem;
}
.price-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6c7494;
    margin-bottom: 0.4rem;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #4fc3f7;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.price-range {
    font-size: 0.8rem;
    color: #555e7a;
}

/* Metric pills */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.metric-pill {
    background: #131722;
    border: 1px solid #1e2540;
    border-radius: 10px;
    padding: 0.6rem 1.1rem;
    flex: 1;
    min-width: 100px;
}
.metric-pill-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555e7a;
    margin-bottom: 0.2rem;
}
.metric-pill-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #c8cfe8;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4fc3f7;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2540;
}

/* Influence bar */
.influence-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.45rem;
}
.influence-name {
    font-size: 0.78rem;
    color: #8891b0;
    width: 140px;
    flex-shrink: 0;
}
.influence-bar-bg {
    flex: 1;
    height: 6px;
    background: #1a2035;
    border-radius: 3px;
    overflow: hidden;
}
.influence-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #1565c0, #4fc3f7);
}
.influence-pct {
    font-size: 0.75rem;
    color: #4fc3f7;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
}

/* Divider */
hr { border-color: #1e2540; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA & MODEL  (cached so they only run once per session)
# =============================================================================

@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str = "ToyotaCorolla.csv") -> pd.DataFrame:
    drop_cols = ["Id", "Model", "Fuel_Type"]
    df = pd.read_csv(path)
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df = df.select_dtypes(include=[np.number])
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Training model…")
def train_model(df: pd.DataFrame):
    TARGET   = "Price"
    features = [c for c in df.columns if c != TARGET]
    X, y     = df[features], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Cross-validated depth selection
    best_depth, best_mae = 1, float("inf")
    for d in range(1, 16):
        scores = cross_val_score(
            DecisionTreeRegressor(max_depth=d, random_state=42),
            X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        mae = -scores.mean()
        if mae < best_mae:
            best_mae, best_depth = mae, d

    model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "best_depth" : best_depth,
        "train_mae"  : mean_absolute_error(y_train, model.predict(X_train)),
        "test_mae"   : mean_absolute_error(y_test, y_pred),
        "test_rmse"  : np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2"    : r2_score(y_test, y_pred),
        "y_test"     : y_test,
        "y_pred"     : y_pred,
    }

    importances = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    # Normalized version for EDA charts
    scaler     = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(df), columns=df.columns
    )

    return model, features, metrics, importances, normalized


# =============================================================================
# LOAD
# =============================================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "**ToyotaCorolla.csv not found.**  "
        "Make sure the file is at the root of your repository alongside `app.py`."
    )
    st.stop()

model, features, metrics, importances, normalized = train_model(df)

PRICE_MIN = int(df["Price"].min())
PRICE_MAX = int(df["Price"].max())
PRICE_MED = int(df["Price"].median())

# =============================================================================
# SIDEBAR  —  feature sliders
# =============================================================================
with st.sidebar:
    st.markdown("## 🚗 Car Specifications")
    st.markdown("Adjust the filters to configure the vehicle.")
    st.markdown("---")

    st.markdown('<p class="section-title">Core Features</p>', unsafe_allow_html=True)

    age = st.slider(
        "Age (months)", min_value=1, max_value=80,
        value=36, help="Age of the vehicle in months as of August 2004"
    )
    km = st.slider(
        "Mileage (KM)", min_value=0, max_value=250_000,
        value=60_000, step=1_000,
        help="Total kilometres driven"
    )
    hp = st.slider(
        "Horsepower (HP)", min_value=60, max_value=192,
        value=90, help="Engine horsepower"
    )
    weight = st.slider(
        "Weight (kg)", min_value=900, max_value=1_615,
        value=1_050, help="Vehicle weight in kilograms"
    )
    cc = st.slider(
        "Engine Size (cc)", min_value=1_300, max_value=2_000,
        value=1_600, step=100, help="Engine displacement in cubic centimetres"
    )
    quarterly_tax = st.slider(
        "Quarterly Tax (€)", min_value=0, max_value=300,
        value=85, help="Quarterly road tax in euros"
    )

    st.markdown("---")
    st.markdown('<p class="section-title">Comfort & Safety</p>', unsafe_allow_html=True)

    airco          = st.toggle("Air Conditioning",          value=True)
    automatic_airco= st.toggle("Automatic Air Conditioning",value=False)
    abs_           = st.toggle("ABS",                       value=True)
    airbag1        = st.toggle("Driver Airbag",             value=True)
    airbag2        = st.toggle("Passenger Airbag",          value=True)
    power_steering = st.toggle("Power Steering",            value=True)
    central_lock   = st.toggle("Central Lock",              value=True)
    powered_windows= st.toggle("Powered Windows",           value=False)
    automatic      = st.toggle("Automatic Gearbox",         value=False)

    st.markdown("---")
    st.markdown('<p class="section-title">Extras</p>', unsafe_allow_html=True)

    cd_player       = st.toggle("CD Player",               value=False)
    met_color       = st.toggle("Metallic Paint",          value=True)
    sport_model     = st.toggle("Sport Model",             value=False)
    tow_bar         = st.toggle("Tow Bar",                 value=False)
    mistlamps       = st.toggle("Fog Lamps",               value=False)
    metallic_rim    = st.toggle("Metallic Rims",           value=False)
    boardcomputer   = st.toggle("On-Board Computer",       value=False)
    backseat_divider= st.toggle("Backseat Divider",        value=True)


# =============================================================================
# BUILD INPUT VECTOR  (all other features → dataset median)
# =============================================================================
medians = df[features].median().to_dict()

car_input = {**medians, **{
    "Age_08_04"       : age,
    "KM"              : km,
    "HP"              : hp,
    "Weight"          : weight,
    "cc"              : cc,
    "Quarterly_Tax"   : quarterly_tax,
    "Airco"           : int(airco),
    "Automatic_airco" : int(automatic_airco),
    "ABS"             : int(abs_),
    "Airbag_1"        : int(airbag1),
    "Airbag_2"        : int(airbag2),
    "Power_Steering"  : int(power_steering),
    "Central_Lock"    : int(central_lock),
    "Powered_Windows" : int(powered_windows),
    "Automatic"       : int(automatic),
    "CD_Player"       : int(cd_player),
    "Met_Color"       : int(met_color),
    "Sport_Model"     : int(sport_model),
    "Tow_Bar"         : int(tow_bar),
    "Mistlamps"       : int(mistlamps),
    "Metallic_Rim"    : int(metallic_rim),
    "Boardcomputer"   : int(boardcomputer),
    "Backseat_Divider": int(backseat_divider),
}}

X_input     = pd.DataFrame([car_input])[features]
pred_price  = float(model.predict(X_input)[0])
price_pct   = (pred_price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN) * 100


# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown(
    "<h1 style='font-family:Syne;font-size:2rem;margin-bottom:0'>🚗 Toyota Corolla — Resale Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#6c7494;margin-top:0.2rem;margin-bottom:1.5rem;font-size:0.9rem'>"
    "Used-car dealership pricing tool · Decision Tree model trained on 1,436 vehicles"
    "</p>",
    unsafe_allow_html=True
)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["💰  Price Predictor", "📊  Data Explorer", "🌳  Model Insights"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRICE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: price card + gauge ──────────────────────────────────────────────
    with col_left:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Resale Price</div>
            <div class="price-value">€{pred_price:,.0f}</div>
            <div class="price-range">
                Dataset range: €{PRICE_MIN:,} – €{PRICE_MAX:,} &nbsp;|&nbsp; Median: €{PRICE_MED:,}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = pred_price,
            delta = {"reference": PRICE_MED, "valueformat": ",.0f",
                     "prefix": "€", "increasing": {"color": "#4fc3f7"},
                     "decreasing": {"color": "#ef5350"}},
            number= {"prefix": "€", "valueformat": ",.0f",
                     "font": {"size": 28, "color": "#c8cfe8", "family": "Syne"}},
            gauge = {
                "axis"      : {"range": [PRICE_MIN, PRICE_MAX],
                               "tickformat": ",.0f", "tickprefix": "€",
                               "tickcolor": "#3a4060", "tickwidth": 1,
                               "tickfont": {"color": "#555e7a", "size": 10}},
                "bar"       : {"color": "#4fc3f7", "thickness": 0.28},
                "bgcolor"   : "#0d1426",
                "borderwidth": 0,
                "steps"     : [
                    {"range": [PRICE_MIN, PRICE_MIN + (PRICE_MAX-PRICE_MIN)*0.33], "color": "#131a2e"},
                    {"range": [PRICE_MIN + (PRICE_MAX-PRICE_MIN)*0.33,
                               PRICE_MIN + (PRICE_MAX-PRICE_MIN)*0.66], "color": "#162035"},
                    {"range": [PRICE_MIN + (PRICE_MAX-PRICE_MIN)*0.66, PRICE_MAX], "color": "#1a2640"},
                ],
                "threshold" : {"line": {"color": "#ff8a65", "width": 3},
                               "thickness": 0.75, "value": PRICE_MED},
            },
            title = {"text": "Position in market range<br><span style='font-size:11px;color:#555e7a'>"
                             "Orange line = median</span>",
                     "font": {"color": "#8891b0", "size": 12}},
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="#0d1426", font_color="#c8cfe8",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Quick metrics
        diff = pred_price - PRICE_MED
        diff_str = f"+€{diff:,.0f}" if diff >= 0 else f"-€{abs(diff):,.0f}"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">
                <div class="metric-pill-label">vs. Median</div>
                <div class="metric-pill-value" style="color:{'#4fc3f7' if diff>=0 else '#ef5350'}">{diff_str}</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Market Position</div>
                <div class="metric-pill-value">{price_pct:.0f}th pct.</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Model Accuracy</div>
                <div class="metric-pill-value">R² {metrics['test_r2']:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Right: feature influence ──────────────────────────────────────────────
    with col_right:
        st.markdown('<p class="section-title">Feature Influence on This Prediction</p>',
                    unsafe_allow_html=True)
        st.caption("Importance of each feature in the trained decision tree (top 10)")

        top10 = importances.head(10)
        total = top10.sum()
        for feat, imp in top10.items():
            pct   = imp / importances.sum() * 100
            width = imp / importances.iloc[0] * 100
            label = feat.replace("_", " ").title()
            st.markdown(f"""
            <div class="influence-row">
                <span class="influence-name">{label}</span>
                <div class="influence-bar-bg">
                    <div class="influence-bar-fill" style="width:{width:.1f}%"></div>
                </div>
                <span class="influence-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<p class="section-title">Your Configuration Summary</p>',
                    unsafe_allow_html=True)

        summary_cols = st.columns(2)
        summary_items = [
            ("Age",        f"{age} months"),
            ("Mileage",    f"{km:,} km"),
            ("Horsepower", f"{hp} HP"),
            ("Weight",     f"{weight} kg"),
            ("Engine",     f"{cc} cc"),
            ("Tax/quarter",f"€{quarterly_tax}"),
            ("Airco",      "✅" if airco else "❌"),
            ("Auto. Airco",  "✅" if automatic_airco else "❌"),
            ("ABS",        "✅" if abs_ else "❌"),
            ("Airbags",    f"{'✅' if airbag1 else '❌'} / {'✅' if airbag2 else '❌'}"),
        ]
        for i, (label, val) in enumerate(summary_items):
            with summary_cols[i % 2]:
                st.markdown(
                    f"<div style='font-size:0.78rem;color:#555e7a;margin-bottom:2px'>{label}</div>"
                    f"<div style='font-size:0.92rem;color:#c8cfe8;margin-bottom:10px;font-weight:500'>{val}</div>",
                    unsafe_allow_html=True
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Explore Feature Relationships")
    st.caption("Visualize how individual features relate to sale price across the full dataset.")

    exp_col1, exp_col2 = st.columns([1, 3], gap="large")

    with exp_col1:
        continuous_features = [f for f in features if df[f].nunique() > 10]
        x_feat = st.selectbox("X-axis feature", continuous_features,
                              index=continuous_features.index("Age_08_04") if "Age_08_04" in continuous_features else 0)
        y_feat = st.selectbox("Y-axis feature", ["Price"] + continuous_features,
                              index=0)
        color_by = st.selectbox("Colour by", ["Price", "Age_08_04", "KM", "HP"],
                                index=0)
        show_trendline = st.checkbox("Show trend line", value=True)

        # Pearson r
        r = df[[x_feat, y_feat]].corr().iloc[0, 1]
        direction = "positive" if r > 0 else "negative"
        strength  = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
        st.markdown(f"""
        <div style="background:#131722;border:1px solid #1e2540;border-radius:10px;
                    padding:1rem;margin-top:1rem">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#555e7a;margin-bottom:0.3rem">
                Pearson r
            </div>
            <div style="font-family:Syne;font-size:1.8rem;font-weight:800;
                        color:{'#4fc3f7' if r>0 else '#ef5350'}">
                {r:.3f}
            </div>
            <div style="font-size:0.75rem;color:#6c7494;margin-top:0.3rem">
                {strength.capitalize()} {direction} correlation
            </div>
        </div>
        """, unsafe_allow_html=True)

    with exp_col2:
        trendline_opt = "ols" if show_trendline else None
        fig_scatter = px.scatter(
            df, x=x_feat, y=y_feat,
            color=color_by,
            color_continuous_scale="Blues",
            trendline=trendline_opt,
            opacity=0.55,
            labels={x_feat: x_feat.replace("_", " "), y_feat: y_feat.replace("_", " ")},
            title=f"{x_feat.replace('_',' ')} vs {y_feat.replace('_',' ')}",
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(
            height=420,
            paper_bgcolor="#0d1426",
            plot_bgcolor="#0d1426",
            font_color="#c8cfe8",
            title_font=dict(family="Syne", size=14, color="#c8cfe8"),
            coloraxis_colorbar=dict(tickfont=dict(color="#8891b0")),
            xaxis=dict(gridcolor="#1a2035", linecolor="#1e2540"),
            yaxis=dict(gridcolor="#1a2035", linecolor="#1e2540"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### Price Distribution by Feature")

    dist_col1, dist_col2 = st.columns([1, 3], gap="large")
    with dist_col1:
        binary_features = [c for c in features if df[c].nunique() == 2]
        box_feat = st.selectbox("Group box plot by", binary_features,
                                index=binary_features.index("Airco") if "Airco" in binary_features else 0)
    with dist_col2:
        fig_box = px.box(
            df, x=box_feat, y="Price",
            color=box_feat,
            color_discrete_sequence=["#1565c0", "#4fc3f7"],
            labels={box_feat: box_feat.replace("_", " "), "Price": "Price (€)"},
            title=f"Price Distribution — {box_feat.replace('_', ' ')} present vs. absent",
            points="outliers",
        )
        fig_box.update_layout(
            height=380,
            paper_bgcolor="#0d1426",
            plot_bgcolor="#0d1426",
            font_color="#c8cfe8",
            title_font=dict(family="Syne", size=13, color="#c8cfe8"),
            showlegend=False,
            xaxis=dict(gridcolor="#1a2035", linecolor="#1e2540",
                       tickvals=[0, 1], ticktext=["Absent (0)", "Present (1)"]),
            yaxis=dict(gridcolor="#1a2035", linecolor="#1e2540"),
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE",  f"€{metrics['test_mae']:,.0f}",  help="Mean Absolute Error on test set")
    m2.metric("Test RMSE", f"€{metrics['test_rmse']:,.0f}", help="Root Mean Squared Error on test set")
    m3.metric("R² Score",  f"{metrics['test_r2']:.4f}",     help="Proportion of variance explained")
    m4.metric("Tree Depth",f"{metrics['best_depth']}",      help="Optimal depth found by 5-fold CV")

    ins_col1, ins_col2 = st.columns([1, 1], gap="large")

    with ins_col1:
        st.markdown("#### Feature Importances (Top 15)")
        top15 = importances.head(15).reset_index()
        top15.columns = ["Feature", "Importance"]
        top15["Feature"] = top15["Feature"].str.replace("_", " ").str.title()

        fig_imp = px.bar(
            top15[::-1], x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#1565c0", "#4fc3f7"],
        )
        fig_imp.update_layout(
            height=480, showlegend=False,
            paper_bgcolor="#0d1426", plot_bgcolor="#0d1426",
            font_color="#c8cfe8",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1a2035", linecolor="#1e2540"),
            yaxis=dict(gridcolor="#1a2035", linecolor="#1e2540"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with ins_col2:
        st.markdown("#### Predicted vs Actual (Test Set)")
        y_test_vals  = metrics["y_test"].values
        y_pred_vals  = metrics["y_pred"]

        fig_pva = go.Figure()
        fig_pva.add_trace(go.Scatter(
            x=y_test_vals, y=y_pred_vals,
            mode="markers",
            marker=dict(color="#4fc3f7", size=5, opacity=0.5),
            name="Predictions",
        ))
        lims = [PRICE_MIN - 500, PRICE_MAX + 500]
        fig_pva.add_trace(go.Scatter(
            x=lims, y=lims,
            mode="lines",
            line=dict(color="#ef5350", dash="dash", width=1.5),
            name="Perfect prediction",
        ))
        fig_pva.update_layout(
            height=480,
            paper_bgcolor="#0d1426", plot_bgcolor="#0d1426",
            font_color="#c8cfe8",
            xaxis=dict(title="Actual Price (€)", gridcolor="#1a2035",
                       linecolor="#1e2540", tickformat=",.0f"),
            yaxis=dict(title="Predicted Price (€)", gridcolor="#1a2035",
                       linecolor="#1e2540", tickformat=",.0f"),
            legend=dict(bgcolor="#131722", bordercolor="#1e2540", borderwidth=1),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pva, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Residuals Distribution")

    residuals = y_test_vals - y_pred_vals
    fig_res = go.Figure()
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=35,
        marker_color="#4fc3f7", opacity=0.8,
        name="Residuals",
    ))
    fig_res.add_vline(x=0, line_color="#ef5350", line_dash="dash", line_width=2)
    fig_res.update_layout(
        height=280,
        paper_bgcolor="#0d1426", plot_bgcolor="#0d1426",
        font_color="#c8cfe8",
        xaxis=dict(title="Residual (€)  [Actual − Predicted]",
                   gridcolor="#1a2035", linecolor="#1e2540", tickformat=",.0f"),
        yaxis=dict(title="Count", gridcolor="#1a2035", linecolor="#1e2540"),
        margin=dict(l=10, r=10, t=10, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_res, use_container_width=True)
    st.caption(
        f"Mean residual: €{residuals.mean():,.0f} (ideal = 0)  |  "
        f"Std: €{residuals.std():,.0f}  |  "
        f"Train MAE: €{metrics['train_mae']:,.0f}  |  Test MAE: €{metrics['test_mae']:,.0f}"
    )
