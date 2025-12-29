# src/app.py
# Run from repo root:
#   conda activate crash-app
#   streamlit run src/app.py

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Severe Injury Risk Predictor",
    page_icon="🚗",
    layout="centered",
)

# ----------------------------
# Paths
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "severe_injury_model.joblib"

# ----------------------------
# Optional SHAP (local explanation)
# ----------------------------
SHAP_AVAILABLE = False
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ----------------------------
# Code -> label mappings (CRSS-style)
# NOTE: If your dataset uses slightly different codes, adjust these maps.
# ----------------------------
SEX_IM_MAP = {
    1: "Male",
    2: "Female",
    8: "Not reported",
    9: "Unknown / Reported as unknown",
}

ALCOHOL_MAP = {
    1: "Alcohol involved",
    2: "No alcohol involved",
    8: "No applicable person",
    9: "Unknown",
}

DRUGS_MAP = {
    0: "No (not involved)",
    1: "Yes (involved)",
    8: "Not reported",
    9: "Unknown / Reported as unknown",
}

SPEEDREL_MAP = {
    0: "No",
    2: "Yes, racing",
    3: "Yes, exceeded speed limit",
    4: "Yes, too fast for conditions",
    5: "Yes, specifics unknown",
    8: "No driver present / Unknown if driver present",
    9: "Unknown / Reported as unknown",
}

DAY_WEEK_MAP = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}

# Light condition (common CRSS/GES-style codes; tweak if needed)
LGTCON_IM_MAP = {
    1: "Daylight",
    2: "Dark (not lighted)",
    3: "Dark (lighted)",
    4: "Dawn",
    5: "Dusk",
    6: "Other",
    8: "Not reported",
    9: "Unknown",
}

# Weather (your notebook uses WEATHR_IM)
WEATHR_IM_MAP = {
    1: "Clear",
    2: "Rain",
    3: "Sleet / Hail",
    4: "Snow",
    5: "Fog / Smog / Smoke",
    6: "Severe crosswinds",
    7: "Blowing sand/soil/dirt",
    8: "Other",
    10: "Cloudy",
    98: "Not reported",
    99: "Unknown / Reported as unknown",
}

# Region (you used REGION in modeling; adjust labels if your coding differs)
REGION_MAP = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West",
    8: "Not reported",
    9: "Unknown",
}

REL_ROAD_MAP = {
    1: "On roadway",
    2: "On shoulder",
    3: "On median",
    4: "On roadside",
    5: "Outside trafficway",
    6: "Off roadway – location unknown",
    7: "In parking lane/zone",
    8: "Gore",
    10: "Separator",
    11: "Continuous left turn lane",
    12: "Pedestrian refuge island / Traffic island",
    98: "Not reported",
    99: "Unknown / Reported as unknown",
}

# Relation to junction (common CRSS-style; tweak if needed)
RELJCT2_IM_MAP = {
    0: "Non-junction",
    1: "Intersection",
    2: "Intersection-related",
    3: "Driveway / Alley access",
    4: "Ramp-related",
    5: "Other junction",
    8: "Not reported",
    9: "Unknown",
}

# Restraint use (your table had REST_USE_20 prominent; keep 20)
REST_USE_MAP = {
    0: "Not applicable",
    1: "Shoulder belt only used",
    2: "Lap belt only used",
    3: "Shoulder and lap belt used",
    6: "Racing-style harness used",
    7: "None used",
    8: "Restraint used – type unknown",
    10: "Child restraint – forward facing",
    11: "Child restraint – rear facing",
    12: "Booster seat",
    16: "Helmet, other than DOT-compliant motorcycle helmet",
    17: "No helmet",
    19: "Helmet, unknown DOT-compliance",
    20: "None used / Not applicable",
    96: "Not a motor vehicle occupant",
    97: "Other",
    98: "Not reported",
    99: "Unknown / Reported as unknown",
}

# Vehicle type group helper (simple, optional)
BODY_TYP_GROUP_MAP = {
    1: "Passenger Cars (01–11, 17)",
    14: "Light Trucks & Vans (14–16, 19–22, 28–41, 45–49)",
    50: "Buses (50–59)",
    60: "Large Trucks (60–64, 66, 67, 71, 72, 78)",
    80: "Motorcycles (80–89)",
    98: "Other / Unknown Vehicles",
    99: "Unknown (98/99)",
}

# ----------------------------
# Helpers
# ----------------------------
def select_from_mapping(label: str, mapping: dict, default_key: int, help_text: str | None = None):
    items = [(k, mapping[k]) for k in sorted(mapping.keys())]
    labels = [v for _, v in items]
    codes = [k for k, _ in items]

    default_idx = codes.index(default_key) if default_key in codes else 0
    chosen_label = st.selectbox(label, labels, index=default_idx, help=help_text)
    return codes[labels.index(chosen_label)]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def _find_preprocess_step(model_pipeline):
    for k in model_pipeline.named_steps.keys():
        if k.startswith("preprocess"):
            return k, model_pipeline.named_steps[k]
    return None, None

@st.cache_resource
def make_shap_explainer(_model_pipeline):
    if not SHAP_AVAILABLE:
        return None, None, None

    pre_key, pre = _find_preprocess_step(_model_pipeline)
    mdl = _model_pipeline.named_steps.get("model", None)
    if pre is None or mdl is None:
        return None, None, None

    feat_names = pre.get_feature_names_out()

    try:
        explainer = shap.Explainer(mdl)
        return explainer, feat_names, "explainer_call"
    except Exception:
        pass

    try:
        explainer = shap.TreeExplainer(mdl)
        return explainer, feat_names, "treeexplainer_shap_values"
    except Exception:
        return None, feat_names, None

def shap_local_bar(explainer, feat_names, mode, model_pipeline, x_row_df, top_n=12):
    if explainer is None or not SHAP_AVAILABLE:
        return None

    pre_key, pre = _find_preprocess_step(model_pipeline)
    if pre is None:
        return None

    Xt = pre.transform(x_row_df)
    Xd = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    try:
        if mode == "explainer_call":
            exp = explainer(Xd)
            vals = exp.values
            if getattr(vals, "ndim", 0) == 3:
                cls_idx = 1 if vals.shape[2] > 1 else vals.shape[2] - 1
                sv = vals[0, :, cls_idx]
            else:
                sv = vals[0, :]
        else:
            sv = explainer.shap_values(Xd)
            if isinstance(sv, list) and len(sv) > 1:
                sv = sv[1]
            sv = sv[0]

        s = pd.Series(sv, index=feat_names).sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(7, 5))
        s[::-1].plot(kind="barh", ax=ax)
        ax.set_title("Local explanation (this input)")
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_ylabel("")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.info(f"SHAP couldn't compute values for this model/input. Details: {e}")
        return None

# ----------------------------
# Model card / info panel
# ----------------------------
def render_model_card(threshold: float):
    with st.expander("📌 Model card (what this is / what it isn’t)", expanded=False):
        st.markdown(
            """
**What the model does**
- Predicts the probability that a crash results in **severe injury** (incapacitated or fatal), based on crash report variables.
- Output is a **risk score** (0–1). It is **not** a diagnosis or certainty.

**How to use the threshold**
- **HIGH risk** if `p ≥ threshold`, else **LOW risk**.
- Lower threshold → higher recall (catch more severe cases) but more false alarms.
- Higher threshold → fewer false alarms but more missed severe cases.

**Important notes**
- This model is **predictive**, not causal. Features can be signals of crash intensity and context.
- Predictions are only as reliable as the inputs (Unknown/Not reported values reduce confidence).
            """
        )
        st.markdown(f"**Current threshold:** `{threshold:.2f}`")

# ----------------------------
# App header + model load
# ----------------------------
st.title("🚗 Severe Injury Risk Predictor")
st.caption("Predicts the probability that a crash results in severe injury (incapacitated or fatal).")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Put your trained pipeline joblib here, or update MODEL_PATH.")
    st.stop()

model = load_model()

# ----------------------------
# Threshold policy
# ----------------------------
st.subheader("Decision Policy")
threshold = st.slider(
    "Decision threshold (lower = higher recall, more false alarms)",
    min_value=0.01,
    max_value=0.99,
    value=0.25,
    step=0.01,
)

render_model_card(threshold)

# ----------------------------
# Inputs (MATCH YOUR TRAINING FEATURES)
# ----------------------------
st.subheader("Inputs")

c1, c2 = st.columns(2)

with c1:
    AGE_IM = st.number_input("Driver age (AGE_IM)", min_value=0, max_value=120, value=30)
    SEX_IM = select_from_mapping("Sex (SEX_IM)", SEX_IM_MAP, default_key=1)

    ALCOHOL = select_from_mapping("Alcohol involved (ALCOHOL)", ALCOHOL_MAP, default_key=2)
    DRUGS = select_from_mapping("Drugs involved (DRUGS)", DRUGS_MAP, default_key=0)

    HOUR_acc = st.number_input("Crash hour (HOUR_acc)", min_value=0, max_value=23, value=12)
    MONTH_acc = st.number_input("Month (MONTH_acc)", min_value=1, max_value=12, value=6)

    NUMOCCS = st.number_input("Number of occupants (NUMOCCS)", min_value=0, max_value=20, value=1)

with c2:
    DAY_WEEK = select_from_mapping("Day of week (DAY_WEEK)", DAY_WEEK_MAP, default_key=3)
    LGTCON_IM = select_from_mapping("Light condition (LGTCON_IM)", LGTCON_IM_MAP, default_key=1)
    WEATHR_IM = select_from_mapping("Weather (WEATHR_IM)", WEATHR_IM_MAP, default_key=1)
    REGION = select_from_mapping("Region (REGION)", REGION_MAP, default_key=1)

    REL_ROAD = select_from_mapping("Relation to roadway (REL_ROAD)", REL_ROAD_MAP, default_key=1)
    RELJCT2_IM = select_from_mapping("Relation to junction (RELJCT2_IM)", RELJCT2_IM_MAP, default_key=0)
    SPEEDREL = select_from_mapping("Speed-related (SPEEDREL)", SPEEDREL_MAP, default_key=0)

    REST_USE = select_from_mapping(
        "Restraint use (REST_USE)",
        REST_USE_MAP,
        default_key=3,
        help_text="Predictive signal, not causal. Lower use is often associated with higher severity."
    )

st.divider()

# Vehicle type
st.subheader("Vehicle Type (BODY_TYP)")

group_code = select_from_mapping(
    "Vehicle type group (based on BODY_TYP ranges)",
    BODY_TYP_GROUP_MAP,
    default_key=1
)

with st.expander("Advanced: use exact BODY_TYP code instead of group"):
    st.caption("If you know the exact BODY_TYP code from your dataset, enter it here.")
    exact_body_typ = st.number_input("Exact BODY_TYP code", min_value=0, max_value=999, value=int(group_code))
    use_exact = st.checkbox("Use exact BODY_TYP code", value=False)

BODY_TYP = int(exact_body_typ) if use_exact else int(group_code)

# ----------------------------
# Build input row (MUST match training columns exactly)
# ----------------------------
x = pd.DataFrame([{
    "AGE_IM": AGE_IM,
    "SEX_IM": SEX_IM,
    "ALCOHOL": ALCOHOL,
    "DRUGS": DRUGS,
    "HOUR_acc": HOUR_acc,
    "MONTH_acc": MONTH_acc,
    "DAY_WEEK": DAY_WEEK,
    "LGTCON_IM": LGTCON_IM,
    "WEATHR_IM": WEATHR_IM,
    "REGION": REGION,
    "REL_ROAD": REL_ROAD,
    "RELJCT2_IM": RELJCT2_IM,
    "BODY_TYP": BODY_TYP,
    "REST_USE": REST_USE,
    "SPEEDREL": SPEEDREL,
    "NUMOCCS": NUMOCCS,
}])

# ----------------------------
# Guardrail 1: column drift check + stable ordering
# ----------------------------
EXPECTED_COLS = [
    "AGE_IM",
    "SEX_IM",
    "ALCOHOL",
    "DRUGS",
    "HOUR_acc",
    "MONTH_acc",
    "NUMOCCS",
    "DAY_WEEK",
    "LGTCON_IM",
    "WEATHR_IM",
    "REGION",
    "REL_ROAD",
    "RELJCT2_IM",
    "BODY_TYP",
    "REST_USE",
    "SPEEDREL",
]

missing = [c for c in EXPECTED_COLS if c not in x.columns]
extra = [c for c in x.columns if c not in EXPECTED_COLS]
if missing or extra:
    st.error(f"Column mismatch. Missing={missing} Extra={extra}")
    st.stop()

x = x[EXPECTED_COLS]

# ----------------------------
# Guardrail 2: warn if too many unknowns
# ----------------------------
UNKNOWN_CODES = {8, 9, 98, 99}
unknown_rate = float(x.iloc[0].isin(list(UNKNOWN_CODES)).mean())
if unknown_rate > 0.4:
    st.warning("Many inputs are Unknown/Not reported. Prediction may be less reliable.")

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict", type="primary"):
    proba = float(model.predict_proba(x)[:, 1][0])
    pred = int(proba >= threshold)

    st.subheader("Prediction")
    st.metric("Predicted probability (0–1)", f"{proba:.3f}")
    st.write(f"**As a percentage:** {proba * 100:.1f}%")

    st.caption(
        "HIGH/LOW is a decision label based on the threshold above (not a guarantee). "
        "It flags whether the predicted probability crosses your chosen cutoff."
    )

    if pred == 1:
        st.error(f"⚠️ HIGH risk (p ≥ {threshold:.2f})")
    else:
        st.success(f"✅ LOW risk (p < {threshold:.2f})")

    with st.expander("Show raw model input (codes)"):
        st.dataframe(x)

    # ----------------------------
    # Local explanation (SHAP)
    # ----------------------------
    st.subheader("Why this prediction")
    st.caption(
        "SHAP is a local explanation for THIS input. Bars to the right increase predicted risk; "
        "bars to the left decrease it. Larger magnitude = stronger influence. "
        "This is not causal—features can be signals of crash intensity or context."
    )

    if SHAP_AVAILABLE:
        explainer, feat_names, shap_mode = make_shap_explainer(model)
        if explainer is None:
            st.info("SHAP is installed but couldn't create an explainer for this model.")
        else:
            fig_local = shap_local_bar(explainer, feat_names, shap_mode, model, x, top_n=12)
            if fig_local is None:
                st.info("Couldn't compute a local explanation for this input.")
            else:
                st.pyplot(fig_local)
                st.caption("Local: which encoded features pushed this probability up/down for THIS input.")
    else:
        st.caption("Install `shap` if you want per-case explanations: `conda install -c conda-forge shap`")

st.divider()
st.caption("Note: Labels shown are human-friendly; the model receives CRSS numeric codes.")
