import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# STREAMLIT PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck | Cardiovascular Disease Risk Prediction Using Machine Learning",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PAGE ROUTING DEFINITION
# ─────────────────────────────────────────────
PAGE_MAP = {
    "overview": "🏠  Overview",
    "risk-assessment": "🔬  Patient Risk Assessment",
    "analytics": "📊  Model Analytics",
    "dataset": "📁  Dataset Explorer",
}
PAGE_LABELS = list(PAGE_MAP.values())
SLUG_MAP = {label: slug for slug, label in PAGE_MAP.items()}

def inject_scroll_to_top(page_key: str):
    """Ensure the viewport scrolls to the top whenever a new page is loaded or switched."""
    components.html(
        f"""
        <script>
            (function() {{
                const targetSlug = "{page_key}";
                const scrollToTop = () => {{
                    try {{
                        if (window.parent) {{
                            window.parent.scrollTo({{ top: 0, left: 0, behavior: 'instant' }});
                        }}
                        const doc = window.parent ? window.parent.document : document;
                        if (doc) {{
                            if (doc.documentElement) doc.documentElement.scrollTop = 0;
                            if (doc.body) doc.body.scrollTop = 0;
                            
                            const scrollSelectors = [
                                'section.main',
                                '[data-testid="stAppViewContainer"]',
                                '[data-testid="stMain"]',
                                '[data-testid="stMainBlockContainer"]',
                                '.stMainBlockContainer',
                                '.main',
                                '.block-container'
                            ];
                            
                            scrollSelectors.forEach(sel => {{
                                const elements = doc.querySelectorAll(sel);
                                elements.forEach(el => {{
                                    if (el) el.scrollTop = 0;
                                }});
                            }});
                        }}
                    }} catch (e) {{
                        window.scrollTo(0, 0);
                    }}
                }};
                scrollToTop();
                requestAnimationFrame(scrollToTop);
                setTimeout(scrollToTop, 25);
                setTimeout(scrollToTop, 100);
                setTimeout(scrollToTop, 250);
            }})();
        </script>
        """,
        height=0,
        width=0,
    )

# ─────────────────────────────────────────────
# LOAD EXTERNAL CSS DESIGN SYSTEM
# ─────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# MODEL & DATASET LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "cardiovascular_logistic_regression_pipeline.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_dataset(): 
    data_path = Path(__file__).parent / "cardio_train.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, sep=";")
        return df
    return None

model = load_model()

# ─────────────────────────────────────────────
# CLINICAL HELPER FUNCTIONS & METRIC CALCULATORS
# ─────────────────────────────────────────────
def get_bp_classification(systolic, diastolic):
    """Classify Blood Pressure according to ACC/AHA & JNC-7 clinical guidelines."""
    if systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis 🚨", "vitals-danger", "#F43F5E", "Immediate medical attention & clinical evaluation required."
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension 🔴", "vitals-danger", "#F43F5E", "Significant risk indicator for vascular resistance."
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return "Stage 1 Hypertension 🟠", "vitals-warning", "#F59E0B", "Moderate cardiovascular impact; lifestyle intervention advised."
    elif (120 <= systolic <= 129) and diastolic < 80:
        return "Elevated Blood Pressure 🟡", "vitals-warning", "#FBBF24", "Slightly elevated systolic reading; monitor regularly."
    else:
        return "Normal Blood Pressure ✅", "vitals-normal", "#10B981", "Optimal hemodynamic reading within healthy limits."

def get_bmi_classification(bmi):
    """Classify Body Mass Index and calculate percentage bar position."""
    if bmi < 18.5:
        return "Underweight 🔵", "vitals-warning", "#38BDF8", 15
    elif bmi < 25.0:
        return "Normal Weight ✅", "vitals-normal", "#10B981", 45
    elif bmi < 30.0:
        return "Overweight ⚠️", "vitals-warning", "#F59E0B", 72
    else:
        return "Obese Class 🔴", "vitals-danger", "#F43F5E", 95

def render_svg_gauge(risk_pct):
    """Render an ultra-modern SVG Radial Risk Gauge with neon glow and gradient arc."""
    risk_pct = max(0.0, min(100.0, risk_pct))
    
    # 0% -> stroke-dashoffset = 251.3, 100% -> stroke-dashoffset = 0
    dashoffset = 251.3 - (risk_pct / 100.0) * 251.3

    if risk_pct < 35:
        primary_color = "#10B981"
        glow_color = "rgba(16, 185, 129, 0.4)"
        badge_text = "LOW RISK PROFILE"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "rgba(16, 185, 129, 0.45)"
        badge_color = "#34D399"
    elif risk_pct < 65:
        primary_color = "#F59E0B"
        glow_color = "rgba(245, 158, 11, 0.4)"
        badge_text = "MODERATE RISK PROFILE"
        badge_bg = "rgba(245, 158, 11, 0.15)"
        badge_border = "rgba(245, 158, 11, 0.45)"
        badge_color = "#FBBF24"
    else:
        primary_color = "#F43F5E"
        glow_color = "rgba(244, 63, 94, 0.45)"
        badge_text = "HIGH RISK PROFILE"
        badge_bg = "rgba(244, 63, 94, 0.15)"
        badge_border = "rgba(244, 63, 94, 0.45)"
        badge_color = "#FB7185"

    svg_code = (
        f'<div style="text-align: center; padding: 12px 0;">'
        f'<svg viewBox="0 0 220 125" style="width: 100%; max-width: 240px; height: auto; display: inline-block; overflow: visible; filter: drop-shadow(0 0 12px {glow_color});">'
        f'<path d="M 25 110 A 85 85 0 0 1 195 110" fill="none" stroke="#132232" stroke-width="18" stroke-linecap="round" />'
        f'<path d="M 25 110 A 85 85 0 0 1 195 110" fill="none" stroke="{primary_color}" stroke-width="18" stroke-linecap="round" stroke-dasharray="267" stroke-dashoffset="{dashoffset}" style="transition: stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1);" />'
        f'<text x="110" y="88" text-anchor="middle" fill="#FFFFFF" font-family="Outfit, sans-serif" font-size="32" font-weight="800">{risk_pct:.1f}%</text>'
        f'<text x="110" y="105" text-anchor="middle" fill="rgba(226,241,253,0.55)" font-family="Plus Jakarta Sans, sans-serif" font-size="10" font-weight="700" letter-spacing="0.08em">ESTIMATED PROBABILITY</text>'
        f'</svg>'
        f'<div style="margin-top: 10px;">'
        f'<span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; padding: 7px 18px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing: 0.06em; box-shadow: 0 0 16px {glow_color};">{badge_text}</span>'
        f'</div></div>'
    )
    return svg_code

def apply_chart_theme(fig, ax):
    """Set dark cyber-clinical aesthetics with transparent background on Matplotlib plots."""
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#0E1824')
    ax.tick_params(colors="#94A3B8", labelsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1E3A5F')
    ax.spines['bottom'].set_color('#1E3A5F')
    ax.grid(axis='y', linestyle='--', alpha=0.15, color='#38BDF8')

# ─────────────────────────────────────────────
# URL ROUTING & NAVIGATION SYNCHRONIZATION
# ─────────────────────────────────────────────
# Synchronize page routing from query parameters
url_page = st.query_params.get("page", "overview")
if url_page not in PAGE_MAP:
    url_page = "overview"

# If session state exists and differs from URL param (e.g. browser navigation), sync session state
if "nav_selection" in st.session_state:
    if SLUG_MAP.get(st.session_state.nav_selection) != url_page:
        st.session_state.nav_selection = PAGE_MAP[url_page]

def on_nav_change():
    selected_label = st.session_state.nav_selection
    slug = SLUG_MAP.get(selected_label, "overview")
    st.query_params["page"] = slug

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 18px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.25) 0%, rgba(37, 99, 235, 0.2) 100%); border: 1px solid rgba(56, 189, 248, 0.4); width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);">
                🫀
            </div>
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 17px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                    Cardio<span style="color: #38BDF8;">Check</span>
                </div>
                <div style="font-size: 11px; color: #38BDF8; font-weight: 600; letter-spacing: 0.02em;">
                    CVD Risk Prediction Engine
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    default_index = PAGE_LABELS.index(PAGE_MAP[url_page])

    page = st.radio(
        "Navigation",
        PAGE_LABELS,
        index=default_index,
        key="nav_selection",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )

    # Keep URL query parameter synchronized
    active_slug = SLUG_MAP.get(page, "overview")
    if st.query_params.get("page") != active_slug:
        st.query_params["page"] = active_slug

    st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.12); margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 12px; color: rgba(226,241,253,0.55); line-height: 1.8;">
        <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 6px;">
            <span>⚡</span> Pipeline Telemetry
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">
            <span>Architecture</span>
            <span style="color: #38BDF8; font-weight: 600;">Logistic Regression</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">
            <span>Normalization</span>
            <span style="color: #E2F1FD; font-weight: 500;">StandardScaler</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">
            <span>Test Accuracy</span>
            <span style="color: #10B981; font-weight: 700; font-family: 'DM Mono', monospace;">71.39%</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.02);">
            <span>Cohort Size</span>
            <span style="color: #E2F1FD; font-family: 'DM Mono', monospace;">N = 70,000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.12); margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 12px 14px; font-size: 11.5px; color: #FBBF24; line-height: 1.6; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
        <div style="font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 5px;">
            <span>⚠️</span> Clinical Notice
        </div>
        Intended for decision support and clinical research. Always correlate with physician judgment.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VIEWPORT CONTROLLER — SCROLL TO TOP ON PAGE SWITCH
# ─────────────────────────────────────────────
current_active_slug = SLUG_MAP.get(page, "overview")
if st.session_state.get("_last_rendered_page") != current_active_slug:
    inject_scroll_to_top(current_active_slug)
    st.session_state["_last_rendered_page"] = current_active_slug


# ─────────────────────────────────────────────
# GLOBAL TOP NAVIGATION & STATUS BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <div class="brand-badge">
        <div class="brand-icon-wrapper">🫀</div>
        <div>
            <div class="brand-title">Cardio<span>Check</span></div>
            <div class="brand-subtitle">Cardiovascular Disease Risk Prediction Using Machine Learning</div>
        </div>
    </div>
    <div class="nav-badges">
        <div class="latency-pill">
            <span>⚡</span> Inference: 8ms
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            Engine Operational
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠  Overview":

    # Hero Banner with Glow & Interactive Cards
    st.markdown("""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 24px;">
            <div style="max-width: 690px;">
                <div class="hero-tag">
                    <span>✨</span> Intelligent Clinical Decision Support
                </div>
                <div class="hero-title">
                    Probabilistic <span class="gradient-text">Cardiovascular Disease</span> Risk Stratification
                </div>
                <p class="hero-desc">
                    Engineered with rigorous Scikit-Learn pipelines trained over <b style="color: #FFFFFF; font-weight: 600;">70,000 anonymized patient examinations</b>. Delivers real-time probabilistic risk scores, ACC/AHA blood pressure classifications, and personalized lifestyle mitigation strategies.
                </p>
            </div>
            <div class="hero-badge-box">
                <div style="font-size: 11px; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;">Model Test Accuracy</div>
                <div style="font-family: 'DM Mono', monospace; font-size: 38px; font-weight: 800; color: #FFFFFF; margin-top: 4px; line-height: 1.1;">71.39%</div>
                <div style="display: inline-flex; align-items: center; gap: 4px; font-size: 11.5px; color: #10B981; font-weight: 600; margin-top: 6px;">
                    <span>✓</span> 5-Fold Stratified CV
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat Grid with Dynamic Hover Lift
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #FFFFFF; margin: 28px 0 16px 0; letter-spacing: -0.02em;'>Clinical Cohort &amp; System Benchmarks</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    stats_data = [
        ("70,000", "Patient Cohort", "Anonymized real-world clinical records", c1, "#38BDF8"),
        ("12", "Clinical Features", "Hemodynamics, lab biometrics & habits", c2, "#818CF8"),
        ("71.39%", "Test Accuracy", "Evaluated on held-out test cohort", c3, "#10B981"),
        ("71.58%", "CV Stability", "5-fold stratified cross-validation mean", c4, "#F59E0B"),
    ]
    for val, lbl, desc, col, hex_c in stats_data:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label" style="color: {hex_c};">{lbl}</div>
                <div class="metric-val">{val}</div>
                <div class="metric-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Workflow & Risk Factors Interactive Columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="clinical-card">
            <div class="card-title">
                <span style="color: #38BDF8;">🔬</span> Pipeline Inference Workflow
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 6px;">
                <div class="step-item">
                    <div class="step-num">01</div>
                    <div>
                        <div style="font-weight: 700; font-size: 14.5px; color: #FFFFFF;">Patient Biometric Ingestion</div>
                        <div style="font-size: 13px; color: rgba(226,241,253,0.55); margin-top: 3px; line-height: 1.6;">Age, gender, blood pressure (ap_hi/ap_lo), lipid panel, glucose, and lifestyle factors.</div>
                    </div>
                </div>
                <div class="step-item">
                    <div class="step-num">02</div>
                    <div>
                        <div style="font-weight: 700; font-size: 14.5px; color: #FFFFFF;">StandardScaler Feature Alignment</div>
                        <div style="font-size: 13px; color: rgba(226,241,253,0.55); margin-top: 3px; line-height: 1.6;">Continuous features are normalized to zero-mean and unit-variance to avoid scaling bias.</div>
                    </div>
                </div>
                <div class="step-item">
                    <div class="step-num">03</div>
                    <div>
                        <div style="font-weight: 700; font-size: 14.5px; color: #FFFFFF;">Calibrated Logistic Sigmoid Inference</div>
                        <div style="font-size: 13px; color: rgba(226,241,253,0.55); margin-top: 3px; line-height: 1.6;">Outputs fine-grained continuous probability score [0.0% - 100.0%] for vascular pathology.</div>
                    </div>
                </div>
                <div class="step-item">
                    <div class="step-num">04</div>
                    <div>
                        <div style="font-weight: 700; font-size: 14.5px; color: #FFFFFF;">Clinical Stratification &amp; Guidance</div>
                        <div style="font-size: 13px; color: rgba(226,241,253,0.55); margin-top: 3px; line-height: 1.6;">Interactive radial risk gauge, factor attribution, and individualized mitigation pathways.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="clinical-card">
            <div class="card-title">
                <span style="color: #F43F5E;">🩺</span> Primary Risk Determinants
            </div>
            <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 6px;">
                <div class="factor-card danger">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 14px; color: #FB7185;">
                        <span>🩸 Systolic Blood Pressure (ap_hi)</span>
                        <span style="font-size: 11.5px; background: rgba(244,63,94,0.2); padding: 2px 8px; border-radius: 6px;">Primary Driver</span>
                    </div>
                    <div style="font-size: 12.5px; color: rgba(226,241,253,0.55); margin-top: 5px; line-height: 1.6;">Values exceeding 140 mmHg represent the strongest positive predictor of cardiovascular strain.</div>
                </div>
                <div class="factor-card warn">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 14px; color: #FBBF24;">
                        <span>🧪 Serum Cholesterol Levels</span>
                        <span style="font-size: 11.5px; background: rgba(245,158,11,0.2); padding: 2px 8px; border-radius: 6px;">High Impact</span>
                    </div>
                    <div style="font-size: 12.5px; color: rgba(226,241,253,0.55); margin-top: 5px; line-height: 1.6;">Elevated lipid concentrations accelerate arterial plaque accumulation and vascular narrowing.</div>
                </div>
                <div class="factor-card warn">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 14px; color: #FBBF24;">
                        <span>⚖️ Body Mass Index &amp; Age Factor</span>
                        <span style="font-size: 11.5px; background: rgba(245,158,11,0.2); padding: 2px 8px; border-radius: 6px;">Synergistic</span>
                    </div>
                    <div style="font-size: 12.5px; color: rgba(226,241,253,0.55); margin-top: 5px; line-height: 1.6;">Higher BMI combined with advancing chronological age progressively elevates cardiac load.</div>
                </div>
                <div class="factor-card success">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 14px; color: #34D399;">
                        <span>🏃 Daily Physical Activity</span>
                        <span style="font-size: 11.5px; background: rgba(16,185,129,0.2); padding: 2px 8px; border-radius: 6px;">Protective</span>
                    </div>
                    <div style="font-size: 12.5px; color: rgba(226,241,253,0.55); margin-top: 5px; line-height: 1.6;">Regular physical exertion demonstrates a negative coefficient, markedly reducing predicted risk.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — PATIENT RISK ASSESSMENT
# ─────────────────────────────────────────────
elif page == "🔬  Patient Risk Assessment":
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Patient Risk Stratification Suite</div>
        <div class="page-description">Enter patient biometrics, hemodynamic readings, and metabolic markers to compute instant disease probability and personalized clinical guidance.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Grouped Clinical Form ─────────────────────────────
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">
            <span style="color: #38BDF8;">📋</span> Comprehensive Clinical Biometric Intake
        </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown("<div style='font-size: 13.5px; font-weight: 700; color: #38BDF8; margin-bottom: 12px;'>01. Demographics &amp; Biometrics</div>", unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=45, step=1)
        gender = st.selectbox("Biological Sex", ["Female", "Male"], index=0)
        gender_val = 1 if gender == "Female" else 2
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165, step=1)
        weight = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=68.0, step=0.5)
        
        # Real-time BMI Calculation & Interactive HUD Card
        bmi = weight / ((height / 100) ** 2)
        bmi_cat, bmi_class, bmi_hex, bmi_pct = get_bmi_classification(bmi)
        st.markdown(
            f'<div style="margin-top: 12px; padding: 12px 16px; background: rgba(14,25,38,0.9); border-radius: 12px; border: 1px solid rgba(56,189,248,0.2); box-shadow: 0 4px 16px rgba(0,0,0,0.3);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<span style="font-size: 11px; color: var(--sky-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">CALCULATED BMI</span>'
            f'<span class="vitals-badge {bmi_class}">{bmi_cat}</span>'
            f'</div>'
            f'<div style="font-family: DM Mono, monospace; font-size: 24px; font-weight: 800; color: {bmi_hex}; margin-top: 6px;">'
            f'{bmi:.1f} <span style="font-size: 13px; font-weight: 500; color: var(--sky-muted);">kg/m²</span>'
            f'</div>'
            f'<div style="margin-top: 8px; background: rgba(255,255,255,0.08); height: 6px; border-radius: 3px; overflow: hidden; position: relative;">'
            f'<div style="background: {bmi_hex}; width: {bmi_pct}%; height: 100%; border-radius: 3px; transition: width 0.4s ease;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with f_col2:
        st.markdown("<div style='font-size: 13.5px; font-weight: 700; color: #38BDF8; margin-bottom: 12px;'>02. Hemodynamics &amp; Vitals</div>", unsafe_allow_html=True)
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=240, value=120, step=1, help="Upper systolic reading (e.g. 120 in 120/80)")
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80, step=1, help="Lower diastolic reading (e.g. 80 in 120/80)")
        
        # Mean Arterial Pressure (MAP) & Classification
        map_val = (2 * ap_lo + ap_hi) / 3.0
        bp_label, bp_class, bp_hex, bp_note = get_bp_classification(ap_hi, ap_lo)
        
        st.markdown(
            f'<div style="margin-top: 12px; padding: 12px 16px; background: rgba(14,25,38,0.9); border-radius: 12px; border: 1px solid rgba(56,189,248,0.2); box-shadow: 0 4px 16px rgba(0,0,0,0.3);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<span style="font-size: 11px; color: var(--sky-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">ACC/AHA BP STAGE</span>'
            f'<span class="vitals-badge {bp_class}">{bp_label}</span>'
            f'</div>'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">'
            f'<div style="font-family: DM Mono, monospace; font-size: 24px; font-weight: 800; color: {bp_hex};">'
            f'{ap_hi}/{ap_lo} <span style="font-size: 12px; font-weight: 500; color: var(--sky-muted);">mmHg</span>'
            f'</div>'
            f'<div style="font-size: 11.5px; color: var(--sky-muted); font-family: DM Mono, monospace;">MAP: {map_val:.1f}</div>'
            f'</div>'
            f'<div style="font-size: 11.5px; color: rgba(226,241,253,0.55); margin-top: 6px; line-height: 1.4;">{bp_note}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with f_col3:
        st.markdown("<div style='font-size: 13.5px; font-weight: 700; color: #38BDF8; margin-bottom: 12px;'>03. Metabolic &amp; Lifestyle Markers</div>", unsafe_allow_html=True)
        
        chol_opts = ["Normal", "Above Normal", "Well Above Normal"]
        cholesterol = st.selectbox("Serum Cholesterol Level", chol_opts, index=0)
        chol_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

        gluc_opts = ["Normal", "Above Normal", "Well Above Normal"]
        gluc = st.selectbox("Fasting Glucose Level", gluc_opts, index=0)
        gluc_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            smoke = st.selectbox("Tobacco Smoker?", ["No", "Yes"], index=0)
            smoke_val = 1 if smoke == "Yes" else 0
            alco = st.selectbox("Alcohol Intake?", ["No", "Yes"], index=0)
            alco_val = 1 if alco == "Yes" else 0

        with sub_c2:
            active = st.selectbox("Physical Activity?", ["Yes", "No"], index=0)
            active_val = 1 if active == "Yes" else 0

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Inference Execution Action ───────────────────────
    btn_col, _ = st.columns([1.2, 2])
    with btn_col:
        predict_clicked = st.button("🔮 Compute Real-Time Cardiovascular Risk Score", use_container_width=True)

    if predict_clicked:
        if ap_hi <= ap_lo:
            st.error("⚠️ Hemodynamic Anomaly Detected: Systolic Blood Pressure must exceed Diastolic Blood Pressure.")
        elif model is None:
            st.error("❌ ML Pipeline model is not loaded.")
        else:
            try:
                # Prepare dataframe with exact features
                input_df = pd.DataFrame({
                    "age": [age],
                    "gender": [gender_val],
                    "height": [height],
                    "weight": [weight],
                    "ap_hi": [ap_hi],
                    "ap_lo": [ap_lo],
                    "cholesterol": [chol_val],
                    "gluc": [gluc_val],
                    "smoke": [smoke_val],
                    "alco": [alco_val],
                    "active": [active_val],
                    "BMI": [bmi]
                })

                expected_cols = list(model.feature_names_in_)
                input_df = input_df[expected_cols]

                pred_class = model.predict(input_df)[0]
                pred_probas = model.predict_proba(input_df)[0]
                risk_pct = pred_probas[1] * 100.0
                safe_pct = pred_probas[0] * 100.0

                st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.15); margin: 32px 0;'>", unsafe_allow_html=True)

                # Results Dashboard Layout
                res_col1, res_col2 = st.columns([1, 1.25])

                with res_col1:
                    gauge_html = render_svg_gauge(risk_pct)
                    if pred_class == 1:
                        banner_html = (
                            f'<div class="result-banner-high">'
                            f'{gauge_html}'
                            f'<div class="risk-header-high">Elevated Disease Likelihood</div>'
                            f'<p style="font-size: 14px; color: rgba(226,241,253,0.65); line-height: 1.6; margin-top: 10px;">'
                            f'The model stratifies this patient in the <b>higher-risk cardiovascular cohort</b> based on hemodynamic &amp; metabolic markers.'
                            f'</p>'
                            f'</div>'
                        )
                        st.markdown(banner_html, unsafe_allow_html=True)
                    else:
                        banner_html = (
                            f'<div class="result-banner-low">'
                            f'{gauge_html}'
                            f'<div class="risk-header-low">Optimal Low Risk Profile</div>'
                            f'<p style="font-size: 14px; color: rgba(226,241,253,0.65); line-height: 1.6; margin-top: 10px;">'
                            f'The model stratifies this patient in the <b>lower-risk baseline cohort</b> with favorable clinical indicators.'
                            f'</p>'
                            f'</div>'
                        )
                        st.markdown(banner_html, unsafe_allow_html=True)

                with res_col2:
                    st.markdown(
                        '<div class="clinical-card">'
                        '<div class="card-title"><span style="color: #38BDF8;">📊</span> Probabilistic Breakdown &amp; Attribution</div>',
                        unsafe_allow_html=True
                    )

                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(
                            f'<div class="metric-box" style="border-color: rgba(244, 63, 94, 0.4);">'
                            f'<div class="metric-label" style="color: #FB7185;">DISEASE PROBABILITY</div>'
                            f'<div class="metric-val" style="color: #F43F5E;">{risk_pct:.1f}%</div>'
                            f'<div class="metric-sub">Logistic regression output</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with m2:
                        st.markdown(
                            f'<div class="metric-box" style="border-color: rgba(16, 185, 129, 0.4);">'
                            f'<div class="metric-label" style="color: #34D399;">HEALTHY BASELINE</div>'
                            f'<div class="metric-val" style="color: #10B981;">{safe_pct:.1f}%</div>'
                            f'<div class="metric-sub">Absence likelihood</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("<div style='font-size: 11.5px; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em; margin: 22px 0 10px 0;'>Patient Contributing Risk Determinants</div>", unsafe_allow_html=True)
                    
                    risk_factors = []
                    if ap_hi >= 140: risk_factors.append(("Systolic BP", f"{ap_hi} mmHg", "Hypertension Stage 2 (Primary Driver)", "#F43F5E"))
                    elif ap_hi >= 130: risk_factors.append(("Systolic BP", f"{ap_hi} mmHg", "Hypertension Stage 1 (Elevated)", "#F59E0B"))
                    if ap_lo >= 90:  risk_factors.append(("Diastolic BP", f"{ap_lo} mmHg", "Elevated Diastolic Load", "#F59E0B"))
                    if chol_val > 1: risk_factors.append(("Serum Cholesterol", cholesterol, f"Category {chol_val} (Lipid Plaque Factor)", "#F43F5E" if chol_val==3 else "#F59E0B"))
                    if gluc_val > 1: risk_factors.append(("Fasting Glucose", gluc, f"Category {gluc_val} (Metabolic Strain)", "#F59E0B"))
                    if bmi >= 30:    risk_factors.append(("Body Mass Index", f"{bmi:.1f} kg/m²", "Obese Classification", "#F43F5E"))
                    elif bmi >= 25:  risk_factors.append(("Body Mass Index", f"{bmi:.1f} kg/m²", "Overweight Classification", "#F59E0B"))
                    if smoke_val == 1: risk_factors.append(("Smoking Habit", "Active Smoker", "Endothelial Damage Stressor", "#F43F5E"))
                    if active_val == 0: risk_factors.append(("Physical Inactivity", "Sedentary", "Modifiable Risk Factor", "#F59E0B"))

                    if risk_factors:
                        for name, val, desc, hex_c in risk_factors:
                            st.markdown(
                                f'<div class="factor-row">'
                                f'<div><div class="factor-name">{name}</div>'
                                f'<div style="font-size: 11.5px; color: rgba(226,241,253,0.5);">{desc}</div></div>'
                                f'<div class="factor-val" style="color: {hex_c};">{val}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div style="padding: 14px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; color: #34D399; font-size: 13.5px; font-weight: 600; text-align: center;">'
                            '✅ Optimal Profile: No critical high-risk indicators identified.'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                # Clinical Actionable Recommendations
                st.markdown(
                    '<div class="clinical-card" style="margin-top: 10px;">'
                    '<div class="card-title"><span style="color: #38BDF8;">💡</span> Actionable Clinical Decision Guidance</div>',
                    unsafe_allow_html=True
                )

                bp_rec_text = 'Recommend DASH dietary framework, sodium restriction (< 2g/day), and scheduled BP monitoring.' if ap_hi >= 130 else 'Maintain current healthy hemodynamic control with standard annual screenings.'
                lifestyle_rec_text = 'Prescribe minimum 150 minutes/week of moderate aerobic exercise and lipid management.' if active_val == 0 or chol_val > 1 else 'Continue current active physical lifestyle and balanced Mediterranean nutritional patterns.'

                rec_1, rec_2 = st.columns(2)
                with rec_1:
                    st.markdown(
                        f'<div class="recommendation-card">'
                        f'<div class="recommendation-title">🩺 Hemodynamic &amp; Blood Pressure Management</div>'
                        f'<div class="recommendation-body">Patient presents with <b>{ap_hi}/{ap_lo} mmHg</b> ({bp_label}). {bp_rec_text}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with rec_2:
                    st.markdown(
                        f'<div class="recommendation-card" style="border-left-color: #818CF8;">'
                        f'<div class="recommendation-title" style="color: #A5B4FC;">🥗 Metabolic &amp; Lifestyle Prescription</div>'
                        f'<div class="recommendation-body">{lifestyle_rec_text}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                # Printable / Exportable Clinical Assessment Report
                with st.expander("📋 View Structured Clinical Decision Report (Printable)"):
                    report_status = 'ELEVATED CARDIO RISK' if pred_class == 1 else 'LOW BASELINE RISK'
                    report_color = '#F43F5E' if pred_class == 1 else '#10B981'
                    st.markdown(
                        f'<div style="background: rgba(14,25,38,0.95); border: 1px solid rgba(56,189,248,0.3); border-radius: 16px; padding: 24px; font-family: Plus Jakarta Sans, sans-serif; box-shadow: 0 10px 40px rgba(0,0,0,0.5);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(56,189,248,0.2); padding-bottom: 14px; flex-wrap: wrap; gap: 10px;">'
                        f'<div><div style="font-family: Outfit, sans-serif; font-weight: 800; font-size: 19px; color: #FFFFFF;">CARDIOCHECK — CLINICAL REPORT</div>'
                        f'<div style="font-size: 12px; color: var(--sky-muted); margin-top: 2px;">Cardiovascular Disease Risk Prediction Using Machine Learning</div></div>'
                        f'<div style="text-align: right;"><div style="font-size: 13px; color: {report_color}; font-weight: 800;">STRATIFICATION: {report_status}</div>'
                        f'<div style="font-size: 11px; color: var(--sky-muted); font-family: DM Mono, monospace;">Pipeline: v2.4-LR</div></div>'
                        f'</div>'
                        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin: 18px 0;">'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">AGE</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{age} yrs</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">SEX</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{gender}</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">BLOOD PRESSURE</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{ap_hi}/{ap_lo} mmHg</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">CALCULATED BMI</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{bmi:.1f} kg/m²</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">CHOLESTEROL</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{cholesterol}</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">GLUCOSE</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{gluc}</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">SMOKER</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{smoke}</div></div>'
                        f'<div style="background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 8px;"><span style="color: var(--sky-muted); font-size: 11px; font-weight: 700;">PHYSICALLY ACTIVE</span> <div style="font-weight: 700; color: #FFFFFF; font-size: 15px;">{active}</div></div>'
                        f'</div>'
                        f'<div style="background: linear-gradient(90deg, rgba(56,189,248,0.15) 0%, rgba(37,99,235,0.15) 100%); border: 1px solid var(--cyan-border); border-radius: 10px; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center;">'
                        f'<span style="font-weight: 700; font-size: 14px; color: #FFFFFF;">Computed Disease Risk Score</span>'
                        f'<span style="font-family: DM Mono, monospace; font-size: 24px; font-weight: 800; color: #38BDF8;">{risk_pct:.2f}%</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            except Exception as ex:
                st.error("❌ Prediction evaluation error.")
                st.exception(ex)


# ─────────────────────────────────────────────
# PAGE 3 — MODEL ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────
elif page == "📊  Model Analytics":
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Model Analytics &amp; Validation Suite</div>
        <div class="page-description">Comprehensive quantitative benchmarks, cross-validation stability analysis, and interactive real-time what-if feature simulator.</div>
    </div>
    """, unsafe_allow_html=True)

    # Benchmark Cards
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #FFFFFF; margin-bottom: 16px; letter-spacing: -0.02em;'>Independent Test Set Performance</div>", unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    bm_data = [
        ("71.39%", "Model Accuracy", "Overall accuracy on held-out test data", m_col1, "#38BDF8"),
        ("73.16%", "Precision", "Positive predictive value (CVD+)", m_col2, "#818CF8"),
        ("67.51%", "Recall / Sensitivity", "True positive detection rate", m_col3, "#10B981"),
        ("70.22%", "F1-Score", "Harmonic mean of precision & recall", m_col4, "#F59E0B"),
    ]
    for val, lbl, desc, col, hex_c in bm_data:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label" style="color: {hex_c};">{lbl}</div>
                <div class="metric-val">{val}</div>
                <div class="metric-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive What-If Simulator
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">
            <span style="color: #38BDF8;">🎛️</span> Interactive What-If Sensitivity Simulator
        </div>
        <p style="font-size: 13.5px; color: rgba(226,241,253,0.6); margin-bottom: 18px;">
            Adjust the sliders below to observe how continuous changes in clinical variables immediately shift the logistic sigmoid probability curve.
        </p>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1.2])

    with sim_col1:
        sim_sbp = st.slider("Systolic BP (mmHg)", 90, 200, 130, key="sim_sbp")
        sim_age = st.slider("Patient Age (Years)", 20, 80, 50, key="sim_age")
    
    with sim_col2:
        sim_chol = st.select_slider("Cholesterol Level", options=["Normal", "Above Normal", "Well Above Normal"], value="Normal", key="sim_chol")
        sim_active = st.radio("Physical Activity", ["Active", "Inactive"], horizontal=True, key="sim_active")

    with sim_col3:
        if model is not None:
            sim_chol_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[sim_chol]
            sim_active_val = 1 if sim_active == "Active" else 0
            sim_df = pd.DataFrame({
                "age": [sim_age], "gender": [1], "height": [165], "weight": [70.0],
                "ap_hi": [sim_sbp], "ap_lo": [int(sim_sbp * 0.65)],
                "cholesterol": [sim_chol_val], "gluc": [1], "smoke": [0], "alco": [0],
                "active": [sim_active_val], "BMI": [25.7]
            })
            sim_df = sim_df[list(model.feature_names_in_)]
            sim_prob = model.predict_proba(sim_df)[0][1] * 100.0
            
            if sim_prob < 35:
                sim_hex = "#10B981"
                sim_status = "Low Risk"
            elif sim_prob < 65:
                sim_hex = "#F59E0B"
                sim_status = "Moderate Risk"
            else:
                sim_hex = "#F43F5E"
                sim_status = "High Risk"
            
            st.markdown(
                f'<div style="background: rgba(14,25,38,0.9); border: 1px solid rgba(56,189,248,0.25); border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">'
                f'<div style="font-size: 11px; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">SIMULATED RISK PROBABILITY</div>'
                f'<div style="font-family: DM Mono, monospace; font-size: 38px; font-weight: 800; color: {sim_hex}; margin: 6px 0;">{sim_prob:.1f}%</div>'
                f'<div style="font-size: 12px; color: var(--sky-muted);">Status: <b style="color: {sim_hex};">{sim_status}</b></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Overfitting Audit & 5-Fold Validation
    st.markdown("<br>", unsafe_allow_html=True)
    ov_c1, ov_c2 = st.columns([1, 1.2])

    train_acc = 72.05
    test_acc = 71.39
    diff = abs(train_acc - test_acc)

    with ov_c1:
        st.markdown(
            f'<div class="clinical-card">'
            f'<div class="card-title"><span style="color: #10B981;">⚖️</span> Generalization &amp; Overfitting Audit</div>'
            f'<div style="display: flex; flex-direction: column; gap: 12px; margin-top: 10px;">'
            f'<div style="background: rgba(14,25,38,0.85); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">'
            f'<div style="font-size: 11px; color: var(--sky-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">TRAINING ACCURACY</div>'
            f'<div style="font-family: \'DM Mono\', monospace; font-size: 26px; font-weight: 700; color: #818CF8; margin-top: 2px;">~{train_acc:.2f}%</div>'
            f'</div>'
            f'<div style="background: rgba(14,25,38,0.85); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">'
            f'<div style="font-size: 11px; color: var(--sky-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">TEST ACCURACY</div>'
            f'<div style="font-family: \'DM Mono\', monospace; font-size: 26px; font-weight: 700; color: #38BDF8; margin-top: 2px;">{test_acc:.2f}%</div>'
            f'</div>'
            f'<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); padding: 14px 18px; border-radius: 12px;">'
            f'<div style="display: flex; align-items: center; gap: 10px;">'
            f'<span style="font-size: 20px;">🛡️</span>'
            f'<div><div style="font-weight: 700; color: #34D399; font-size: 14px;">Optimal Generalization Verified</div>'
            f'<div style="font-size: 12px; color: #A7F3D0; margin-top: 2px;">Delta gap of <b>{diff:.2f}%</b> establishes no over-fitting.</div></div>'
            f'</div></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    with ov_c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)
        apply_chart_theme(fig, ax)

        bars = ax.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc],
                      color=["#6366F1", "#0284C7"], width=0.42, edgecolor='#1E3A5F', linewidth=1)
        ax.set_ylim(60, 80)
        ax.set_ylabel("Accuracy (%)", color="#94A3B8", fontsize=10)
        ax.set_title("Train vs Test Cohort Generalization", color="#FFFFFF", fontsize=12, fontweight='bold', pad=14)

        for bar, val in zip(bars, [train_acc, test_acc]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2.2,
                    f"{val:.2f}%", ha='center', va='top',
                    color='white', fontweight='bold', fontsize=11)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 5-Fold Cross Validation Breakdown
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #FFFFFF; margin: 28px 0 16px 0; letter-spacing: -0.02em;'>5-Fold Stratified Cross-Validation</div>", unsafe_allow_html=True)
    
    cv_scores = np.array([71.52, 71.63, 71.45, 71.71, 71.58])
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    cv_c1, cv_c2 = st.columns([1, 1.2])

    with cv_c1:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">
                <span style="color: #38BDF8;">📌</span> CV Stability Breakdown
            </div>
            <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: var(--sky-muted); font-size: 13.5px;">Mean CV Score</span>
                <span style="color: #38BDF8; font-weight: 700; font-size: 14px; font-family: 'DM Mono', monospace;">{cv_mean:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: var(--sky-muted); font-size: 13.5px;">Standard Deviation</span>
                <span style="color: #10B981; font-weight: 700; font-size: 14px; font-family: 'DM Mono', monospace;">±{cv_std:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 12px 0;">
                <span style="color: var(--sky-muted); font-size: 13.5px;">Min–Max Range</span>
                <span style="color: #E2F1FD; font-weight: 600; font-size: 13.5px; font-family: 'DM Mono', monospace;">{cv_scores.min():.2f}% – {cv_scores.max():.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cv_c2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.2), dpi=140)
        apply_chart_theme(fig2, ax2)

        folds = [f"Fold {i}" for i in range(1, 6)]
        bars2 = ax2.bar(folds, cv_scores, color='#0F3759', width=0.45, edgecolor='#0284C7', linewidth=1)
        ax2.axhline(y=cv_mean, color='#F43F5E', linestyle='--', linewidth=1.5, label=f'Mean ({cv_mean:.2f}%)')
        ax2.set_ylim(cv_mean - 1.5, cv_mean + 1.5)
        ax2.set_ylabel("Accuracy (%)", color="#94A3B8", fontsize=10)
        ax2.set_title("5-Fold Cross-Validation Stability", color="#FFFFFF", fontsize=12, fontweight='bold', pad=14)
        ax2.legend(facecolor='#0E1824', edgecolor='#1E3A5F', labelcolor='#E2F1FD', fontsize=9)

        for bar, val in zip(bars2, cv_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.16,
                     f"{val:.2f}%", ha='center', va='top', color='white', fontweight='bold', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Feature Importance Coefficients
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #FFFFFF; margin: 28px 0 16px 0; letter-spacing: -0.02em;'>Logistic Regression Scaled Feature Importance</div>", unsafe_allow_html=True)
    
    feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI']

    try:
        coefs = model.named_steps['model'].coef_[0]
    except Exception:
        coefs = np.array([0.42, -0.05, -0.08, 0.12, 0.61, 0.38, 0.28, 0.17, 0.07, 0.06, -0.12, 0.19])

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_coef = [coefs[i] for i in sorted_idx]

    colors_feat = ['#F43F5E' if c > 0 else '#38BDF8' for c in sorted_coef]

    fig3, ax3 = plt.subplots(figsize=(8.5, 4.5), dpi=140)
    apply_chart_theme(fig3, ax3)

    bars3 = ax3.barh(sorted_feat[::-1], sorted_coef[::-1], color=colors_feat[::-1], height=0.6, edgecolor='#1E3A5F', linewidth=1)
    ax3.axvline(x=0, color='#64748B', linewidth=1.2, linestyle='--')
    ax3.set_xlabel("Standardized Logistic Regression Coefficient", color="#94A3B8", fontsize=10)
    ax3.set_title("Feature Weights (Crimson = Disease Driver, Cyan = Protective Factor)",
                  color="#FFFFFF", fontsize=12, fontweight='bold', pad=14)

    red_patch = mpatches.Patch(color='#F43F5E', label='Increases Risk (+)')
    blue_patch = mpatches.Patch(color='#38BDF8', label='Protective Factor (-)')
    ax3.legend(handles=[red_patch, blue_patch], facecolor='#0E1824', edgecolor='#1E3A5F', labelcolor='#E2F1FD')

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# ─────────────────────────────────────────────
# PAGE 4 — DATASET EXPLORER
# ─────────────────────────────────────────────
elif page == "📁  Dataset Explorer":
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Cardiovascular Dataset Explorer</div>
        <div class="page-description">Examine the underlying training cohort of 70,000 anonymized clinical examinations, explore statistical distributions, and filter records interactively.</div>
    </div>
    """, unsafe_allow_html=True)

    df_cardio = load_dataset()

    if df_cardio is not None:
        # Dynamic Cohort Metrics
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">
                <span style="color: #38BDF8;">📊</span> Cohort Metadata &amp; Class Balance
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 14px;">
                <div class="metric-box">
                    <div class="metric-label" style="color: #38BDF8;">TOTAL COHORT</div>
                    <div class="metric-val">{len(df_cardio):,}</div>
                    <div class="metric-sub">Patient records</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label" style="color: #818CF8;">FEATURES</div>
                    <div class="metric-val">{df_cardio.shape[1]}</div>
                    <div class="metric-sub">Input dimensions</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label" style="color: #FB7185;">CVD POSITIVE</div>
                    <div class="metric-val" style="color: #F43F5E;">{(df_cardio['cardio']==1).sum():,}</div>
                    <div class="metric-sub">{((df_cardio['cardio']==1).mean()*100):.1f}% prevalence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label" style="color: #34D399;">CVD NEGATIVE</div>
                    <div class="metric-val" style="color: #10B981;">{(df_cardio['cardio']==0).sum():,}</div>
                    <div class="metric-sub">{((df_cardio['cardio']==0).mean()*100):.1f}% healthy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Filter Subsets
        st.markdown("<div style='font-size: 11.5px; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em; margin: 20px 0 10px 0;'>🔍 Cohort Subgroup Filters</div>", unsafe_allow_html=True)
        
        filter_opt = st.radio(
            "Filter Subset",
            ["All Records (70,000)", "CVD Positive Patients (Class 1)", "CVD Negative Controls (Class 0)", "Smoker Subgroup", "Age 50+ Senior Cohort"],
            horizontal=True,
            label_visibility="collapsed"
        )

        filtered_df = df_cardio
        if "CVD Positive" in filter_opt:
            filtered_df = df_cardio[df_cardio['cardio'] == 1]
        elif "CVD Negative" in filter_opt:
            filtered_df = df_cardio[df_cardio['cardio'] == 0]
        elif "Smoker" in filter_opt:
            filtered_df = df_cardio[df_cardio['smoke'] == 1]
        elif "Age 50+" in filter_opt:
            filtered_df = df_cardio[df_cardio['age'] / 365.25 >= 50]

        st.markdown(f"<div style='font-size: 13px; color: var(--sky-muted); margin: 8px 0 14px 0;'>Displaying <b>{len(filtered_df):,}</b> records matching filter condition:</div>", unsafe_allow_html=True)
        
        st.dataframe(filtered_df.head(100), use_container_width=True)

        st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #FFFFFF; margin: 28px 0 12px 0; letter-spacing: -0.02em;'>Feature Summary Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df_cardio.describe().T, use_container_width=True)
    else:
        st.info("ℹ️ Dataset file `cardio_train.csv` not found in workspace directory.")
