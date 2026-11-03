
# ============================================================
# SAT-GUARD AI  |  3D ORBITAL DEFENSE DASHBOARD
# Single-file Streamlit app
#
# Pages:
#   1. Mission Overview      (animated 3D Earth / satellites / debris)
#   2. YOLO Detection        (upload image -> detections -> visual features)
#   3. Orbital Monitoring    (TCA / conjunction explorer + 3D plots)
#   4. Risk Analysis + SHAP  (fusion model, uncertainty, explainability)
#   5. System Diagnostics
#
# Run:  streamlit run app.py
# Needs: streamlit ultralytics xgboost scikit-learn joblib shap plotly pillow
# ============================================================

import os
import sys
import json
import platform
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

#PROJECT_DIR = Path(os.environ.get("SATGUARD_DIR", "/content/drive/MyDrive/SAT_GUARD_AI"))
PROJECT_DIR = Path(os.environ.get("SATGUARD_DIR", Path(__file__).resolve().parent))
YOLO_PATH = PROJECT_DIR / "results/yolov8_debris/weights/best.pt"
FUSION_PATH = PROJECT_DIR / "models/fusion_risk_model.pkl"
FEATURES_PATH = PROJECT_DIR / "models/fusion_feature_columns.pkl"
TCA_PATH = PROJECT_DIR / "data/orbital/tca_results.csv"
ORBITAL_FEATURES_PATH = PROJECT_DIR / "data/orbital/real_orbital_features.csv"
VISUAL_AGG_PATH = PROJECT_DIR / "data/visual_features_aggregated.csv"

DEFAULT_FEATURES = [
    "miss_distance_km",
    "relative_velocity_km_s",
    "proximity_score",
    "relative_velocity_score",
    "visual_detection_count",
    "visual_mean_confidence",
    "visual_max_confidence",
    "visual_mean_bbox_area",
    "visual_max_bbox_area",
    "visual_mean_bbox_width",
    "visual_mean_bbox_height",
    "visual_mean_x_center",
    "visual_mean_y_center",
]

LOW_T, HIGH_T = 0.33, 0.66  # same thresholds as the notebook (Step 85/87)
RISK_COLORS = {"LOW": "#2bff9a", "MEDIUM": "#ffc547", "HIGH": "#ff4560"}

st.set_page_config(
    page_title="SAT-GUARD AI | Orbital Defense",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# THEME
# ------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, #10284a 0%, transparent 38%),
    radial-gradient(ellipse at 92% 8%, #1a1244 0%, transparent 34%),
    #030711;
  color: #e8f2ff;
  font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] { background: rgba(3,7,17,.85); }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1650px; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 1.2px; }
h1 { color: #6ce8ff; text-shadow: 0 0 18px rgba(40,190,255,.45); }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #081528, #050914);
  border-right: 1px solid #1b3a5a;
}
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #0b1b30, #091120);
  border: 1px solid #214362; border-radius: 12px; padding: 14px 16px;
  box-shadow: 0 0 18px rgba(30,140,220,.10);
}
[data-testid="stMetricLabel"] { color: #8daac7 !important; }
[data-testid="stMetricValue"] { color: #55e6ff !important; font-family: 'Orbitron', sans-serif; }

.stButton button, .stDownloadButton button {
  background: linear-gradient(90deg, #087fbe, #5448d8);
  color: #fff; border: 1px solid #4dcfff; border-radius: 8px; font-weight: 600;
}
.stButton button:hover { box-shadow: 0 0 16px rgba(77,207,255,.55); border-color: #9be9ff; }
hr { border-color: #183451; }

.panel {
  background: linear-gradient(135deg, rgba(11,27,48,.96), rgba(5,12,26,.96));
  border: 1px solid #1e4666; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
}
.alert-box {
  border-radius: 12px; padding: 16px 20px; margin: 10px 0 16px 0;
  font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
}
.badge {
  display:inline-block; padding: 3px 12px; border-radius: 20px; font-size: 12px;
  border: 1px solid; font-family: 'Orbitron', sans-serif;
}
.muted { color: #86a2bf; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# VERSION-SAFE DISPLAY HELPERS
# (Streamlit renamed use_container_width -> width="stretch")
# ------------------------------------------------------------


def _stretch(fn, *args, **kwargs):
    """Call Streamlit display functions across old/new Streamlit versions.

    Some page functions pass use_container_width=True. Remove it here so the
    newer width="stretch" path never receives duplicate width arguments.
    """
    kwargs.pop("use_container_width", None)
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def show_df(df, **kw):
    return _stretch(st.dataframe, df, **kw)


def show_plot(fig, **kw):
    return _stretch(st.plotly_chart, fig, **kw)


def show_img(img, **kw):
    return _stretch(st.image, img, **kw)


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,10,24,.75)",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(color="#6ce8ff", family="Orbitron"),
    )
    return fig


# ------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------



@st.cache_resource(show_spinner=False)
def load_yolo(path):
    import traceback

    model_path = Path(path).expanduser().resolve()

    print(f"YOLO model path: {model_path}", flush=True)
    print(f"File exists: {model_path.is_file()}", flush=True)

    if not model_path.is_file():
        print(f"YOLO FILE NOT FOUND: {model_path}", flush=True)
        print(f"Parent folder exists: {model_path.parent.is_dir()}", flush=True)

        if model_path.parent.is_dir():
            print(
                f"Parent folder contents: {list(model_path.parent.iterdir())}",
                flush=True
            )
        return None

    print(
        f"YOLO file size: {model_path.stat().st_size / (1024 * 1024):.2f} MB",
        flush=True
    )

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO

        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"Ultralytics version: {ultralytics.__version__}", flush=True)

        model = YOLO(str(model_path))

        print("YOLO MODEL LOADED SUCCESSFULLY!", flush=True)
        return model

    except Exception as exc:
        print(f"YOLO LOAD ERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return None


@st.cache_resource(show_spinner=False)
def load_joblib(path):
    if not Path(path).is_file():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Could not load {path}: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_explainer(_model):
    try:
        import shap
        return shap.TreeExplainer(_model)
    except Exception as exc:
        print("SHAP explainer failed:", exc)
        return None


@st.cache_data(show_spinner=False)
def load_tca(path, mtime):
    if not Path(path).is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "tca" in df.columns:
        df["tca"] = pd.to_datetime(df["tca"], errors="coerce", utc=True)
    for c in ("miss_distance_km", "relative_velocity_km_s"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "screening_flag" in df.columns:
        df["screening_flag"] = (
            df["screening_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    return df.dropna(subset=["miss_distance_km", "relative_velocity_km_s"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv(path, mtime):
    if not Path(path).is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def mtime(p):
    return Path(p).stat().st_mtime if Path(p).exists() else 0


yolo_model = load_yolo(YOLO_PATH)
fusion_model = load_joblib(FUSION_PATH)
feature_columns = load_joblib(FEATURES_PATH)
feature_columns = list(feature_columns) if feature_columns is not None else DEFAULT_FEATURES
tca_df = load_tca(str(TCA_PATH), mtime(TCA_PATH))
orbital_feat_df = load_csv(str(ORBITAL_FEATURES_PATH), mtime(ORBITAL_FEATURES_PATH))
visual_agg_df = load_csv(str(VISUAL_AGG_PATH), mtime(VISUAL_AGG_PATH))

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

for key, default in [
    ("visual_features", None),
    ("detection_table", None),
    ("detection_image", None),
    ("last_result", None),
    ("selected_conjunction", None),
    ("mission_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------

ZERO_VISUAL = {
    "visual_detection_count": 0.0,
    "visual_mean_confidence": 0.0,
    "visual_max_confidence": 0.0,
    "visual_mean_bbox_area": 0.0,
    "visual_max_bbox_area": 0.0,
    "visual_mean_bbox_width": 0.0,
    "visual_mean_bbox_height": 0.0,
    "visual_mean_x_center": 0.0,
    "visual_mean_y_center": 0.0,
}


def training_visual_defaults():
    """Same constant visual summary the fusion model was trained with (mean of
    visual_features_aggregated.csv). Falls back to zeros if file is missing."""
    if visual_agg_df.empty:
        return dict(ZERO_VISUAL)
    cols = [
        "detection_count", "mean_confidence", "max_confidence", "mean_bbox_area",
        "max_bbox_area", "mean_bbox_width", "mean_bbox_height",
        "mean_x_center", "mean_y_center",
    ]
    out = {}
    for c in cols:
        out["visual_" + c] = float(visual_agg_df[c].mean()) if c in visual_agg_df.columns else 0.0
    return out


def extract_visual_features(result):
    """Pixel-space aggregates, identical to the notebook's visual pipeline."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dict(ZERO_VISUAL), []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    area = w * h
    xc = xyxy[:, 0] + w / 2
    yc = xyxy[:, 1] + h / 2

    feats = {
        "visual_detection_count": float(len(xyxy)),
        "visual_mean_confidence": float(conf.mean()),
        "visual_max_confidence": float(conf.max()),
        "visual_mean_bbox_area": float(area.mean()),
        "visual_max_bbox_area": float(area.max()),
        "visual_mean_bbox_width": float(w.mean()),
        "visual_mean_bbox_height": float(h.mean()),
        "visual_mean_x_center": float(xc.mean()),
        "visual_mean_y_center": float(yc.mean()),
    }

    names = getattr(result, "names", {})
    rows = []
    for i in range(len(xyxy)):
        rows.append({
            "class": names.get(int(cls[i]), str(cls[i])),
            "confidence": round(float(conf[i]), 4),
            "x1": round(float(xyxy[i, 0]), 1), "y1": round(float(xyxy[i, 1]), 1),
            "x2": round(float(xyxy[i, 2]), 1), "y2": round(float(xyxy[i, 3]), 1),
            "width": round(float(w[i]), 1), "height": round(float(h[i]), 1),
            "area": round(float(area[i]), 1),
        })
    return feats, rows


def build_input(miss_km, rel_vel, visual):
    row = {
        "miss_distance_km": float(miss_km),
        "relative_velocity_km_s": float(rel_vel),
        # Same definitions used when the multimodal dataset was created
        "proximity_score": 1.0 / (float(miss_km) + 1e-6),
        "relative_velocity_score": float(rel_vel),
    }
    row.update(visual)
    return pd.DataFrame([[row.get(c, 0.0) for c in feature_columns]], columns=feature_columns)


def _class_labels(model):
    """Return model class labels in predict_proba column order."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return []
    return [str(c).upper() for c in list(classes)]


def prob_high(model, X):
    """Probability for the class named HIGH, respecting predict_proba order."""
    p = np.asarray(model.predict_proba(X))
    if p.ndim == 1:
        return p.reshape(-1)
    labels = _class_labels(model)
    if "HIGH" in labels and p.shape[1] == len(labels):
        return p[:, labels.index("HIGH")]
    # Binary models with nonstandard labels: use positive-class probability.
    if p.shape[1] == 2:
        return p[:, 1]
    # Last-resort fallback is explicit and avoids crashing on unknown wrappers.
    return p.max(axis=1)


def predict_risk_class(model, X):
    """Return the classifier's predicted class, normalized to LOW/MEDIUM/HIGH."""
    try:
        pred = str(np.asarray(model.predict(X)).reshape(-1)[0]).upper()
        if pred in {"LOW", "MEDIUM", "HIGH"}:
            return pred
    except Exception:
        pass
    p_high = float(prob_high(model, X)[0])
    if p_high < LOW_T:
        return "LOW"
    if p_high < HIGH_T:
        return "MEDIUM"
    return "HIGH"


def estimate_uncertainty(model, X, n_iter=30, rel_noise=0.01, seed=42):
    """Input-perturbation uncertainty, same method as Step 87 of the notebook."""
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(n_iter):
        Xp = X.copy()
        for c in X.columns:
            v = float(Xp.loc[0, c])
            Xp.loc[0, c] = v + rng.normal(0, max(abs(v) * rel_noise, 1e-6))
        probs.append(float(prob_high(model, Xp)[0]))
    probs = np.array(probs)
    unc = float(probs.std())
    conf = float(np.clip(1 - unc / 0.5, 0, 1))
    return float(probs.mean()), unc, conf


def shap_contributions(model, X):
    """SHAP contributions for the predicted class (or HIGH if class output is unavailable)."""
    explainer = load_explainer(model)
    if explainer is None:
        return None
    try:
        sv = explainer(X)
        raw = getattr(sv, "values", sv)
        # Some SHAP versions return a list (one array per class).
        if isinstance(raw, list):
            labels = _class_labels(model)
            try:
                class_idx = labels.index("HIGH")
            except ValueError:
                class_idx = len(raw) - 1
            vals = np.asarray(raw[class_idx])[0]
        else:
            vals = np.asarray(raw)
            if vals.ndim == 3:
                labels = _class_labels(model)
                try:
                    class_idx = labels.index("HIGH")
                except ValueError:
                    class_idx = vals.shape[-1] - 1
                vals = vals[0, :, class_idx]
            elif vals.ndim == 2:
                vals = vals[0]
            else:
                vals = vals.reshape(-1)
        vals = np.asarray(vals).reshape(-1)
        n = min(len(X.columns), len(vals))
        df = pd.DataFrame({
            "feature": list(X.columns)[:n],
            "value": X.iloc[0].values[:n],
            "shap_value": vals[:n],
        })
        df["abs"] = df["shap_value"].abs()
        df["effect"] = np.where(df["shap_value"] > 0,
                                "Pushes selected output higher", "Pushes selected output lower")
        return df.sort_values("abs", ascending=False).reset_index(drop=True)
    except Exception as exc:
        print("SHAP failed:", exc)
        return None


def risk_class(value):
    """Map a HIGH-class probability to the dashboard's display bands."""
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 0.0
    probability = max(0.0, min(1.0, probability))
    if probability < LOW_T:
        return "LOW"
    if probability < HIGH_T:
        return "MEDIUM"
    return "HIGH"

def gauge(value, title):
    color = RISK_COLORS[risk_class(value)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"color": color, "family": "Orbitron"}},
        title={"text": title, "font": {"color": "#8daac7", "size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6a8aa8"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1, "bordercolor": "#214362",
            "steps": [
                {"range": [0, LOW_T * 100], "color": "rgba(43,255,154,.13)"},
                {"range": [LOW_T * 100, HIGH_T * 100], "color": "rgba(255,197,71,.13)"},
                {"range": [HIGH_T * 100, 100], "color": "rgba(255,69,96,.16)"},
            ],
        },
    ))
    return style_fig(fig, 300)


# ------------------------------------------------------------
# 3D ANIMATED SPACE SCENE (Three.js)
# ------------------------------------------------------------

SCENE_HTML = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box} html,body{margin:0;background:#020611;overflow:hidden;font-family:'Courier New',monospace;color:#bfeaff}
#scene{width:100%;height:__H__px;cursor:grab;position:relative}
#scene:active{cursor:grabbing}
.hud{position:absolute;z-index:3;padding:10px 14px;border:1px solid #1b83a8;background:rgba(3,12,29,.72);
 border-radius:8px;font-size:11px;line-height:1.75;letter-spacing:1px;pointer-events:none;backdrop-filter:blur(3px)}
.tl{top:12px;left:12px}.tr{top:12px;right:12px;text-align:right;border-color:#4c3fb8}
.bl{bottom:12px;left:12px;font-size:10px;color:#7b95b4;max-width:60%}
.br{bottom:12px;right:12px;text-align:right;border-color:#a8283c;color:#ff9aa8}
.k{color:#51f6ff;font-weight:bold}.ok{color:#70ffbd}.warn{color:#ff6378}
.scan{position:absolute;inset:0;z-index:2;pointer-events:none;
 background:repeating-linear-gradient(0deg,rgba(80,180,255,.025) 0 1px,transparent 1px 3px)}
</style></head><body>
<div id="scene">
 <div class="scan"></div>
 <div class="hud tl"><div class="k">SAT-GUARD // ORBITAL VIEW</div>
  <div>LINK: <span class="ok">NOMINAL</span></div>
  <div>TRACKED OBJECTS: <span id="obj">--</span></div>
  <div>TCA RECORDS: __RECORDS__</div>
  <div>SCREENED (&le;50 km): <span class="warn">__FLAGGED__</span></div></div>
 <div class="hud tr"><div class="k">ORBITAL SURVEILLANCE</div>
  <div style="color:#a9b9ff">EARTH &middot; SATELLITES &middot; DEBRIS</div>
  <div id="clock"></div><div>EARTH ROT: <span id="rot">0</span></div></div>
 <div class="hud br"><div class="warn">&#9888; CONJUNCTION WATCH</div>
  <div>CLOSEST TCA IN DATA: __MINMISS__</div>
  <div id="cdist">SIM RANGE: --</div></div>
 <div class="hud bl">Drag to rotate &middot; scroll to zoom. Animated illustration &mdash; positions are simulated, not live telemetry.</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
(function(){
const host=document.getElementById('scene');
if(!window.THREE){host.insertAdjacentHTML('beforeend','<div style="padding:120px;text-align:center;color:#72dfff">3D library could not load (internet access required).</div>');return;}
const W=()=>host.clientWidth,H=()=>host.clientHeight;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x020611);
const camera=new THREE.PerspectiveCamera(45,W()/H(),0.1,1000);camera.position.set(0,4,18);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(W(),H());
host.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0x7f9fc4,1.4));
const sun=new THREE.DirectionalLight(0xffffff,3.2);sun.position.set(-9,5,8);scene.add(sun);

// stars (fixed in world, not rotated by user)
const sg=new THREE.BufferGeometry(),sp=[];
for(let i=0;i<3200;i++){sp.push((Math.random()-.5)*220,(Math.random()-.5)*150,(Math.random()-.5)*220);}
sg.setAttribute('position',new THREE.Float32BufferAttribute(sp,3));
const stars=new THREE.Points(sg,new THREE.PointsMaterial({color:0xbad8ff,size:0.15,transparent:true,opacity:.9}));
scene.add(stars);

const world=new THREE.Group();scene.add(world);

// Earth
const earth=new THREE.Mesh(new THREE.SphereGeometry(4,96,96),
 new THREE.MeshPhongMaterial({color:0x1a5f9d,specular:0x417ea5,shininess:26}));
world.add(earth);
new THREE.TextureLoader().load('https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg',
 t=>{t.colorSpace=THREE.SRGBColorSpace;earth.material.map=t;earth.material.color.set(0xffffff);earth.material.needsUpdate=true;},
 undefined,()=>{});
const cloud=new THREE.Mesh(new THREE.SphereGeometry(4.04,64,64),
 new THREE.MeshPhongMaterial({color:0xcdeeff,transparent:true,opacity:.09,depthWrite:false}));
world.add(cloud);
// atmosphere glow (fresnel shader)
const atmo=new THREE.Mesh(new THREE.SphereGeometry(4.45,64,64),new THREE.ShaderMaterial({
 transparent:true,side:THREE.BackSide,blending:THREE.AdditiveBlending,depthWrite:false,
 vertexShader:'varying vec3 n;void main(){n=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
 fragmentShader:'varying vec3 n;void main(){float i=pow(.72-dot(n,vec3(0,0,1.)),3.);gl_FragColor=vec4(.25,.62,1.,1.)*i*1.6;}'}));
world.add(atmo);

// orbit rings
const specs=[{r:5.3,c:0x25dfff,t:.35},{r:6.2,c:0x617cff,t:-.55},{r:7.1,c:0x28b8a5,t:.8},{r:7.9,c:0x3d5bb0,t:-.2}];
specs.forEach(o=>{const m=new THREE.Mesh(new THREE.TorusGeometry(o.r,.012,8,260),
 new THREE.MeshBasicMaterial({color:o.c,transparent:true,opacity:.5}));
 m.rotation.x=Math.PI/2+o.t;m.rotation.y=o.t*.35;world.add(m);});

function makeSat(col){
 const g=new THREE.Group();
 g.add(new THREE.Mesh(new THREE.BoxGeometry(.38,.38,.5),new THREE.MeshStandardMaterial({color:col,metalness:.8,roughness:.3})));
 const pm=new THREE.MeshStandardMaterial({color:0x1253a0,metalness:.4,roughness:.35,emissive:0x071e50});
 const p1=new THREE.Mesh(new THREE.BoxGeometry(1.05,.045,.42),pm);p1.position.x=-.78;g.add(p1);
 const p2=p1.clone();p2.position.x=.78;g.add(p2);
 const a=new THREE.Mesh(new THREE.CylinderGeometry(.018,.018,.5,8),new THREE.MeshStandardMaterial({color:0xe7f4ff}));
 a.position.y=.37;g.add(a);
 const dish=new THREE.Mesh(new THREE.SphereGeometry(.12,12,8,0,Math.PI*2,0,Math.PI/2),new THREE.MeshStandardMaterial({color:0xffffff,metalness:.6}));
 dish.position.set(0,-.25,.2);dish.rotation.x=Math.PI;g.add(dish);
 return g;}

const sats=[];
for(let i=0;i<6;i++){
 const m=makeSat(i===0?0xffd78b:0xdde8f5);world.add(m);
 sats.push({m,r:5.3+i*.5,a:i*1.1,s:.12+i*.022,t:.4+i*.13});}

const debris=[];
for(let i=0;i<__NDEB__;i++){
 const size=.05+Math.random()*.1;
 const m=new THREE.Mesh(new THREE.IcosahedronGeometry(size,0),
  new THREE.MeshStandardMaterial({color:i%4===0?0xff5d68:0xc6d6e6,emissive:i%4===0?0x6c1018:0x172638,metalness:.65,roughness:.4}));
 world.add(m);
 debris.push({m,r:4.8+Math.random()*3.2,a:Math.random()*6.28,s:(Math.random()<.5?-1:1)*(.09+Math.random()*.14),t:(Math.random()-.5)*1.6});}
document.getElementById('obj').textContent=String(sats.length+debris.length).padStart(3,'0');

// debris trail for a few red pieces
function pos(o){return new THREE.Vector3(Math.cos(o.a)*o.r,Math.sin(o.a)*o.r*Math.sin(o.t),Math.sin(o.a)*o.r*Math.cos(o.t));}

// conjunction link: satellite 0 <-> debris 0
const linkGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
const link=new THREE.Line(linkGeo,new THREE.LineBasicMaterial({color:0xff3d58,transparent:true,opacity:.8}));
world.add(link);
const ring=new THREE.Mesh(new THREE.RingGeometry(.28,.32,40),new THREE.MeshBasicMaterial({color:0xff3d58,transparent:true,opacity:.8,side:THREE.DoubleSide}));
world.add(ring);

// interaction
let rx=.25,ry=0,vx=0,vy=.0016,drag=false,lx=0,ly=0,zoom=18;
const el=renderer.domElement;
el.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{if(!drag)return;vy=(e.clientX-lx)*.0006;vx=(e.clientY-ly)*.0006;
 ry+=(e.clientX-lx)*.005;rx+=(e.clientY-ly)*.005;rx=Math.max(-1.2,Math.min(1.2,rx));lx=e.clientX;ly=e.clientY;});
el.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(9,Math.min(32,zoom+e.deltaY*.01));},{passive:false});

let t=0;
function animate(){
 requestAnimationFrame(animate);t+=.006;
 if(!drag){ry+=vy;vy+=(.0016-vy)*.02;}
 world.rotation.set(rx,ry,0);
 earth.rotation.y+=.0017;cloud.rotation.y+=.0021;stars.rotation.y+=.00008;
 sats.forEach(s=>{s.a+=s.s*.006;const p=pos(s);s.m.position.copy(p);s.m.rotation.y+=.008;});
 debris.forEach(d=>{d.a+=d.s*.006;d.m.position.copy(pos(d));d.m.rotation.x+=.01;d.m.rotation.y+=.012;});
 const A=sats[0].m.position,B=debris[0].m.position;
 link.geometry.setFromPoints([A,B]);
 link.material.opacity=.45+.4*Math.sin(t*6);
 ring.position.copy(B);ring.lookAt(camera.position);
 ring.scale.setScalar(1+.4*Math.sin(t*5));
 document.getElementById('cdist').textContent='SIM RANGE: '+(A.distanceTo(B)*1000).toFixed(0)+' (arb. units)';
 camera.position.z+=(zoom-camera.position.z)*.08;
 camera.position.x=Math.sin(t*.1)*.6;camera.lookAt(0,0,0);
 renderer.render(scene,camera);
 document.getElementById('clock').textContent=new Date().toUTCString().slice(17,25)+' UTC';
 document.getElementById('rot').textContent=((t*10)%360).toFixed(1)+'\u00B0';
}
addEventListener('resize',()=>{camera.aspect=W()/H();camera.updateProjectionMatrix();renderer.setSize(W(),H());});
animate();
})();
</script></body></html>
"""


def render_space_scene(height=560, n_debris=36):
    n_rec = f"{len(tca_df):,}" if not tca_df.empty else "N/A"
    flagged = (
        f"{int(tca_df['screening_flag'].sum()):,}"
        if (not tca_df.empty and "screening_flag" in tca_df.columns) else "N/A"
    )
    min_miss = (
        f"{tca_df['miss_distance_km'].min():.2f} km" if not tca_df.empty else "N/A"
    )
    html = (
        SCENE_HTML.replace("__H__", str(height))
        .replace("__RECORDS__", n_rec)
        .replace("__FLAGGED__", flagged)
        .replace("__MINMISS__", min_miss)
        .replace("__NDEB__", str(n_debris))
    )
    components.html(html, height=height + 6, scrolling=False)


def page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


def risk_banner(label, prob):
    color = RISK_COLORS[label]
    icon = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[label]
    st.markdown(
        f"""<div class="alert-box" style="border:1px solid {color};
        background:linear-gradient(90deg, {color}22, transparent);">
        <span style="font-size:22px;color:{color}">{icon} COLLISION RISK: {label}</span>
        <span class="muted" style="margin-left:18px">model probability {prob:.1%}</span></div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🛰️ SAT-GUARD AI")
    st.caption("SPACE SITUATIONAL AWARENESS SYSTEM")
    st.markdown("---")
    page = st.radio(
        "MISSION CONTROL",
        [
            "Mission Overview",
            "YOLO Detection",
            "Orbital Monitoring",
            "Risk Analysis + SHAP",
            "Research Results",
            "Dataset & Feature Analysis",
            "System Diagnostics",
        ],
    )
    st.markdown("---")
    st.markdown("### MODEL STATUS")
    st.write("🟢 YOLO model loaded" if yolo_model is not None else "🔴 YOLO model missing")
    st.write("🟢 Fusion model loaded" if fusion_model is not None else "🔴 Fusion model missing")
    st.write("🟢 Orbital data loaded" if not tca_df.empty else "🔴 Orbital data missing")
    st.markdown("---")
    st.caption("Research visualization — not an operational collision-avoidance system.")

# ------------------------------------------------------------
# PAGE 1 — MISSION OVERVIEW
# ------------------------------------------------------------


def page_overview():
    page_header("SAT-GUARD AI", "Space debris detection · orbital conjunction screening · explainable risk")
    render_space_scene(height=560)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TCA Records", f"{len(tca_df):,}")
    if not tca_df.empty:
        c2.metric("Min Miss Distance", f"{tca_df['miss_distance_km'].min():.2f} km")
        c3.metric("Median Rel. Velocity", f"{tca_df['relative_velocity_km_s'].median():.2f} km/s")
        flagged = int(tca_df["screening_flag"].sum()) if "screening_flag" in tca_df.columns else 0
        c4.metric("Screened ≤ 50 km", f"{flagged:,}")
    else:
        c2.metric("Min Miss Distance", "N/A")
        c3.metric("Median Rel. Velocity", "N/A")
        c4.metric("Screened ≤ 50 km", "N/A")
    c5.metric("Fusion Model", "ONLINE" if fusion_model is not None else "OFFLINE")

    if tca_df.empty:
        st.warning(f"Orbital data not found: {TCA_PATH}")
        return

    st.markdown("### Closest approaches")
    cols = [c for c in ["object_1", "object_2", "tca", "miss_distance_km",
                        "relative_velocity_km_s", "screening_flag"] if c in tca_df.columns]
    show_df(tca_df.nsmallest(10, "miss_distance_km")[cols], hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = px.histogram(tca_df, x="miss_distance_km", nbins=60,
                           title="Miss Distance Distribution (km)",
                           color_discrete_sequence=["#39dfff"])
        show_plot(style_fig(fig, 340))
    with right:
        fig = px.histogram(tca_df, x="relative_velocity_km_s", nbins=60,
                           title="Relative Velocity Distribution (km/s)",
                           color_discrete_sequence=["#8d7bff"])
        show_plot(style_fig(fig, 340))

    # Model-based risk distribution on a sample
    if fusion_model is not None:
        st.markdown("### Fusion-model risk distribution")
        try:
            sample = tca_df.sample(min(len(tca_df), 20000), random_state=42)
            vis = training_visual_defaults()
            X = pd.DataFrame({
                "miss_distance_km": sample["miss_distance_km"].values,
                "relative_velocity_km_s": sample["relative_velocity_km_s"].values,
            })
            X["proximity_score"] = 1.0 / (X["miss_distance_km"] + 1e-6)
            X["relative_velocity_score"] = X["relative_velocity_km_s"]
            for k, v in vis.items():
                X[k] = v
            X = X[feature_columns]
            predicted = np.asarray(fusion_model.predict(X)).reshape(-1)
            cats = pd.Series([str(x).upper() for x in predicted]).value_counts()
            cats = cats.reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0).reset_index()
            cats.columns = ["Risk", "Count"]
            fig = px.bar(cats, x="Risk", y="Count", color="Risk",
                         color_discrete_map=RISK_COLORS,
                         title=f"Risk classes (sample of {len(sample):,} conjunctions)")
            show_plot(style_fig(fig, 330))
        except Exception as exc:
            st.info(f"Risk distribution unavailable: {exc}")

    if st.session_state.mission_log:
        st.markdown("### Mission log")
        show_df(pd.DataFrame(st.session_state.mission_log[::-1]), hide_index=True)


# ------------------------------------------------------------
# PAGE 2 — YOLO DETECTION
# ------------------------------------------------------------


def page_detection():
    page_header("YOLO Detection", "Detect satellites and debris in an image; extract the visual features used by the fusion model")

    if yolo_model is None:
        st.error(f"YOLO weights not found or failed to load: {YOLO_PATH}")
        return

    left, right = st.columns([1, 1])
    with left:
        up = st.file_uploader("Upload a space image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    with right:
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        imgsz = st.select_slider("Inference size", options=[320, 480, 640, 800], value=640)

    if up is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(up).convert("RGB")
    c1, c2 = st.columns(2)
    with c1:
        show_img(image, caption="Input")

    if st.button("🚀 Run detection", type="primary"):
        with st.spinner("Scanning..."):
            res = yolo_model.predict(source=np.array(image), conf=conf, imgsz=imgsz, verbose=False)[0]
        annotated = Image.fromarray(res.plot()[:, :, ::-1])
        feats, rows = extract_visual_features(res)
        st.session_state.visual_features = feats
        st.session_state.detection_table = rows
        st.session_state.detection_image = (up.name, annotated)

    if st.session_state.detection_image is not None:
        name, annotated = st.session_state.detection_image
        with c2:
            show_img(annotated, caption=f"Detections — {name}")

        rows = st.session_state.detection_table or []
        feats = st.session_state.visual_features or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects detected", len(rows))
        m2.metric("Mean confidence", f"{feats.get('visual_mean_confidence', 0):.3f}")
        m3.metric("Max bbox area (px²)", f"{feats.get('visual_max_bbox_area', 0):,.0f}")

        if rows:
            df = pd.DataFrame(rows)
            show_df(df, hide_index=True)
            fig = px.bar(df.reset_index(), x="index", y="confidence", color="class",
                         title="Detection confidence", labels={"index": "detection"})
            show_plot(style_fig(fig, 320))
            st.download_button("⬇ Download detections CSV", df.to_csv(index=False),
                               "satguard_detections.csv", "text/csv")
        else:
            st.warning("No objects detected at this threshold.")

        with st.expander("Visual feature vector (sent to the fusion model)"):
            st.json(feats)
        st.success("Visual features saved — open **Risk Analysis + SHAP** to fuse them with an orbital conjunction.")


# ------------------------------------------------------------
# PAGE 3 — ORBITAL MONITORING
# ------------------------------------------------------------


def page_orbital():
    page_header("Orbital Monitoring", "Time-of-closest-approach (TCA) records and orbital element statistics")

    if tca_df.empty:
        st.error(f"Unable to load orbital data: {TCA_PATH}")
        return

    df = tca_df
    lo, hi = float(df["miss_distance_km"].min()), float(df["miss_distance_km"].max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        st.error("The TCA file has no valid miss-distance values.")
        return
    if hi <= lo:
        hi = lo + 1e-6

    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        default_hi = min(hi, lo + (hi - lo) * 0.1) if hi > lo else hi
        rng = st.slider("Miss distance range (km)", lo, hi, (lo, max(lo, default_hi)))
    with f2:
        only_flag = st.checkbox("Screened only (≤ 50 km)", value=False)
    with f3:
        text = st.text_input("Object name contains")

    view = df[df["miss_distance_km"].between(*rng)].copy()
    if only_flag:
        # The control is explicitly defined as a <=50 km close-approach filter.
        # Apply the physical threshold first; also respect a usable screening flag
        # if the CSV stores it as a recognized boolean/text value.
        view = view[view["miss_distance_km"] <= 50.0].copy()
        if "screening_flag" in view.columns and not view.empty:
            raw_flag = view["screening_flag"]
            if pd.api.types.is_bool_dtype(raw_flag):
                flag_mask = raw_flag.fillna(False)
                view = view[flag_mask].copy()
            else:
                normalized = raw_flag.astype(str).str.strip().str.lower()
                true_tokens = {"true", "1", "yes", "y", "screened", "flagged", "pass", "passed"}
                false_tokens = {"false", "0", "no", "n", "not screened", "nan", "none", ""}
                recognized = normalized.isin(true_tokens | false_tokens)
                # Only apply the flag column when it contains interpretable values.
                # Some exports store a descriptive screening status instead.
                if recognized.any() and normalized.isin(true_tokens).any():
                    view = view[normalized.isin(true_tokens)].copy()
                elif recognized.any() and not normalized.isin(true_tokens).any():
                    st.caption("Screening flag has no recognized positive values; showing records within 50 km based on miss distance.")
    if text.strip():
        mask = pd.Series(False, index=view.index)
        for c in ("object_1", "object_2"):
            if c in view.columns:
                mask |= view[c].astype(str).str.contains(text.strip(), case=False, na=False)
        view = view[mask]

    m1, m2, m3 = st.columns(3)
    m1.metric("Matching records", f"{len(view):,}")
    m2.metric("Closest", f"{view['miss_distance_km'].min():.3f} km" if len(view) else "N/A")
    m3.metric("Fastest", f"{view['relative_velocity_km_s'].max():.2f} km/s" if len(view) else "N/A")

    if len(view):
        plot_df = view.sample(min(len(view), 6000), random_state=42).copy()
        plot_df["proximity"] = 1.0 / (plot_df["miss_distance_km"] + 1e-6)
        if "tca" in plot_df.columns and plot_df["tca"].notna().any():
            plot_df["minutes_from_start"] = (
                (plot_df["tca"] - plot_df["tca"].min()).dt.total_seconds() / 60.0
            )
        else:
            plot_df["minutes_from_start"] = np.arange(len(plot_df))
        hover = [c for c in ["object_1", "object_2"] if c in plot_df.columns]

        fig = go.Figure(go.Scatter3d(
            x=plot_df["miss_distance_km"],
            y=plot_df["relative_velocity_km_s"],
            z=plot_df["minutes_from_start"],
            mode="markers",
            marker=dict(size=3.2, color=np.log10(plot_df["proximity"] + 1e-9),
                        colorscale="Turbo", opacity=.85,
                        colorbar=dict(title="log proximity")),
            text=plot_df[hover].astype(str).agg(" × ".join, axis=1) if hover else None,
            hovertemplate="%{text}<br>miss %{x:.2f} km<br>v_rel %{y:.2f} km/s<br>t+%{z:.0f} min<extra></extra>",
        ))
        fig.update_layout(
            title="3D Conjunction Space (miss distance · relative velocity · time)",
            scene=dict(
                xaxis_title="Miss distance (km)", yaxis_title="Rel. velocity (km/s)", zaxis_title="Minutes",
                xaxis=dict(backgroundcolor="#050b18", gridcolor="#173250"),
                yaxis=dict(backgroundcolor="#050b18", gridcolor="#173250"),
                zaxis=dict(backgroundcolor="#050b18", gridcolor="#173250"),
            ),
        )
        show_plot(style_fig(fig, 560))

        fig2 = px.scatter(plot_df, x="miss_distance_km", y="relative_velocity_km_s", opacity=.6,
                          title="Miss Distance vs Relative Velocity",
                          color_discrete_sequence=["#39dfff"])
        show_plot(style_fig(fig2, 380))

        st.markdown("### Select a conjunction for risk analysis")
        selection_df = view.reset_index(drop=True)
        option_labels = [
            f"#{i + 1} · {r.get('object_1', '?')} × {r.get('object_2', '?')} · "
            f"{float(r['miss_distance_km']):.3f} km · {float(r['relative_velocity_km_s']):.3f} km/s"
            for i, (_, r) in enumerate(selection_df.iterrows())
        ]
        selected_idx = st.selectbox(
            "Conjunction to send to Risk Analysis",
            options=list(range(len(option_labels))),
            format_func=lambda i: option_labels[i],
            key="orbital_conjunction_choice",
        )
        selected_row = selection_df.iloc[selected_idx]
        if st.button("Use selected conjunction in Risk Analysis", type="primary"):
            st.session_state.selected_conjunction = {
                "miss_distance_km": float(selected_row["miss_distance_km"]),
                "relative_velocity_km_s": float(selected_row["relative_velocity_km_s"]),
                "object_1": str(selected_row.get("object_1", "?")),
                "object_2": str(selected_row.get("object_2", "?")),
                "tca": str(selected_row.get("tca", "")),
                "screening_flag": str(selected_row.get("screening_flag", "")),
                "label": option_labels[selected_idx],
            }
            st.success("Conjunction saved. Open Risk Analysis + SHAP; this record will be preselected.")

        st.markdown("### Records")
        show_df(view.head(1000), hide_index=True)
        st.download_button("⬇ Download filtered records", view.to_csv(index=False),
                           "satguard_orbital_records.csv", "text/csv")

    if not orbital_feat_df.empty and {"altitude_km", "inclination_deg"}.issubset(orbital_feat_df.columns):
        st.markdown("### Real debris orbital elements (CelesTrak)")
        o = orbital_feat_df.copy()
        if len(o) > 5000:
            o = o.sample(5000, random_state=42)
        fig3 = go.Figure(go.Scatter3d(
            x=o["altitude_km"], y=o["inclination_deg"],
            z=o["eccentricity"] if "eccentricity" in o.columns else np.zeros(len(o)),
            mode="markers",
            marker=dict(size=3, color=o["altitude_km"], colorscale="Viridis", opacity=.8),
        ))
        fig3.update_layout(
            title="Altitude · Inclination · Eccentricity",
            scene=dict(xaxis_title="Altitude (km)", yaxis_title="Inclination (°)", zaxis_title="Eccentricity",
                       xaxis=dict(backgroundcolor="#050b18", gridcolor="#173250"),
                       yaxis=dict(backgroundcolor="#050b18", gridcolor="#173250"),
                       zaxis=dict(backgroundcolor="#050b18", gridcolor="#173250")),
        )
        show_plot(style_fig(fig3, 520))


# ------------------------------------------------------------
# PAGE 4 — RISK ANALYSIS + SHAP
# ------------------------------------------------------------


def page_risk():
    page_header("Risk Analysis + SHAP",
                "Fuse an orbital conjunction with visual detections · risk probability · uncertainty · explanation")

    if fusion_model is None:
        st.error(f"Fusion model could not be loaded: {FUSION_PATH}")
        return

    st.markdown("#### 1 · Orbital conjunction")
    mode = st.radio("Source", ["Pick from TCA records", "Enter manually"], horizontal=True)

    miss_km, rel_vel, label_txt = 25.0, 1.0, "manual input"
    if mode == "Pick from TCA records" and not tca_df.empty:
        n = st.slider("Show N closest approaches", 10, 500, 50, 10)
        closest = tca_df.nsmallest(n, "miss_distance_km").reset_index(drop=True)
        names = [
            f"#{i+1} · {r.get('object_1', '?')} × {r.get('object_2', '?')} · {r['miss_distance_km']:.2f} km"
            for i, r in closest.iterrows()
        ]
        saved = st.session_state.get("selected_conjunction")
        if saved:
            miss_km = float(saved["miss_distance_km"])
            rel_vel = float(saved["relative_velocity_km_s"])
            label_txt = str(saved.get("label", "Selected from Orbital Monitoring"))
            st.success(f"Using selected orbital record: {label_txt}")
            st.caption("The miss distance and relative velocity below are taken from the conjunction selected in Orbital Monitoring.")
            if st.button("Clear saved orbital selection"):
                st.session_state.selected_conjunction = None
                st.rerun()
        else:
            idx = st.selectbox("Conjunction", range(len(closest)),
                               format_func=lambda i: names[i], key="risk_tca_choice")
            row = closest.iloc[idx]
            miss_km = float(row["miss_distance_km"])
            rel_vel = float(row["relative_velocity_km_s"])
            label_txt = names[idx]
        c1, c2 = st.columns(2)
        c1.metric("Miss distance", f"{miss_km:.3f} km")
        c2.metric("Relative velocity", f"{rel_vel:.3f} km/s")
    else:
        c1, c2 = st.columns(2)
        miss_km = c1.number_input("Miss distance (km)", 0.0, 100000.0, 25.0, 1.0)
        rel_vel = c2.number_input("Relative velocity (km/s)", 0.0, 30.0, 1.0, 0.1)

    st.markdown("#### 2 · Visual evidence")
    options = ["Training-set average (neutral)", "No visual detections"]
    if st.session_state.visual_features is not None:
        options.insert(0, "Latest YOLO detection")
    vis_choice = st.radio("Visual features", options, horizontal=True)
    if vis_choice == "Latest YOLO detection":
        visual = st.session_state.visual_features
    elif vis_choice == "No visual detections":
        visual = dict(ZERO_VISUAL)
    else:
        visual = training_visual_defaults()

    with st.expander("Feature vector"):
        st.dataframe(build_input(miss_km, rel_vel, visual).T.rename(columns={0: "value"}))

    st.markdown("#### 3 · Analysis")
    n_iter = st.slider("Uncertainty iterations", 10, 100, 30, 10)

    if st.button("🛰️ Analyze risk", type="primary"):
        X = build_input(miss_km, rel_vel, visual)
        base_p = float(prob_high(fusion_model, X)[0])
        mean_p, unc, conf = estimate_uncertainty(fusion_model, X, n_iter=n_iter)
        label = predict_risk_class(fusion_model, X)
        contrib = shap_contributions(fusion_model, X)

        st.session_state.last_result = dict(label=label, p=mean_p, unc=unc, conf=conf,
                                            base=base_p, contrib=contrib, src=label_txt)
        st.session_state.mission_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "conjunction": label_txt[:60],
            "risk": label,
            "probability": round(mean_p, 4),
            "uncertainty": round(unc, 4),
        })

    r = st.session_state.last_result
    if r is None:
        return

    risk_banner(r["label"], r["p"])
    g1, g2 = st.columns([1.2, 1])
    with g1:
        show_plot(gauge(r["p"], "HIGH-class probability"))
    with g2:
        st.metric("Risk class", r["label"])
        st.metric("Uncertainty (σ)", f"{r['unc'] * 100:.2f}%")
        st.metric("Confidence", f"{r['conf'] * 100:.1f}%")
        if r["label"] == "HIGH" and r["unc"] >= 0.15:
            st.warning("High risk with high uncertainty — flag for analyst review.")

    st.markdown("### Explainable AI — SHAP")
    st.caption("SHAP contributions are shown for the HIGH class when supported by the model's SHAP output. Positive values push that output higher; negative values push it lower.")
    c = r["contrib"]
    if c is None:
        st.info("SHAP is unavailable (install `shap`).")
        if hasattr(fusion_model, "feature_importances_"):
            imp = pd.DataFrame({"feature": feature_columns,
                                "importance": fusion_model.feature_importances_}).sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         title="Model feature importance (fallback)")
            show_plot(style_fig(fig, 460))
        return

    plot_c = c.sort_values("abs", ascending=True)
    colors = ["#ff4560" if v > 0 else "#2bb5ff" for v in plot_c["shap_value"]]
    fig = go.Figure(go.Bar(x=plot_c["shap_value"], y=plot_c["feature"], orientation="h",
                           marker_color=colors,
                           hovertemplate="%{y}<br>SHAP %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Feature contributions for HIGH-class output (red ↑ · blue ↓)",
                      xaxis_title="SHAP value (model output scale)")
    show_plot(style_fig(fig, 480))

    st.markdown("**Top drivers**")
    for _, row in c.head(5).iterrows():
        arrow = "🔺" if row["shap_value"] > 0 else "🔻"
        st.write(f"{arrow} **{row['feature']}** = {row['value']:.4g} → {row['effect']} (SHAP {row['shap_value']:+.4f})")
    with st.expander("Full SHAP table"):
        show_df(c.drop(columns="abs"), hide_index=True)
        st.download_button("⬇ Download explanation CSV", c.to_csv(index=False),
                           "satguard_shap_explanation.csv", "text/csv")

    st.caption("Research limitation: if the fusion model was trained on proximity-screening labels (miss distance ≤ 50 km), miss distance may dominate the result. Treat this as a research indicator, not operational collision-avoidance guidance.")


# ============================================================
# 5. RESEARCH RESULTS
# ============================================================

def research_results():
    page_header(
        "Research Results & Model Comparison",
        "Evaluation metrics from the SAT-GUARD AI experiments."
    )

    # Keep reported project metrics local to this function.
    yolo_metrics = {
        "Precision": 0.83964036,
        "Recall": 0.7876610,
        "mAP@50": 0.8583596,
        "mAP@50–95": 0.5233342,
    }

    fusion_metrics = {
        "Accuracy": 0.9983,
        "Precision": 0.0556,
        "Recall": 0.3000,
        "Macro F1": 0.0938,
        "ROC-AUC": 0.9980,
    }

    st.subheader("YOLOv8 Object Detection")

    show_df(
        pd.DataFrame([yolo_metrics]).round(4),
        hide_index=True
    )

    yolo_df = pd.DataFrame({
        "Metric": list(yolo_metrics.keys()),
        "Score": list(yolo_metrics.values()),
    })

    fig = px.bar(
        yolo_df,
        x="Metric",
        y="Score",
        title="YOLOv8 Evaluation Metrics",
        range_y=[0, 1],
        color="Score",
        color_continuous_scale=["#17456b", "#00d9ff"],
    )
    fig.update_traces(
        text=[f"{v:.3f}" for v in yolo_df["Score"]],
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(showlegend=False)
    show_plot(style_fig(fig, height=400), use_container_width=True)

    st.divider()
    st.subheader("Multimodal Fusion Classifier")

    show_df(
        pd.DataFrame([fusion_metrics]).round(4),
        hide_index=True
    )

    fusion_df = pd.DataFrame({
        "Metric": list(fusion_metrics.keys()),
        "Score": list(fusion_metrics.values()),
    })

    fig = px.bar(
        fusion_df,
        x="Metric",
        y="Score",
        title="Fusion XGBoost Evaluation Metrics",
        range_y=[0, 1],
        color="Score",
        color_continuous_scale=["#572c45", "#ff4560"],
    )
    fig.update_traces(
        text=[f"{v:.4f}" for v in fusion_df["Score"]],
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(showlegend=False)
    show_plot(style_fig(fig, height=420), use_container_width=True)

    st.warning(
        "Research interpretation: the fusion model has high reported "
        "accuracy (99.83%), but precision is 5.56% and macro F1 is 9.38%. "
        "Accuracy alone is misleading for an imbalanced dataset. Review "
        "per-class precision, recall, F1, the prediction threshold, and "
        "the confusion matrix before drawing conclusions."
    )

    st.divider()
    st.subheader("Reported Model Metrics Comparison")

    comparison_rows = []

    for metric, score in yolo_metrics.items():
        comparison_rows.append({
            "Model": "YOLOv8",
            "Metric": metric,
            "Score": score,
        })

    for metric, score in fusion_metrics.items():
        comparison_rows.append({
            "Model": "Fusion XGBoost",
            "Metric": metric,
            "Score": score,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    fig = px.bar(
        comparison_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0, 1],
        title="Reported Evaluation Scores",
        color_discrete_map={
            "YOLOv8": "#00d9ff",
            "Fusion XGBoost": "#ff4560",
        },
    )
    fig.update_layout(xaxis_tickangle=-25)
    show_plot(style_fig(fig, height=470), use_container_width=True)

    st.caption(
        "YOLO detection metrics and fusion classification metrics measure "
        "different tasks. Their scores should not be interpreted as a "
        "direct head-to-head model ranking."
    )

    st.download_button(
        "Download reported metrics CSV",
        comparison_df.to_csv(index=False).encode("utf-8"),
        "satguard_research_metrics.csv",
        "text/csv",
        key="research_metrics_download",
    )

    # Discover saved evaluation artifacts, if available.
    st.divider()
    st.subheader("Saved Confusion Matrices & Classification Reports")

    artifact_terms = (
        "confusion_matrix",
        "confusion-matrix",
        "classification_report",
        "classification-report",
        "results.csv",
        "metrics.json",
    )

    found_files = []

    if PROJECT_DIR.exists():
        for path in PROJECT_DIR.rglob("*"):
            if not path.is_file():
                continue

            if any(term in path.name.lower() for term in artifact_terms):
                found_files.append(path)

            # Avoid excessively large scans in the UI.
            if len(found_files) >= 100:
                break

    if not found_files:
        st.info(
            "No saved confusion matrix or classification report was found "
            "under the project directory. Export the evaluation artifacts "
            "from your training notebook to display them here."
        )

    for index, path in enumerate(found_files):
        try:
            relative_path = path.relative_to(PROJECT_DIR)

            with st.expander(str(relative_path)):
                suffix = path.suffix.lower()

                if suffix == ".csv":
                    result_df = pd.read_csv(path)
                    show_df(result_df.head(1000), hide_index=True)

                    st.download_button(
                        "Download this CSV",
                        result_df.to_csv(index=False).encode("utf-8"),
                        path.name,
                        "text/csv",
                        key=f"research_csv_{index}",
                    )

                elif suffix == ".json":
                    with open(path, "r", encoding="utf-8") as file:
                        result_json = json.load(file)
                    st.json(result_json)

                elif suffix in (".png", ".jpg", ".jpeg"):
                    show_img(str(path), use_container_width=True)

        except Exception as exc:
            st.caption(f"Could not display {path.name}: {exc}")



# ============================================================
# 6. DATASET & FEATURE ANALYSIS
# ============================================================

def dataset_feature_analysis():
    page_header(
        "Dataset & Feature Analysis",
        "Explore image splits, YOLO annotations, orbital features, "
        "visual features, correlations, and downloadable data."
    )

    # --------------------------------------------------------
    # Locate dataset paths without changing global configuration
    # --------------------------------------------------------
    possible_roots = [
        PROJECT_DIR / "data",
        PROJECT_DIR,
    ]

    dataset_root = None

    for candidate in possible_roots:
        if any(
            (candidate / split / "images").is_dir()
            for split in ("train", "valid", "test")
        ):
            dataset_root = candidate
            break

    if dataset_root is None:
        dataset_root = PROJECT_DIR / "data"

    split_names = ("train", "valid", "test")
    split_labels = {
        "train": "Training",
        "valid": "Validation",
        "test": "Test",
    }

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
    }

    split_rows = []

    for split in split_names:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        if image_dir.is_dir():
            image_files = [
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            image_count = len(image_files)
        else:
            image_count = 0

        label_count = (
            sum(1 for p in label_dir.glob("*.txt") if p.is_file())
            if label_dir.is_dir()
            else 0
        )

        split_rows.append({
            "Split": split_labels[split],
            "Images": image_count,
            "Label files": label_count,
            "Image directory": str(image_dir),
            "Label directory": str(label_dir),
        })

    split_df = pd.DataFrame(split_rows)

    st.subheader("Dataset Split Summary")

    c1, c2, c3, c4 = st.columns(4)

    total_images = int(split_df["Images"].sum())
    total_labels = int(split_df["Label files"].sum())

    c1.metric("Total Images", f"{total_images:,}")
    c2.metric("Label Files", f"{total_labels:,}")
    c3.metric(
        "Image Splits Found",
        int((split_df["Images"] > 0).sum())
    )
    c4.metric(
        "Images Without Label Files",
        max(total_images - total_labels, 0)
    )

    show_df(split_df, hide_index=True)

    if total_images > 0:
        fig = px.bar(
            split_df,
            x="Split",
            y="Images",
            title="Images per Dataset Split",
            color="Split",
            color_discrete_sequence=[
                "#00d9ff", "#9575ff", "#00e5a8"
            ],
        )
        show_plot(style_fig(fig, height=360),
                  use_container_width=True)

    st.download_button(
        "Download split summary",
        split_df.to_csv(index=False).encode("utf-8"),
        "satguard_dataset_splits.csv",
        "text/csv",
        key="dataset_split_download",
    )

    # --------------------------------------------------------
    # Read YOLO class names from data.yaml when available
    # --------------------------------------------------------
    yaml_candidates = [
        dataset_root / "data.yaml",
        dataset_root / "data.yml",
        PROJECT_DIR / "data.yaml",
        PROJECT_DIR / "data.yml",
    ]

    yaml_path = next(
        (path for path in yaml_candidates if path.is_file()),
        None
    )

    class_names = {}

    if yaml_path is not None:
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}

            names = config.get("names", [])

            if isinstance(names, dict):
                class_names = {
                    int(key): str(value)
                    for key, value in names.items()
                }
            elif isinstance(names, list):
                class_names = {
                    index: str(value)
                    for index, value in enumerate(names)
                }

        except Exception as exc:
            st.caption(
                f"Could not read class names from data.yaml: {exc}"
            )

    # --------------------------------------------------------
    # Compute class counts from actual YOLO label files
    # Each non-empty YOLO label line starts with a class ID.
    # --------------------------------------------------------
    class_counts = {}
    invalid_lines = 0
    valid_lines = 0
    empty_label_files = 0

    for split in split_names:
        label_dir = dataset_root / split / "labels"

        if not label_dir.is_dir():
            continue

        for label_path in label_dir.glob("*.txt"):
            try:
                content = label_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).strip()

                if not content:
                    empty_label_files += 1
                    continue

                for line in content.splitlines():
                    parts = line.strip().split()

                    if not parts:
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        if class_id < 0:
                            raise ValueError("Negative class ID")

                        class_counts[class_id] = (
                            class_counts.get(class_id, 0) + 1
                        )
                        valid_lines += 1

                    except (ValueError, IndexError):
                        invalid_lines += 1

            except OSError:
                continue

    st.divider()
    st.subheader("YOLO Annotation Class Distribution")

    if class_counts:
        class_df = pd.DataFrame([
            {
                "Class ID": class_id,
                "Class": class_names.get(
                    class_id, f"class_{class_id}"
                ),
                "Annotations": count,
            }
            for class_id, count in sorted(class_counts.items())
        ])

        fig = px.bar(
            class_df,
            x="Class",
            y="Annotations",
            color="Annotations",
            title="Annotation Instances by Class",
            color_continuous_scale=["#17456b", "#00d9ff"],
        )
        fig.update_layout(xaxis_tickangle=-35)
        show_plot(style_fig(fig, height=440),
                  use_container_width=True)

        show_df(class_df, hide_index=True)

        st.download_button(
            "Download annotation counts",
            class_df.to_csv(index=False).encode("utf-8"),
            "satguard_annotation_class_counts.csv",
            "text/csv",
            key="annotation_counts_download",
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Valid Annotation Lines", f"{valid_lines:,}")
        c2.metric("Invalid Annotation Lines", f"{invalid_lines:,}")
        c3.metric("Empty Label Files", f"{empty_label_files:,}")

    else:
        st.info(
            "No YOLO class annotations were found. Check that the dataset "
            "contains train/labels, valid/labels, and test/labels folders."
        )

    if yaml_path is not None:
        with st.expander("View dataset configuration"):
            st.code(
                yaml_path.read_text(encoding="utf-8")[:8000],
                language="yaml"
            )

    # --------------------------------------------------------
    # Existing feature datasets from the application's loaders
    # --------------------------------------------------------
    feature_tables = [
        ("Visual Aggregated Features", visual_agg_df),
        ("Real Orbital Features", orbital_feat_df),
        ("TCA / Conjunction Records", tca_df),
    ]

    st.divider()
    st.subheader("Visual, Orbital & TCA Feature Analysis")

    for table_index, (table_name, source_df) in enumerate(feature_tables):
        if source_df is None or source_df.empty:
            st.info(
                f"{table_name} was not found or contains no rows. "
                "Check the configured CSV path in the application."
            )
            continue

        df = source_df.copy()
        df.columns = [str(column).strip() for column in df.columns]

        with st.expander(
            f"{table_name} — {len(df):,} rows × {len(df.columns)} columns",
            expanded=(table_index == 0),
        ):
            st.write("Preview")
            show_df(df.head(1000), hide_index=True)

            st.download_button(
                f"Download {table_name}",
                df.to_csv(index=False).encode("utf-8"),
                f"{table_name.lower().replace(' ', '_').replace('/', '_')}.csv",
                "text/csv",
                key=f"feature_download_{table_index}",
            )

            # Convert numeric-looking columns safely for plotting.
            numeric_df = df.select_dtypes(include=[np.number]).copy()

            if numeric_df.empty:
                st.info(
                    "This table has no numeric columns available for "
                    "correlation or distribution plots."
                )
                continue

            # Correlation matrix
            if numeric_df.shape[1] >= 2:
                selected_features = st.multiselect(
                    "Select numeric features for correlation",
                    options=list(numeric_df.columns),
                    default=list(
                        numeric_df.columns[
                            :min(8, len(numeric_df.columns))
                        ]
                    ),
                    key=f"corr_features_{table_index}",
                )

                if len(selected_features) >= 2:
                    corr_df = (
                        numeric_df[selected_features]
                        .replace([np.inf, -np.inf], np.nan)
                        .corr()
                    )

                    fig = px.imshow(
                        corr_df,
                        labels={
                            "x": "Feature",
                            "y": "Feature",
                            "color": "Correlation",
                        },
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        title=f"{table_name}: Pearson Correlation",
                    )
                    show_plot(style_fig(fig, height=500),
                              use_container_width=True)

            # Feature distribution
            selected_distribution = st.selectbox(
                "Select feature to inspect",
                options=list(numeric_df.columns),
                key=f"distribution_feature_{table_index}",
            )

            distribution_values = pd.to_numeric(
                numeric_df[selected_distribution],
                errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if not distribution_values.empty:
                fig = px.histogram(
                    x=distribution_values,
                    nbins=35,
                    title=f"{selected_distribution}: Distribution",
                    labels={"x": selected_distribution, "y": "Frequency"},
                )
                show_plot(style_fig(fig, height=350),
                          use_container_width=True)
            else:
                st.info("No valid numeric values are available to plot.")



# ------------------------------------------------------------
# PAGE 7 — SYSTEM DIAGNOSTICS
# ------------------------------------------------------------


def page_diagnostics():
    page_header("System Diagnostics", "Files, models, data quality and runtime")

    files = [
        ("YOLO weights", YOLO_PATH), ("Fusion model", FUSION_PATH),
        ("Feature columns", FEATURES_PATH), ("TCA results", TCA_PATH),
        ("Orbital features", ORBITAL_FEATURES_PATH), ("Visual aggregate", VISUAL_AGG_PATH),
    ]
    rows = [{
        "Component": n,
        "Status": "AVAILABLE" if p.is_file() else "MISSING",
        "Size (MB)": round(p.stat().st_size / 1048576, 3) if p.is_file() else None,
        "Path": str(p),
    } for n, p in files]
    show_df(pd.DataFrame(rows), hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Python", sys.version.split()[0])
    c2.metric("Platform", platform.system())
    c3.metric("Model features", len(feature_columns))

    st.write({
        "YOLO loaded": yolo_model is not None,
        "Fusion model loaded": fusion_model is not None,
        "Model type": type(fusion_model).__name__ if fusion_model is not None else None,
        "Feature columns": feature_columns,
        "Matches expected 13 features": list(feature_columns) == DEFAULT_FEATURES,
    })

    if hasattr(fusion_model, "feature_names_in_"):
        same = [str(x) for x in fusion_model.feature_names_in_] == list(feature_columns)
        (st.success if same else st.error)(
            "Model feature order matches feature file." if same
            else "Model feature order does NOT match feature file!")

    if not tca_df.empty:
        st.markdown("### TCA data quality")
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(tca_df):,}")
        b.metric("Columns", len(tca_df.columns))
        c.metric("Missing cells", f"{int(tca_df.isna().sum().sum()):,}")
        show_df(pd.DataFrame({
            "column": tca_df.columns.astype(str),
            "dtype": tca_df.dtypes.astype(str).values,
            "non-null": tca_df.notna().sum().values,
        }), hide_index=True)

    if st.button("♻ Clear caches and reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------

{
    "Mission Overview": page_overview,
    "YOLO Detection": page_detection,
    "Orbital Monitoring": page_orbital,
    "Risk Analysis + SHAP": page_risk,
    "Research Results": research_results,
    "Dataset & Feature Analysis":dataset_feature_analysis,
    "System Diagnostics": page_diagnostics,
}[page]()
