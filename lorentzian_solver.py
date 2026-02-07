import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
from scipy.integrate import simpson
import plotly.graph_objects as go
import io
import time

# --- 1. RESEARCH-GRADE UI CONFIGURATION ---
st.set_page_config(
    page_title="Lorentzian Metric Solver", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# FORCE HIGH-CONTRAST NEON HUD THEME WITH FIXED BUTTONS
st.markdown("""
<style>
    .stApp { background-color: #000000; }
    
    /* Headers & Text - Research HUD Cyan */
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 15px; }
    
    /* Science Metrics */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; text-shadow: 0 0 10px rgba(0,255,65,0.4); }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Sidebar Layout */
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    
    /* FIXED: Stealth Export Buttons (Permanently Dark) */
    div.stButton > button { 
        border: 1px solid #00ADB5 !important; 
        color: #00ADB5 !important; 
        background-color: #161B22 !important; /* Deep Slate Gray */
        width: 100%; 
        border-radius: 2px;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.4s ease;
    }
    
    /* Hover State - Subtle Glow instead of White Flash */
    div.stButton > button:hover { 
        background-color: #1f242d !important; 
        color: #00FFF5 !important; 
        border-color: #00FFF5 !important;
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.4);
    }

    /* Target the download buttons specifically */
    div.stDownloadButton > button {
        background-color: #161B22 !important;
        color: #00ADB5 !important;
        border: 1px solid #00ADB5 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. THE PINN PHYSICS CORE ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(r0, r_max, curve, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            return db_dr - (b / r) * curve 
        def bc_throat(x, on_boundary):
            return on_boundary and np.isclose(x[0], r0)
        
        bc = dde.icbc.DirichletBC(geom, lambda x: r0, bc_throat)
        data = dde.data.PDE(geom, pde, bc, num_domain=400, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, r0, r_max, impact_b):
        r_v = np.linspace(r0, r_max, 600).reshape(-1, 1)
        r_t = torch.tensor(r_v, dtype=torch.float32, requires_grad=True)
        b_t = model.net(r_t)
        db_dr = torch.autograd.grad(b_t, r_t, grad_outputs=torch.ones_like(b_t))[0].detach().numpy()
        b = b_t.detach().numpy()
        
        # Energy & Tidal Analysis
        rho = db_dr / (8 * np.pi * r_v**2 + 1e-9)
        tidal = np.abs((1 - (b / r_v)) / (r_v**2 + 1e-9))
        
        # Photon Deflection Heuristic
        lensing = (impact_b / (r_v + 1e-6)) * (b / r_v)
        
        # Manifold Embedding Integration
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-6)) - 1
            z[i] = z[i-1] + (1.0 / np.sqrt(val) if val > 1e-6 else 12.0) * dr
            
        return r_v, b, rho, tidal, z, lensing

# --- 3. DASHBOARD INTERFACE ---
st.title("LORENTZIAN METRIC SOLVER")

# Side Navigation
st.sidebar.markdown("### ⏯️ KERNEL STATE")
pause = st.sidebar.toggle("HALT SIMULATION", value=False)

st.sidebar.markdown("### 🧬 $G_{\mu\\nu}$ TOPOLOGY")
r_throat = st.sidebar.slider("Throat Radius ($r_0$)", 1.0, 10.0, 2.0)
flare = st.sidebar.slider("Curvature Intensity ($\kappa$)", 0.1, 0.9, 0.5)
b_impact = st.sidebar.slider("Impact Parameter ($b$)", 1.0, 25.0, 5.0)

st.sidebar.markdown("### ⚙️ NUMERICAL SOLVER")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)
lr_val = st.sidebar.select_slider("Rate ($\eta$)", options=[1e-2, 1e-3], value=1e-3)

# Execution
model, hist = SpacetimeSolver.solve_manifold(r_throat, 40.0, flare, epochs, lr_val)
r, b, rho, tidal, z, lens = SpacetimeSolver.extract_telemetry(model, r_throat, 40.0, b_impact)

# Top Bar Telemetry
m1, m2, m3, m4 = st.columns(4)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.1e}")
m2.metric("EXOTIC MASS", f"{simpson(rho.flatten(), x=r.flatten()):.3f}")
m3.metric("TIDAL SHEAR", f"{np.max(tidal):.3f} g")

integrity = "NOMINAL" if np.max(tidal) < 0.45 else "DISRUPTED"
i_color = "#00FF41" if integrity == "NOMINAL" else "#FF2E63"
m4.markdown(f"<div style='text-align:center'><span style='color:#888;font-size:12px'>METRIC INTEGRITY</span><br><span style='color:{i_color};font-size:20px;font-weight:bold'>{integrity}</span></div>", unsafe_allow_html=True)

st.markdown("---")

# Main Visualization Block
v_col, d_col = st.columns([2, 1])

with v_col:
    # Plotly 3D Embedding - Color-coded by Tidal Stress
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z = np.tile(z.flatten(), (60, 1))
    
    # 
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, surfacecolor=np.tile(tidal.flatten(), (60, 1)), colorscale='Viridis', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, surfacecolor=np.tile(tidal.flatten(), (60, 1)), colorscale='Viridis', showscale=False)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='cube'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

    # Export Buttons
    e1, e2 = st.columns(2)
    # Generate temporary buffer for image snapshot
    img_buf = io.BytesIO()
    # Using a dummy plot to hold the snapshot space
    e1.download_button("📸 SNAPSHOT MANIFOLD", data=img_buf.getvalue(), file_name="topology_snapshot.png", use_container_width=True)
    
    df_out = pd.DataFrame({"Radius": r.flatten(), "Shape": b.flatten(), "Tidal": tidal.flatten()})
    e2.download_button("📊 EXPORT TELEMETRY (CSV)", data=df_out.to_csv(index=False).encode('utf-8'), file_name="telemetry.csv", use_container_width=True)

with d_col:
    # Secondary Analytics Tabs
    tabs = st.tabs(["🔭 OPTICAL GEODESICS", "📈 TENSOR PROFILES"])
    
    with tabs[0]:
        st.subheader("Null-Geodesic Deflection")
        fig_l, ax_l = plt.subplots(facecolor='black')
        ax_l.set_facecolor('black')
        ax_l.plot(r, lens, color='#00FFF5', lw=2.5)
        ax_l.set_title("Deflection Angle ($\\alpha$)", color='white')
        ax_l.set_xlabel("Impact Radius $r$", color='#888')
        ax_l.grid(color='#222')
        ax_l.tick_params(colors='#888')
        st.pyplot(fig_l)
        st.caption("Visualizing the bending of light paths (Null-Geodesics) around the manifold throat.")
        # 

    with tabs[1]:
        st.subheader("Field Distributions")
        fig_t, ax_t = plt.subplots(2, 1, facecolor='black', figsize=(6, 9))
        ax_t[0].plot(r, b, color='#00ADB5', lw=2); ax_t[0].set_title("Shape Function $b(r)$", color='white')
        ax_t[1].plot(r, rho, color='#FF2E63', lw=2); ax_t[1].set_title("Energy Density $\\rho$", color='white')
        for a in ax_t: 
            a.set_facecolor('black')
            a.tick_params(colors='#888')
            a.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_t)

# Lifecycle Management
if not pause:
    time.sleep(0.01)
    st.rerun()
