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

# --- 1. UI CONFIGURATION & FULL STEALTH STYLING ---
st.set_page_config(
    page_title="Lorentzian Metric Solver", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# TOTAL VOID THEME: No white backgrounds, no bright boxes.
st.markdown(r"""
<style>
    /* Main Background - True Black */
    .stApp { background-color: #000000; }
    
    /* Headers & Text - Research HUD Cyan */
    h1, h2, h3, h4 { color: #00ADB5 !important; font-family: 'Consolas', monospace; }
    p, li, label, .stMarkdown, .stCaption { color: #FFFFFF !important; font-size: 15px; }
    
    /* Stealth Input Boxes - Dark Gray with Cyan Glow */
    input { background-color: #161B22 !important; color: #00FFF5 !important; border: 1px solid #333 !important; }
    div[data-baseweb="input"] { background-color: #161B22 !important; border: 1px solid #00ADB5 !important; }
    div[role="slider"] { background-color: #00ADB5 !important; }
    
    /* Scientific Metrics */
    div[data-testid="stMetricValue"] { color: #00FF41 !important; font-family: 'Consolas', monospace; text-shadow: 0 0 10px rgba(0,255,65,0.4); }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    
    /* Stealth Export & Download Buttons */
    div.stButton > button, div.stDownloadButton > button { 
        border: 1px solid #00ADB5 !important; 
        color: #00ADB5 !important; 
        background-color: #161B22 !important; 
        width: 100%; 
        border-radius: 2px;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.4s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover { 
        background-color: #1f242d !important; 
        color: #00FFF5 !important; 
        border-color: #00FFF5 !important;
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.4);
    }

    /* Fixing potential white backgrounds in tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #000; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00ADB5; border-bottom-color: #00ADB5; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE PINN PHYSICS CORE ---
class SpacetimeSolver:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def solve_manifold(r0, r_max, curve, redshift, iters, lr):
        geom = dde.geometry.Interval(r0, r_max)
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            # General Relativistic constraint for static metric
            return db_dr - (b / r) * curve + (redshift * (1 - b/r))
        
        bc = dde.icbc.DirichletBC(geom, lambda x: r0, lambda x, on: on and np.isclose(x[0], r0))
        data = dde.data.PDE(geom, pde, bc, num_domain=500, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        loss, _ = model.train(iterations=iters, display_every=1000)
        return model, loss

    @staticmethod
    def extract_telemetry(model, r0, r_max, impact_b, redshift_val):
        r_v = np.linspace(r0, r_max, 700).reshape(-1, 1)
        r_t = torch.tensor(r_v, dtype=torch.float32, requires_grad=True)
        b_t = model.net(r_t)
        db_dr = torch.autograd.grad(b_t, r_t, grad_outputs=torch.ones_like(b_t))[0].detach().numpy()
        b = b_t.detach().numpy()
        
        # Physics Analysis
        rho = db_dr / (8 * np.pi * r_v**2 + 1e-12)
        tau = (b / (8 * np.pi * r_v**3)) - (2 * redshift_val * (1 - b/r_v) / (8 * np.pi * r_v))
        xi = rho - tau # Exoticity Index
        lensing = (impact_b / (r_v + 1e-6)) * (b / r_v)
        
        # 3D Embedding Geometry
        z = np.zeros_like(r_v)
        dr = r_v[1] - r_v[0]
        for i in range(1, len(r_v)):
            val = (r_v[i] / (b[i] + 1e-9)) - 1
            z[i] = z[i-1] + (1.0 / np.sqrt(val) if val > 1e-9 else 12.0) * dr
            
        return r_v, b, rho, tau, xi, z, lensing

# --- 3. DASHBOARD INTERFACE ---
st.title("LORENTZIAN METRIC SOLVER")

# SIDEBAR: High-Precision Terminal
st.sidebar.markdown(r"### 🧬 $G_{\mu\nu}$ TOPOLOGY")
r_throat = st.sidebar.number_input(r"Throat Radius ($r_0$)", 0.0001, 500.0, 2.0, format="%.4f")
flare = st.sidebar.slider(r"Curvature ($\kappa$)", 0.01, 0.99, 0.5)
redshift = st.sidebar.slider(r"Redshift ($\Phi$)", 0.0, 1.0, 0.0)

st.sidebar.markdown(r"### ⚙️ NUMERICAL KERNEL")
lr_val = st.sidebar.number_input(r"Rate ($\eta$)", 0.000001, 0.1, 0.001, format="%.6f")
epochs = st.sidebar.select_slider("Epochs", options=[1000, 2500, 5000], value=2500)

pause = st.sidebar.toggle("HALT SIMULATION", value=False)

# Solver Execution
model, hist = SpacetimeSolver.solve_manifold(r_throat, r_throat * 12, flare, redshift, epochs, lr_val)
r, b, rho, tau, xi, z, lens = SpacetimeSolver.extract_telemetry(model, r_throat, r_throat * 12, 5.0, redshift)

# Top Telemetry
m1, m2, m3, m4 = st.columns(4)
m1.metric("CONVERGENCE", f"{hist.loss_train[-1][0]:.1e}")
m2.metric(r"EXOTICITY ($\xi$)", f"{np.min(xi):.4f}")
m3.metric("TIDAL SHEAR", f"{np.max(np.abs(1-b/r)):.3f}")
integrity = "STABLE" if np.min(xi) < 0 else "NULL"
m4.markdown(f"<div style='text-align:center'><span style='color:#888;font-size:11px'>METRIC INTEGRITY</span><br><span style='color:#00FF41;font-size:18px;font-weight:bold'>{integrity}</span></div>", unsafe_allow_html=True)

st.markdown("---")

v_col, d_col = st.columns([2, 1])

with v_col:
    # 3D Mirror Universe Graph
    th = np.linspace(0, 2*np.pi, 60)
    R, T = np.meshgrid(r.flatten(), th)
    Z = np.tile(z.flatten(), (60, 1))
    
    fig = go.Figure(data=[
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=Z, colorscale='Viridis', showscale=False),
        go.Surface(x=R*np.cos(T), y=R*np.sin(T), z=-Z, colorscale='Viridis', showscale=False)
    ])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='cube'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Export Buttons Underneath
    e1, e2 = st.columns(2)
    e1.download_button("📸 SNAPSHOT MANIFOLD", data=io.BytesIO().getvalue(), file_name="topology.png", use_container_width=True)
    df_out = pd.DataFrame({"r": r.flatten(), "b": b.flatten(), "rho": rho.flatten()})
    e2.download_button("📊 EXPORT TELEMETRY (CSV)", data=df_out.to_csv(index=False).encode('utf-8'), file_name="telemetry.csv", use_container_width=True)

with d_col:
    # Data Tabs
    tabs = st.tabs(["📉 EXOTICITY", "📈 FIELD TENSORS"])
    
    with tabs[0]:
        st.subheader("Energy Condition Violation")
        
        fig_e, ax_e = plt.subplots(facecolor='black')
        ax_e.set_facecolor('black')
        ax_e.plot(r, xi, color='#FF2E63', lw=2)
        ax_e.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax_e.fill_between(r.flatten(), xi.flatten(), 0, where=(xi.flatten() < 0), color='#FF2E63', alpha=0.15)
        ax_e.set_title(r"NEC Deviation ($\rho - \tau$)", color='white')
        ax_e.tick_params(colors='#888')
        st.pyplot(fig_e)

    with tabs[1]:
        st.subheader("Geometric Flux")
        fig_t, ax_t = plt.subplots(2, 1, facecolor='black', figsize=(6, 9))
        ax_t[0].plot(r, b, color='#00ADB5', lw=2); ax_t[0].set_title(r"Shape Function $b(r)$", color='white')
        ax_t[1].plot(r, rho, color='#00FF41', lw=2); ax_t[1].set_title(r"Energy Density $\rho$", color='white')
        for a in ax_t: 
            a.set_facecolor('black'); a.tick_params(colors='#888'); a.grid(alpha=0.1)
        plt.tight_layout()
        st.pyplot(fig_t)

# Lifecycle Loop
if not pause:
    time.sleep(0.01)
    st.rerun()
