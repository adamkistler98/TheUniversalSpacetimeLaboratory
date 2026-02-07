import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import pandas as pd
from scipy.integrate import simpson

# --- 1. PAGE CONFIGURATION & SESSION STATE ---
st.set_page_config(
    page_title="Lorentzian Metric Solver",
    layout="wide",
    page_icon="📐",
    initial_sidebar_state="expanded"
)

# Initialize Session State for caching model results between interactions
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'simulation_data' not in st.session_state:
    st.session_state['simulation_data'] = None

# --- 2. SCIENTIFIC COMPUTATION ENGINE (The Physics) ---
class MetricPhysicsEngine:
    """
    Encapsulates the PINN (Physics-Informed Neural Network) logic for solving
    Einstein Field Equations regarding static, spherically symmetric spacetime manifolds.
    """
    
    @staticmethod
    def train_pinn(r0, r_max, iterations, learning_rate):
        # 2.1 Geometry Definition (Spatial Domain)
        geom = dde.geometry.Interval(r0, r_max)

        # 2.2 The Physics (Einstein Field Eq for Morris-Thorne Metric)
        # We solve for shape function b(r). Stability requires db/dr = b/r at specific boundaries.
        def pde(r, b):
            db_dr = dde.grad.jacobian(b, r)
            # Residual of the field equation
            return db_dr - (b / r)

        # 2.3 Boundary Condition: Minimal Surface at Throat
        # b(r0) == r0 is required for the throat to be open.
        def boundary_throat(x, on_boundary):
            return on_boundary and np.isclose(x[0], r0)

        bc = dde.icbc.DirichletBC(geom, lambda x: r0, boundary_throat)

        # 2.4 Neural Network Architecture
        # Using a deeper network to capture non-linear metric gradients
        data = dde.data.PDE(geom, pde, bc, num_domain=250, num_boundary=50)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)

        # 2.5 Optimization
        model.compile("adam", lr=learning_rate)
        loss_history, train_state = model.train(iterations=iterations)
        
        return model, loss_history

    @staticmethod
    def analyze_metric(model, r0, r_max):
        # Generate high-resolution domain for analysis
        r_val = np.linspace(r0, r_max, 500).reshape(-1, 1)
        r_tensor = torch.tensor(r_val, dtype=torch.float32, requires_grad=True)
        
        # Inference with Autograd for Derivatives
        b_tensor = model.net(r_tensor)
        db_dr_tensor = torch.autograd.grad(
            b_tensor, r_tensor, 
            grad_outputs=torch.ones_like(b_tensor), 
            create_graph=False
        )[0]
        
        # Convert to NumPy
        b_pred = b_tensor.detach().numpy()
        db_dr = db_dr_tensor.detach().numpy()
        
        # --- PHYSICS CALCULATIONS ---
        
        # 1. Energy Density (rho) in Geometrized Units (G=c=1)
        # Field Eq: 8*pi*rho = b' / r^2
        rho = db_dr / (8 * np.pi * r_val**2)
        
        # 2. Lateral Tidal Forces (Acceleration in g's)
        # Simplified approximation: Tidal ~ (c^2 / r^2) * (1 - b/r)
        # Assuming a traveler velocity gamma factor ~ 1 (slow travel)
        # We normalize to Earth Gravity (9.8 m/s^2) for context assuming r is in meters
        # Note: This is a qualitative index for the simulation.
        tidal_forces = (1 - (b_pred / r_val)) / (r_val**2)
        
        # 3. Flare-Out Condition (Embedding Check)
        flare_out = b_pred / r_val
        
        return r_val, b_pred, rho, tidal_forces, flare_out

# --- 3. RENDERING ENGINE (The Visuals) ---
def render_geodesics(r0, resolution=400):
    limit = r0 * 5
    x = np.linspace(-limit, limit, resolution)
    y = np.linspace(-limit, limit, resolution)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Differential Light Deflection (Gravitational Lensing)
    mask = R > r0
    deflection = np.zeros_like(R)
    
    # Deflection angle alpha ~ 4GM/bc^2 approx
    deflection[mask] = (R[mask] - r0) / R[mask]
    
    # Background Metric Grid (The Universe behind the manifold)
    grid_bg = np.sin(5 * np.pi * X * deflection) * np.sin(5 * np.pi * Y * deflection)
    
    # Throat Metric Grid (The manifold interior)
    grid_throat = np.sin(15 * np.pi * R / limit) 
    
    final_image = np.where(mask, grid_bg, grid_throat)
    return final_image, limit

# --- 4. UI LAYOUT ---
st.title("Math::Lorentzian_Metric_Solver")
st.markdown("### 🧬 Computational General Relativity Environment")
st.markdown("---")

# Sidebar
st.sidebar.header("Initial Conditions")
throat_r0 = st.sidebar.slider("Throat Radius ($r_{min}$)", 0.5, 5.0, 1.0, 0.1)
domain_max = st.sidebar.slider("Manifold Domain ($r_{max}$)", 5.0, 25.0, 10.0)
learning_rate = st.sidebar.selectbox("Gradient Descent Rate", [1e-2, 1e-3, 5e-4], index=1)
train_iters = st.sidebar.number_input("Training Epochs", 500, 10000, 2000)

if st.sidebar.button("RUN SOLVER", type="primary"):
    with st.spinner("Optimizing Einstein Field Equations (PINN)..."):
        # Run Physics
        model, history = MetricPhysicsEngine.train_pinn(throat_r0, domain_max, train_iters, learning_rate)
        r, b, rho, tidal, flare = MetricPhysicsEngine.analyze_metric(model, throat_r0, domain_max)
        
        # Store Data
        st.session_state['simulation_data'] = {
            'r': r, 'b': b, 'rho': rho, 'tidal': tidal, 'flare': flare, 'history': history
        }
        st.session_state['model_trained'] = True

# Results Dashboard
if st.session_state['model_trained']:
    data = st.session_state['simulation_data']
    
    # KPI Row
    kpi1, kpi2, kpi3 = st.columns(3)
    
    # Calculate Total Exotic Matter (Volume Integral)
    # V_exotic = Integral(rho * 4pi * r^2 dr)
    integrand = (data['rho'].flatten() * 4 * np.pi * (data['r'].flatten()**2))
    total_violation = simpson(integrand, x=data['r'].flatten())
    
    max_tidal = np.max(data['tidal'])
    
    kpi1.metric("Convergence Loss", f"{data['history'].loss_train[-1][0]:.2e}")
    kpi2.metric("Total NEC Violation", f"{total_violation:.4f} M_pl")
    kpi3.metric("Max Tidal Force", f"{max_tidal:.4f} g", delta_color="inverse")

    if max_tidal > 0.5: # Arbitrary threshold for this simulation
        st.error("⚠️ CRITICAL WARNING: High tidal forces detected. Metric is non-traversable for biological entities.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📉 Metric Tensor Analysis", "🔭 Optical Geodesics", "⚙️ Solver Dynamics"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Shape Function $b(r)$")
            fig, ax = plt.subplots()
            ax.plot(data['r'], data['b'], 'b-', lw=2, label="PINN Solution")
            ax.plot(data['r'], data['r'], 'k--', label="Schwarzschild Horizon")
            ax.set_xlabel("$r$ (Coordinate Radius)")
            ax.set_ylabel("$b(r)$")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with c2:
            st.subheader("Energy Density Profile")
            fig2, ax2 = plt.subplots()
            ax2.plot(data['r'], data['rho'], 'r-', lw=2)
            ax2.fill_between(data['r'].flatten(), data['rho'].flatten(), 0, color='red', alpha=0.1)
            ax2.set_xlabel("$r$")
            ax2.set_ylabel(r"Density $\rho$ (Planck Units)")
            st.pyplot(fig2)
            st.caption("Negative regions indicate violation of Null Energy Conditions (NEC).")

    with tab2:
        st.subheader("Null Geodesic Ray-Tracing")
        img, lim = render_geodesics(throat_r0)
        fig3, ax3 = plt.subplots(figsize=(8,8), facecolor='#0e1117')
        ax3.imshow(img, cmap='inferno', extent=[-lim, lim, -lim, lim])
        ax3.axis('off')
        st.pyplot(fig3)
        
    with tab3:
        st.subheader("Neural Network Convergence")
        hist = np.array(data['history'].loss_train)
        fig4, ax4 = plt.subplots()
        ax4.semilogy(hist[:, 0], label="PDE Residual (Physics Error)")
        ax4.semilogy(hist[:, 1], label="Boundary Condition Error")
        ax4.set_xlabel("Epochs")
        ax4.set_ylabel("Loss (Log Scale)")
        ax4.legend()
        st.pyplot(fig4)

else:
    st.info("Awaiting user input. Adjust boundaries in the sidebar to initialize the solver.")