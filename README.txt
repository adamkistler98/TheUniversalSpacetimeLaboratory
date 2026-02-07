# 🌌 The Universal Spacetime Laboratory
### A Computational General Relativity Engine & Metric Solver

## 🚀 Overview
**The Universal Spacetime Laboratory** is a high-fidelity physics engine designed to solve, visualize, and analyze Lorentzian manifolds in real-time. Unlike traditional ray-tracers that rely on pre-computed geometry, this engine utilizes **Physics-Informed Neural Networks (PINNs)** via DeepXDE to solve the **Einstein Field Equations (EFE)** on the fly.

This tool allows researchers to interact with **14 distinct spacetime metrics**, manipulating variables such as Mass ($M$), Spin ($a$), Charge ($Q$), and Cosmological Constants ($\Lambda$) to observe instantaneous changes in spatial curvature, gravitational potential, and event horizon topology.

## 🔬 Core Capabilities

### 1. The 14-Metric Research Library
A complete "Standard Model" of General Relativity solutions, fully interactive and parameter-driven:
* **Singularity Dynamics:** Schwarzschild, Kerr (Rotating), Reissner-Nordström (Charged), Kerr-Newman (Charged & Rotating).
* **Topological Engineering:** Morris-Thorne Wormholes, Einstein-Rosen Bridges, Ellis Drainholes.
* **Cosmological Evolution:** Schwarzschild-de Sitter (Expansion), Anti-de Sitter (Contraction), Gott Cosmic Strings.
* **Exotic Frontiers:** Alcubierre Warp Drive, Vaidya Radiating Stars, GHS Stringy Black Holes, Bonnor-Melvin Magnetic Universes.

### 2. Dual Full-Manifold Visualization (Quad-Quadrant HUD)
The engine renders two simultaneous, bi-directional 3D manifolds to provide a complete topological picture:
* **Manifold Zenith (Graph A):** Visualizes the **Geometric Embedding ($ds^2$)**—the physical "stretch" and curvature of the spatial fabric.
* **Manifold Nadir (Graph B):** Visualizes the **Gravitational Potential ($g_{tt}$)**—the time-dilation depth and "gravity well" profile.

### 3. Analytical Tri-Tab Suite
Real-time tensor analysis performed alongside the 3D rendering:
* **Stress-Energy Tensor ($T_{\mu\nu}$):** Maps the energy density distribution required to sustain the metric.
* **Shape Function ($b(r)$):** Tracks the throat radius and horizon location.
* **Particle Kinematics:** Simulates a test particle's geodesic path, calculating the **Lorentz Factor ($\gamma$)** and infall velocity.

## 🛠️ Technical Architecture

This project abandons standard numerical integration for a Deep Learning approach using **DeepXDE**.

* **Solver:** A fully connected Neural Network (FNN) with `tanh` activation.
* **Loss Function:** The network minimizes the **EFE Residual**, effectively "learning" the shape of spacetime that satisfies General Relativity for the user's chosen parameters.
* **Frontend:** Built with Streamlit for rapid parameter tuning and Plotly/Matplotlib for scientific plotting.
* **Compute:** Optimized for CPU/GPU inference with PyTorch backend.

**Key Libraries:** `Streamlit`, `DeepXDE`, `Torch`, `NumPy`, `Plotly`, `Matplotlib`, `Pandas`.

## 📦 Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/Universal-Spacetime-Lab.git](https://github.com/YourUsername/Universal-Spacetime-Lab.git)
    cd Universal-Spacetime-Lab
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Laboratory:**
    ```bash
    streamlit run lorentzian_solver.py
    ```

4.  **Access the HUD:**
    Open your browser to `http://localhost:8501`.

## 🎛️ Parameter Guide

| Variable | Symbol | Description | Relevant Metrics |

| **Mass** | $M$ | The gravitational mass of the singularity. | Kerr, Schwarzschild, RN |
| **Spin** | $a$ | Angular momentum per unit mass ($J/M$). | Kerr, Kerr-Newman |
| **Charge** | $Q, P$ | Electric and Magnetic charge. | RN, Kerr-Newman |
| **Lambda** | $\Lambda$ | The Cosmological Constant (Dark Energy). | de Sitter, AdS |
| **Dilaton** | $\phi$ | Scalar field coupling in String Theory. | GHS Stringy, Naked Singularity |
| **Velocity** | $v/c$ | Apparent velocity of the warp bubble. | Alcubierre Drive |

## 👨‍💻 Author

**[Adam Kistler]**


---
*"Matter tells Space how to curve; Space tells Matter how to move." — John Archibald Wheeler*
