# Lorentzian Manifold Metric Solver (NMSS)
### A Scientific Machine Learning (SciML) Approach to General Relativity

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Framework: DeepXDE](https://img.shields.io/badge/Framework-DeepXDE-orange.svg)](https://github.com/lululxvi/deepxde)
[![Backend: PyTorch](https://img.shields.io/badge/Backend-PyTorch-red.svg)](https://pytorch.org/)

## 🔭 Overview
This repository contains a **Physics-Informed Neural Network (PINN)** designed to solve the **Einstein Field Equations** for a static, spherically symmetric spacetime metric (Morris-Thorne geometry). Unlike traditional numerical methods, this solver utilizes deep learning to approximate the spacetime metric's shape function while strictly adhering to General Relativity constraints.

## 🧠 Technical Architecture
The core of the solver is a Deep Neural Network constrained by a custom loss function derived from the **Lorentzian Manifold** equations.

* **Physics-Informed Loss:** The model minimizes the residual of the Partial Differential Equation (PDE):  
    $$\frac{db}{dr} - \frac{b}{r} = 0$$
* **Automatic Differentiation:** Uses PyTorch’s `autograd` engine to extract the **Stress-Energy Tensor** components directly from the neural network weights.
* **Metric Analysis:** Quantifies **Null Energy Condition (NEC)** violations and calculates **Gravitational Tidal Forces** to assess the theoretical traversability of the manifold.

## 📊 Key Features
* **Real-time PDE Solver:** Retrains the PINN backend dynamically based on user-defined boundary conditions ($r_min$, $r_max$).
* **Geodesic Ray-Tracing:** Visualizes gravitational lensing and light-path deflection through the solved metric.
* **Engineering Metrics:** Calculates total "Exotic Matter" volume integration and biometric safety thresholds.


## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/lorentzian-metric-solver.git](https://github.com/your-username/lorentzian-metric-solver.git)
   cd lorentzian-metric-solver