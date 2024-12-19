# Neural ODEs for Physical Systems

This repository contains the code and resources for an MSc dissertation exploring the application of Neural Ordinary Differential Equations (NODEs) to model complex physical systems. The study focuses on three distinct physical systems: **the double pendulum, the Lorenz attractor, and the magnetic domain wall problem.** The objective is to investigate and compare the predictive performance of Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Neural ODEs.

---

I'd be glad to create the improved Overview section for your dissertation repository's README.md file:

Markdown

## Overview

This repository provides the resources associated with your dissertation titled "[Dissertation Title]". Here's a breakdown of what you'll find:

**Dissertation:**

* **Dissertation.pdf**: A comprehensive document outlining the theoretical framework, implementation details, results obtained, and in-depth analysis of the research conducted.

**Simulation Code:**

* This folder contains code used to simulate the dynamics of various physical systems:
    * `DW_oscillator.py`: This script simulates the motion of a magnetic domain wall, including the governing equations.
    * `sequence.py`: This script generates time-dependent fields used in the domain wall simulations.

**Model Implementations:**

* This folder houses the code for training and evaluating different machine learning models on the simulated physical systems. These models include:
    * Recurrent Neural Networks (RNN)
    * Long Short-Term Memory (LSTM) networks
    * Neural Ordinary Differential Equations (NODE) models

**Datasets:**

* This folder contains pre-generated datasets used for training, validating, and testing the machine learning models on the simulated physical systems.

---

## Key Concepts

- **Neural Ordinary Differential Equations (NODEs)**: A neural network architecture that parameterizes the derivative of a system's state, enabling continuous-time modeling with ODE solvers.
  
  <img src="PNG/readme/resnodevfields.png" alt="Neural ODE Architecture" width="250" height="180">

- **Chaotic Systems**: Dynamical systems with sensitive dependence on initial conditions, such as the double pendulum and the Lorenz attractor.
- **Recurrent Neural Networks (RNNs)**: Sequential models using hidden states to maintain memory over time.
- **Long Short-Term Memory Networks (LSTMs)**: Enhanced RNNs that use gating mechanisms to overcome vanishing gradient issues.
- **Adjoint Method**: An efficient backpropagation method for Neural ODEs that saves memory and ensures numerical stability.
- **ODE Solvers**: Numerical methods like Euler’s method, RK4, and Dormand-Prince (DOPRI5) to approximate ODE solutions.

---

## Experiments

### Double Pendulum and Lorenz Attractor

<div style="display: flex; justify-content: center; gap: 20px;">
  <div>
    <img src="PNG/readme/pendulum.gif" alt="Double Pendulum" width="400" height="300">
    <p align="center"><b>Double Pendulum:</b> A classic example of a chaotic system. Models predict the angles and angular velocities of the pendulum arms.</p>
  </div>
  <div>
    <img src="PNG/readme/lorenz.png" alt="Lorenz Attractor" width="400" height="300">
    <p align="center"><b>Lorenz Attractor:</b> A well-known chaotic system in atmospheric science. Models predict the evolution of state variables over time.</p>
  </div>
</div>

### Magnetic Domain Wall Dynamics

<div style="display: flex; justify-content: center; gap: 20px;">
  <div>
    <img src="PNG/readme/dw1.gif" alt="Domain Wall 1" width="400" height="300">
    <p align="center"><b>Domain Wall 1:</b> Dynamics of a domain wall placed between two anti-notches of nickel nanowire under oscillating magnetic fields.</p>
  </div>
  <div>
    <img src="PNG/readme/dw2.gif" alt="Domain Wall 2" width="400" height="300">
    <p align="center"><b>Domain Wall 2:</b> Continuation of dynamics under varying field intensities.</p>
  </div>
</div>

---

## Results

The models were evaluated using **Mean Squared Error (MSE)** and qualitative visualizations of predicted vs. actual dynamics.

### Lorenz Attractor Vector Fields
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="PNG/readme/lorvflstm.png" alt="LSTM VF Lorenz" width="300" height="200">
  <img src="PNG/readme/lorvfrnn.png" alt="RNN VF Lorenz" width="300" height="200">
  <img src="PNG/readme/lorvfnode.png" alt="NODE VF Lorenz" width="300" height="200">
</div>

### Double Pendulum Vector Fields
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="PNG/readme/dpvfrnn.png" alt="DP VF RNN" width="300" height="200">
  <img src="PNG/readme/dpvflstm.png" alt="DP VF LSTM" width="300" height="200">
  <img src="PNG/readme/dpvfnode.png" alt="DP VF NODE" width="300" height="200">
</div>

### Domain Wall Dynamics (Single Field vs. Multiple Fields)
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="PNG/readme/dwvfs.png" alt="Node Single Field" width="300" height="200">
  <img src="PNG/readme/dwmft.png" alt="Node Multiple Field Predictions" width="300" height="200">
</div>
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="PNG/readme/dw_single.png" alt="Single Field Domain Wall" width="300" height="200">
  <img src="PNG/readme/dw_multiple.png" alt="Multiple Field Domain Wall" width="300" height="200">
</div>

**Key Observations:**
- Neural ODEs outperform RNNs and LSTMs in modeling long-term dynamics, preserving system behavior better.
- NODEs struggle with:
  - Systems with abrupt changes (e.g., magnetic domain wall dynamics).
  - Stiff ODEs and chaotic systems over extended timeframes.

---

## Future Work

### Proposed Directions:
1. **Augmented Neural ODEs (ANODEs):** Improve NODEs by increasing the dimensions of the state space.
2. **Second-Order Neural ODEs (SONODEs):** Better suited for systems with inherent second-order dynamics.
3. **Hamiltonian Neural Networks (HNNs):** Enforce conservation laws for physical interpretability.
4. **Lagrangian Neural Networks:** Use Lagrangian mechanics for energy conservation in modeling chaotic systems.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/JagpreetJakhar/Dissertation]
