
# Neural ODEs for Physical Systems

This repository contains the code and resources for the MSc dissertation on applying Neural Ordinary Differential Equations (NODEs) to model complex physical systems. The study focuses on three distinct physical systems: **the double pendulum, the Lorenz attractor, and the domain wall problem.** The goal is to investigate how well Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Neural ODEs perform in modeling these systems, comparing their predictive capabilities.

## Overview

This repository includes the following:

*   **Dissertation.pdf:** A PDF of the full dissertation document, detailing the theoretical background, implementation, results, and analysis of the research.
*   **Code for Simulation:** The code for simulating the double pendulum, Lorenz attractor, and magnetic domain wall dynamics as well as the implementation of the RNN, LSTM, and NODE models.
    *   `DW_oscillator.py`:  Contains the class to simulate the magnetic domain wall dynamics including the equations of motion.
    *   `sequence.py`: Contains a class to generate time dependent fields for the domain wall simulations.
*   **Model Implementations:** Code for the implementation of Recurrent Neural Networks(RNNs), Long Short-Term Memory Networks (LSTMs), and Neural Ordinary Differential Equations (NODEs).
    *   Includes the training and evaluation scripts for each model on each of the systems being modeled.
*  **Data:** The datasets used for training, validation and testing.

## Key Concepts

*   **Neural Ordinary Differential Equations (NODEs):** A class of neural network models that parameterize the derivative of a system's state, allowing for continuous-time modeling using ODE solvers.

<img src="PNG/readme/resnodevfields.png" alt="Neural ODE Arch" width="200" height="150">


*   **Chaotic Systems:** Dynamical systems that exhibit sensitivity to initial conditions, making long-term predictions difficult. Examples include the double pendulum and the Lorenz attractor.
*   **Recurrent Neural Networks (RNNs):** Neural networks designed for sequential data that use hidden states to maintain memory of previous inputs.
*   **Long Short-Term Memory Networks (LSTMs):** An enhanced type of RNN that uses memory cells and gate units to address the vanishing and exploding gradient problem in RNNs.
*   **Adjoint Method**:  A method used for backpropagation in Neural ODEs that saves memory and is numerically stable.
*  **ODE Solvers**: Numerical methods used to approximate the solutions of ODEs, such as Euler’s method, Runge-Kutta 4th order method(RK4), and Dormand-Prince method(DOPRI5).

##  Experiments

The dissertation explores the performance of RNNs, LSTMs, and NODEs on three different physical systems:

1.  **Double Pendulum:** A classic example of a chaotic dynamical system. The models are used to predict the angles and angular velocities of the two pendulum arms.

<img src="PNG/readme/pendulum.gif" alt="Double Pendulum" width="400" height="300">

2.  **Lorenz Attractor:** A well-known chaotic system in atmospheric science. The models are used to predict the evolution of the attractor's state variables over time.

<img src="PNG/readme/lorenz.png" alt="Lorenz Attractor" width="400" height="300">

3.  **Magnetic Domain Wall:** Dynamics of a domain wall placed between two anti-notches of nickel nanowire which is fed an oscillating magnetic field. The models are used to predict the domain wall's position and angle (or magnetization angle).
<div style="display: flex; gap: 10px;">
<img src="PNG/readme/dw1.gif" alt="Domain Wall 1" width="400" height="300">
<img src="PNG/readme/dw2.gif" alt="Domain Wall 2" width="400" height="300">
</div>

## Model Evaluation

The models were evaluated based on the following metrics:

*   **Mean Squared Error (MSE):** Used as the primary metric for measuring the difference between predicted and true values.
*   **Visualizations:**  Plots of the predicted and true values, including the paths of the pendulum masses, the Lorenz attractor, and domain wall dynamics to qualitatively assess the model's forecasting ability.
*   **Vector Fields:** Visualizations of the dynamics of model predictions to compare with the actual dynamics of the systems.

##  Results

The results indicate that:

*   Neural ODEs outperform traditional RNNs and LSTMs in modeling the long-term dynamics of chaotic systems, especially in retaining the overall structure of the systems' behavior.
<div style="display: flex; gap: 10px;">
<img src="PNG/readme/lorvflstm.png" alt="LSTM VF lorenz" width="400" height="300">
<img src="PNG/readme/lorvfrnn.png" alt="RNN VF Lorenz" width="400" height="300">
<img src="PNG/readme/lorvfnode.png" alt="NODE VF Lorenz" width="400" height="300">
</div>
<div style="display: flex; gap: 10px;">
<img src="PNG/readme/dpvfrnn.png" alt="DP VF RNN" width="400" height="300">
<img src="PNG/readme/dpvflstm.png" alt="DP VF LSTM" width="400" height="300">
<img src="PNG/readme/dpvfnode.png" alt="DP VF NODE" width="400" height="300">
</div>
<div style="display: flex; gap: 10px;">
<img src="PNG/readme/dwvfs.png" alt="Node single Field" width="400" height="300">
<img src="PNG/readme/dw_single.png" alt="Single Field Domain Wall" width="400" height="300">
</div>
<div style="display: flex; gap: 10px;">
<img src="PNG/readme/dwvfm.png" alt="Node Multiple Field" width="400" height="300">
<img src="PNG/readme/dw_multiple.png" alt="Multiple Field Domain Wall" width="400" height="300">
</div>
<div style="display: flex; gap: 10px;">
<img src="PNG/readme/dwsftm.png" alt="Node Single Field Predictions" width="400" height="300">
<img src="PNG/readme/dwmft.png" alt="Node Multiple Field Predictions" width="400" height="300">
</div>

*   However, NODE models have some limitations; for instance, their approximation of chaotic systems falls out of phase with the true data.
*  The Neural ODE architecture has inherent limitations, including its inability to solve Stiff ODEs and struggles when systems display dynamic and chaotic behavior.
*  The model's ability to learn the dynamics of the Domain Wall problem was limited, indicating a difficulty with handling abrupt changes in the system.

## Future Work

The dissertation concludes by discussing future research directions, including:

*   **Augmented Neural ODEs (ANODEs):**  To improve the function approximation capability of Neural ODEs by increasing the dimensions of the space on which the ODE is solved.
*   **Second-Order Neural ODEs (SONODEs):** To constrain the ANODE to learn second-order dynamics to give better interpretability for classical physics problems.
*   **Hamiltonian Neural Networks (HNN):** To incorporate physical priors through Hamiltonian mechanics, enforcing conservation laws and using reversible models to save memory.
*   **Lagrangian Neural Networks:** To apply Lagrangian mechanics to neural networks to enforce the conservation of total energy and model chaotic systems using arbitrary coordinates.

## Getting Started

1.  Clone the repository:
    ```bash
    git clone [repository_url]
    ```
2.  Install the required libraries using requirements.txt, if provided:
     ```bash
     pip install -r requirements.txt
     ```
3.  Navigate to the code directories to train and evaluate the models.
    *  The data generation scripts are in the `Simulation` directory.
    * The model training and evaluation scripts are located in the `models` directory.

## Citation

If you use this code or the findings in your research, please cite the dissertation:

```
Jagpreet Jakhar, "Neural ODEs for physical systems," MSc Dissertation, University of Sheffield, 2023
```
```
