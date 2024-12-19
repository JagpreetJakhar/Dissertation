# Neural ODEs for Physical Systems

This repository contains the code and resources for an MSc dissertation under supervision of [**Dr Matt Ellis**](https://www.sheffield.ac.uk/cs/people/academic/matt-ellis) in The University of Sheffeild. The application of Neural Ordinary Differential Equations (NODEs) to model complex physical systems is evaluated. The study focuses on three distinct physical systems: **the double pendulum, the Lorenz attractor, and the magnetic domain wall problem.** The objective is to investigate and compare the predictive performance of Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Neural ODEs.

| ![Double Pendulum Vector Fields](PNG/readme/1.gif "Double Pendulum Vector Fields") | ![Domain Wall](PNG/readme/dwall.gif "Domain Wall") |
|:--:|:--:|
| **Double Pendulum & Lorenz Attractor** | **Domain Wall** |


---


## Key Concepts

- **Neural Ordinary Differential Equations (NODEs)**: A neural network architecture that parameterizes the derivative of a system's state, enabling continuous-time modeling with ODE solvers popularised by Ricky TQ Chen in the paper [**Neural Ordinary Differential Equations**](https://arxiv.org/abs/1806.07366).
  
  <img src="PNG/readme/resnodevfields.png" alt="Neural ODE Architecture" width="250" height="180">

- **Chaotic Systems**: Dynamical systems with **sensitive dependence on initial conditions**, such as the double pendulum and the Lorenz attractor.
- **Recurrent Neural Networks (RNNs)**: Sequential models using hidden states to maintain memory over time.
- **Long Short-Term Memory Networks (LSTMs)**: Enhanced RNNs that use gating mechanisms to overcome vanishing gradient issues.
- **Adjoint Method**: An efficient backpropagation method for Neural ODEs that saves memory and ensures numerical stability.
- **ODE Solvers**: Numerical methods like Euler’s method, RK4, and Dormand-Prince (DOPRI5) to approximate ODE solutions.

---
## Methodology

*   A **Multilayer Perceptron** is parameterized using the Neural ODE in all problems.
*   **Torchdiffeq**, an ODE solver library for PyTorch, is used as a blackbox ODE solver with support for Adjoint-based backpropagation.
*   Different **batch sizes and activation functions** were used depending on the problem, including ReLU for the Lorenz problem, LeakyRelu for the Double Pendulum problem, and a combination of LeakyReLu and tanh for the domain wall problem.
* Different Loss Functions were also used for the problems, Huber Loss was able to perform better than MSE for Domain Wall Problem.
*   Evaluation was done based on **forecasting ability** and **generalization ability** .


## Experiments

### Double Pendulum
![Double Pendulum](PNG/readme/pendulum.gif)

**Double Pendulum:** A classic example of a chaotic system. Models predict the angles and angular velocities of the pendulum arms.

## Data Generation

The data for the Double Pendulum problem is created using ten random initial conditions and then using the torchdiffeq odeint solver to generate the data.

Then three model architectures are trained and tested on this data, namely a vanilla Recurrent Neural Network (RNN) and a vanilla Long Short Term Memory RNN (LSTM), which are used as a baseline to compare against the performance of the final NODE model. Mean Squared Error is used as the metric along with vector fields of the predictions made by the models to draw analysis and conclusions.





### Lorenz Attractor

![Lorenz Attractor](PNG/readme/lorenz.png)

**Lorenz Attractor**: A well-known chaotic system in atmospheric science. Models predict the evolution of state variables over time.

## Data Generation

The dataset for Lorenz attractor is made by using ten random initial points within a range of 15 to -15, which cause the attractor to behave chaotically, displaying the butterfly-like effect when plotted. The dataset is divided into:
- The first six points as the training set
- The seventh point as the validation set
- The last three as the test set

This approach is similar to that of the double pendulum considered previously.

Now, the three models are trained on the training data, Mean Squared Error, Visualisation, and Vector Fields, and evaluated to compare the three models. The abilities of the model being evaluated are Forecasting Ability and Generalizability.




### Magnetic Domain Wall Dynamics
![Domain Walls](PNG/readme/dw.png)

### Data Generation
The Magnetic Domain Wall dynamics are simulated using `DW oscillator.py`. The class is initialized with various physical parameters, including:

- Magnetic saturation moment (Ms)
- Exchange stiffness constant (A)
- Gilbert damping coefficient (α)
- Dimensions of the magnetic system (L)
- Anisotropy constants (a and b)
- External magnetic field amplitude (hconst)
- Oscillation frequency (f set to 0.5)
- Time-dependent external magnetic field (htime)

The function **run_field_sequence (RFD)** takes the input range of fields (low, high), the number of fields to be generated, the duration of each field, initial domain wall position, and angle to generate and simulate the set of fields randomly within that range. It returns the domain wall position, angle, and input sequence upon which the NODE is trained.

The external magnetic field, which serves as the oscillating input field to the domain wall, is also used by calculating its gradient using a function sequence. This gradient is attached in the forward pass of the Multilayer Perceptron to always have the correct driving force.

### ODE-Solver
The ODE-solver parametrizes a 5-layer MultiLayer Perceptron, which takes an input of size three and outputs two values: the predicted Domain Wall Position and Angle. The data simulated by the RFD function is first normalized to the range [-1, 1], as the domain wall position initially is at a different scale than the angles, which are in radians. These are then stacked together to form the data along with the time given by the RFD function.

This data is then fed to the ODE Solver using the adjoint method and Dormand-Prince solver (`dopri5`) to make a prediction on the domain wall position and angle using the initial position and time steps selected randomly from the data using a get batch function. The Huber Loss function is used over the Mean Squared Error Loss function, as it gives better results when training. Once the prediction error is calculated, backpropagation of error signals is done using the Adjoint method.

### Optimization
The trained model is tested on a set containing randomly generated fields by the RFD function. The AdamW optimizer, a modification of the Adam optimizer, is used, which decouples weight decay from the gradient update.

---

## Results

The models are analyzed by their performance on the
 three test sets one by one using **Error Metrics**, **visualizations** and **Vector Fields**.

### Lorenz Attractor Vector Fields
![Lorenz Attractor Vector Fields](PNG/readme/lor_vf.png)

### Double Pendulum Vector Fields
 ![Double Pendulum Vector Fields](PNG/readme/dp_vf.png)

### Domain Wall Dynamics (Single Field vs. Multiple Fields)

![Domain Wall Vector Fields](PNG/readme/dw_vf.png)

| ![Domain Wall Single Field](PNG/readme/dwsft.png) | ![Domain Wall Multiple Fields Predictions](PNG/readme/dwmft.png) |
|:--:|:--:|
| **Domain Wall Single Field Predictions** | **Domain Wall Multiple Fields Predictions** |

## Key Observations:
- Neural ODEs are a significant improvement over RNNs and LSTMs for modeling dynamical and chaotic systems over long time horizons. However, they are still far from accurately simulating these systems.
- Neural ODEs as an architecture have inherent limitations, which restrict the functions they can learn or represent i.e. they are not universal approximations.
- NODEs struggle to learn the dynamics of the domain wall, despite the fact that these are coupled first-order differential equations, which they should theoretically be able to learn.

---

## Future Work
1. **Augmented Neural ODEs (ANODEs):** Improve NODEs by increasing the dimensions of the state space.
2. **Second-Order Neural ODEs (SONODEs):** Better suited for systems with inherent second-order dynamics.
3. **Hamiltonian Neural Networks (HNNs):** Enforce conservation laws for physical interpretability.
4. **Lagrangian Neural Networks:** Use Lagrangian mechanics for energy conservation in modeling chaotic systems.

---
<!-- Acknowledgements Section -->
## Acknowledgements
### Special thanks to:

- [**Dr Matt Ellis**](https://www.sheffield.ac.uk/cs/people/academic/matt-ellis), for his Guidance and Supervision throughout my dissertation journey.
- I would like to thank [**Patrick Kidger**](https://kidger.site/) for his very helpful Phd theses [On Neural Differential Equations](https://arxiv.org/abs/2202.02435) and his implementation of torchdiffeq library.

- [**Alexander Norcliffe**](https://github.com/a-norcliffe), who helped out by pointing out the defeciencies in using First Order Neural ODEs for the Domain Wall Problem and suggested the approach of using Second Order Neural ODEs.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/JagpreetJakhar/Dissertation

2. Install the required libraires:
    ```
    pip install -r requirements.txt

    ```
3. Run Code for the three problems from their  Folder names:
    - Navigate to the `Double_Pendulum` folder and run the following files for each model:
        - `pendulum_rnn.ipynb`: Implementation of RNN for the double pendulum.
        - `pendulum_LSTM.ipynb`: Implementation of LSTM for the double pendulum.
        - `pendulum_NODE.ipynb`: Implementation of NODE for the double pendulum.
        

    - Navigate to the `Lorenz` folder and run the following scripts for each model:
        - `lorenz_rnn.ipynb`: Implementation of RNN for the Lorenz attractor.
        - `lorenz_lstm.ipynb`: Implementation of LSTM for the Lorenz attractor.
        - `lorenz_node.ipynb`: Implementation of NODE for the Lorenz attractor.
        
    - Navigate to the `Domain_Wall` folder and run the following scripts:
        - `single_field_node.ipynb`: NODE implementation for the domain wall under a single field.
        - `multiple_fields_node.ipynb`: NODE implementation for the domain wall under multiple fields.