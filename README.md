# Dissertation: Modeling Complex Physical Systems using Neural Ordinary Differential Equations (Neural ODEs)

This repository contains the code and research conducted as part of a thesis exploring the application of **Neural Ordinary Differential Equations (Neural ODEs)** in modeling complex physical systems. The study compares the performance of different machine learning techniques, specifically **Recurrent Neural Networks (RNNs)**, **Long Short-Term Memory Networks (LSTMs)**, and **Neural ODEs** in modeling chaotic and nonlinear systems.

## Abstract

This thesis investigates how Neural ODEs can be used to model physical systems with chaotic dynamics. The study focuses on three distinct physical systems:

1. **Double Pendulum**: A chaotic, nonlinear dynamical system.
2. **Lorenz Attractor**: A system of equations that model atmospheric convection, known for its chaotic behavior.
3. **Domain Wall Problem**: A problem in spintronics that deals with the motion of domain walls in magnetic materials.

The goal is to compare the predictive capabilities of Recurrent Neural Networks (RNNs), Long Short-Term Memory Networks (LSTMs), and Neural ODEs in modeling these systems. This research aims to contribute to understanding the strengths and limitations of these models in capturing the intricacies of chaotic and complex dynamics.

## Project Structure

The repository is organized as follows:


## System Overview

### Double Pendulum
A chaotic and nonlinear dynamical system that is sensitive to initial conditions. The double pendulum serves as an excellent benchmark for testing how well the models can predict chaotic motion.

![Double Pendulum](images/double_pendulum.png)

### Lorenz Attractor
A system that exhibits chaotic behavior, often used as a model for atmospheric convection. We compare how the models capture the system’s highly nonlinear dynamics.

![Lorenz Attractor](images/lorenz_attractor.png)

### Domain Wall Problem
A physical problem in spintronics, where we explore the movement of domain walls in magnetic materials. Neural ODEs are tested against conventional neural networks in this context.

![Domain Wall](images/domain_wall.png)

## Key Models Implemented

1. **Recurrent Neural Networks (RNNs)**:
   - Traditional RNNs used for sequential data prediction.
   - Known for capturing temporal dependencies but limited in handling long-range dependencies.

2. **Long Short-Term Memory Networks (LSTMs)**:
   - An extension of RNNs, LSTMs address issues of vanishing gradients and are better at capturing long-term dependencies.

3. **Neural Ordinary Differential Equations (Neural ODEs)**:
   - A continuous-time model that uses a differential equation to model system dynamics.
   - Neural ODEs are flexible in representing complex, continuous, and chaotic systems.

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/JagpreetJakhar/Dissertation.git
cd Dissertation
pip install -r requirements.txt
