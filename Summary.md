# SUMMARY

This repository provides the **supplementary code** and **experimental materials** for the paper:  
**“A Physics-Informed Neural Networks-Based Model Predictive Control Framework for SIR Epidemics”**  
by *Aiping Zhong*, *Baike She*, and *Philip E. Paré* (IEEE OJ-CSYS, 2025).  

The work introduces an MPC–PINNs closed-loop framework for real-time epidemic control under **noisy infected-state observations**, featuring two enhanced variants—**MPC-LS-PINNs** and **MPC-SI-PINNs**—and a generalized extension for **joint estimation of transmission and recovery rates**.

---

## Repository Structure

- **`PINNs-Based MPC Frameworks/`**  
  Main implementations of the proposed frameworks:  
  - `MPC-PINNs`: baseline joint estimation and control algorithm  
  - `MPC-LS-PINNs`: log-scaled formulation for improved noise robustness  
  - `MPC-SI-PINNs`: split-integral formulation for faster convergence  
  - Generalized versions (`MPC-S-PINNs`, etc.) under known $R_0$
  - Extended scripts (`Generalized_MPC_PINNs_unknownR0`, `Generalized_MPC_LS_PINNs_unknownR0`, `Generalized_MPC_S_PINNs_unknownR0`) for experiments **without** known $R_0$, as discussed in Section IV-C.

- **`Comparison of Different Neural Network Architectures/`**  
  Scripts evaluating SISO, MISO, SIMO, and MIMO network configurations as discussed in Section IV-D.

- **`Comparison with the Extended Kalman Filter/`**  
  Implementation of EKF-based estimation for benchmarking under the same noise conditions (Section IV-E).

- **`Case Study with Real-World Data/`**  
  Experiments using the **COVID-19 dataset from Italy (Oct 2021 – May 2022)** for external validation (Section IV-F).

- **`Supporting Material_Sensitive Analysis/`**  
  Parameter-sensitivity texperiments under different **noise intensities** ($\kappa$) and  **SIR model parameters**.

- **`Supporting Material_SIRS/`**  
Examine the extensibility of the proposed MPC-PINNs framework to **SIRS models with temporary immunity**.

---

## Purpose

This repository aims to:

- Provide **runnable implementations** to reproduce all numerical results and illustrative figures in the paper.  
- Serve as a **reference framework** for integrating physics-informed neural networks with **model predictive control** in dynamic state–parameter estimation problems.  
- Facilitate **extensions** to other epidemic models or control tasks involving partial-state observations and uncertain parameters.
