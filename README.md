# EpiPINN – Supplementary Code



This repository contains the supplementary code for the paper:  

**“A Physics-Informed Neural Networks-Based Model Predictive Control Framework for SIR Epidemics”** (submitted to IEEE Open Journal of Control Systems).



---



## 1. Tested Environment

- Python 3.10 (recommended)

- TensorFlow 2.10.0 (`tensorflow-gpu==2.10.0`; use `tensorflow==2.10.0` if GPU not available)

- SciANN 0.7.0.1  

- CasADi 3.6.7  

- NumPy 1.24.4, SciPy 1.14.1, Matplotlib 3.9.1, Pandas 2.2.3, Scikit-learn 1.5.2  

- Tested on:

&nbsp; - **Linux** (Ubuntu 20.04)

&nbsp; - NVIDIA RTX 4090 GPU + AMD EPYC 7502 CPU  




---



## 2. Installation

Create a virtual environment and install dependencies:



```bash

# Create and activate a virtual environment

python -m venv .venv

source .venv/bin/activate        # On Windows: .venvScriptsactivate



# Install dependencies

pip install -r requirements.txt

```



---


## 3. Code Structure

- **`PINNs-Based MPC Frameworks/`**  
  Main implementations of the proposed frameworks:  
  - `MPC-PINNs`: baseline algorithm for joint estimation and control.  
  - `MPC-LS-PINNs`: log-scaled loss for enhanced noise robustness.  
  - `MPC-SI-PINNs`: split-integral training for faster convergence.  
  - Generalized versions (`MPC-S-PINNs`, etc.) for simultaneous estimation of $\beta$ and $\gamma$ under known $R_0$.
  -  Additional scripts (`Generalized_MPC_PINNs_unknownR0`, `Generalized_MPC_LS_PINNs_unknownR0`, `Generalized_MPC_S_PINNs_unknownR0`) are provided to evaluate the framework performance when the basic reproduction number $R_0$ is **unknown**, following the ablation setup described in Section IV-C.


- **`Comparison of Different Neural Network Architectures/`**  
  Experiments evaluating SISO, MISO, SIMO, and MIMO neural network designs, as discussed in Section IV-D.

- **`Comparison with the Extended Kalman Filter/`**  
  Benchmark experiments comparing the proposed PINNs-based estimators with the classical EKF method (Section IV-E).

- **`Case Study with Real-World Data/`**  
  Validation using real COVID-19 data from Italy (Oct 2021 – May 2022), corresponding to Section IV-F of the paper.

- **`Supporting Material_Sensitive Analysis/`**  
Parameter-sensitivity texperiments under different **noise intensities** ($\kappa$) and  **SIR model parameters**.

- **`Supporting Material_SIRS/`**  
Examine the extensibility of the proposed MPC-PINNs framework to **SIRS models with temporary immunity**.

- **`requirements.txt`**  
  List of dependencies and package versions.

- **`SUMMARY.md`**  
  Overview of the repository content and experiment organization.



---



## 4. Quick Start

Run one of the provided scripts, for example:



```bash

python MPC_LS_PINNs.py

```



- Default settings reproduce the simulation reported in the paper.  

- To **quickly test** functionality, reduce the number of training epochs or collocation points in the script.  

- Output (plots and metrics) will be saved in the current directory.  



---



## 5. Data

- The code generates **synthetic SIR trajectories internally**.  

- **No external dataset is required.**  

- To use your own epidemic data, replace the synthetic data generation block in each script.  



---



## 6. License

This project is released under the **MIT License** (see `LICENSE` file).  



---



## 7. Citation

If you use this code, please cite the paper:



```

@article{zhong2025physics,
  title={A Physics-Informed Neural Networks-Based Model Predictive Control Framework for $ SIR $ Epidemics},
  author={Zhong, Aiping and She, Baike and Par{\'e}, Philip E},
  journal={arXiv preprint arXiv:2509.12226},
  year={2025}
}

```



