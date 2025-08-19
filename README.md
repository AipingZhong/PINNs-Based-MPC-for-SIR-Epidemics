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

&nbsp; - GPU is recommended but not required



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



⚠️ If no GPU is available, edit `requirements.txt` and replace  

`tensorflow-gpu==2.10.0` with `tensorflow==2.10.0`.



---



## 3. Code Structure

- `code/PINNs-Based MPC Frameworks/`  

&nbsp; Main implementations of **MPC-PINNs**, **MPC-LS-PINNs**, **MPC-SI-PINNs**, and their generalized variants.  

- `code/Supporting Material_NN Architectures/`  

&nbsp; Scripts for architecture comparisons.  

- `code/Supporting Material_Sensitive Analysis/`  

&nbsp; Scripts for parameter-sensitivity experiments.  

- `requirements.txt`  

&nbsp; Dependency list.  

- `SUMMARY.md`  

&nbsp; Overview of supplementary materials.  



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

@article{Zhong2025,
  author  = {Aiping Zhong and Baike She and Philip E. Paré},
  title   = {A Physics-Informed Neural Networks-Based Model Predictive Control Framework for SIR Epidemics},
  journal = {IEEE Open Journal of Control Systems},
  year    = {2025},
  note    = {submitted}
}

```



