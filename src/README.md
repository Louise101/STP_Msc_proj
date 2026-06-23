# Hybrid Monte Carlo and Discrete Event Simulation of a Prostate Cancer Diagnostic Pathway

## Overview

This repository contains the code developed for my MSc dissertation in Clinical Scientific Computing. The project investigates the use of a hybrid Monte Carlo (MC) and Discrete Event Simulation (DES) approach to model the Welsh prostate cancer diagnostic pathway and evaluate the impact of the PROSTAD service redesign.

The model combines:

* **Monte Carlo simulation** to represent variability in patient pathways and waiting times using empirical distributions derived from historical data.
* **Discrete Event Simulation** to represent capacity-constrained resources, queueing behaviour and operational bottlenecks.
* **Hybrid MC-DES modelling** to evaluate how pathway redesign and resource allocation affect patient waiting times.

The work was completed as part of the NHS Scientific Training Programme (STP) MSc in Clinical Scientific Computing.

---

## Research Objectives

The project aimed to:

1. Develop a simulation model of the prostate cancer diagnostic pathway.
2. Verify model correctness through automated testing.
3. Validate simulated waiting times against observed pathway data.
4. Reproduce the effects of the PROSTAD service redesign.
5. Explore the impact of MRI and biopsy capacity on pathway performance.

---

## Repository Structure

```text
src/
├── analysis/            # Validation, scenario testing and plotting scripts
├── core/                # Core simulation classes and utilities
├── engine/              # Simulation engine and pathway logic
├── data_prep/           # Data processing and empirical input generation
├── runners/             # Scripts used to run simulation scenarios
├── test_refac/          # Automated verification tests
└── outputs/             # Generated figures and result files

data/
File contains fully annonymised waiting time data derived from the PROSTAD evaluation data. Note: no original patient data is contained within this repository.
```

---

## Simulation Approach

### Standard Pathway

The baseline model uses empirical waiting-time distributions derived from historical pathway data. Patients progress through pathway stages according to observed branching probabilities and waiting-time distributions.

### PROSTAD Pathway

The PROSTAD model incorporates operational changes introduced during the service redesign, including:

* Direct MRI access
* Accelerated MRI reporting
* Streamlined clinical decision-making
* Capacity-constrained MRI and biopsy resources

### Hybrid MC-DES Approach

Monte Carlo methods are used where historical distributions adequately describe pathway behaviour, while DES components are used where queueing and resource constraints influence waiting times.

---

## Validation

Model validation was performed using:

* Empirical cumulative distribution functions (ECDFs)
* Summary statistics
* Kolmogorov–Smirnov (KS) tests
* Comparison against observed standard pathway data
* Comparison against observed PROSTAD pathway data

Simulation results were pooled across 30 random seeds to reduce stochastic variation and improve estimate stability.

---

## Verification

The model was verified using automated unit tests implemented with `pytest`.

Tests cover:

* Sampling functions
* Branching logic
* Patient state transitions
* Queueing behaviour
* Resource allocation
* Pathway definitions
* Simulation engine behaviour

A total of 153 tests were executed successfully.

---

## Scenario Testing

Additional scenario exploration was performed by varying:

* Weekly MRI capacity
* Weekly biopsy capacity

These experiments were used to identify pathway bottlenecks and investigate the interaction between diagnostic resources.

---

## Requirements

Main Python packages used:

```bash
numpy
pandas
scipy
matplotlib
pytest
```



---

## Running the Model



```bash
python src/runners/run_sim.py
```

Validation scripts can be found within:

```text
src/analysis/
```

---

## Reproducing Dissertation Results

All figures and tables presented in the dissertation can be reproduced using the scripts within the `analysis` directory.

Output files are written to:

```text
outputs/
```

---

## Author

Louise Finlayson

MSc Clinical Scientific Computing
NHS Scientific Training Programme (STP)

---

## Licence

This repository is provided for academic and research purposes.
