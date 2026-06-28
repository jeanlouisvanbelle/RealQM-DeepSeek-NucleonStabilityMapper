# RealQM-DeepSeek Nucleon Stability Mapper

## Overview

This repository contains the code for running a full stability scan across the nuclide chart (Z = 1 to 20, N = Z to 3Z) using the RealQM Nuclear Engine V19.

The code maps the binding energy (ΔE) for each nuclide, generating a stability heatmap and identifying predicted stable isotopes based on the RealQM framework.

## Key Features

*   **Calibrated Parameters:** Uses the optimized parameters from the V19 calibration (`alpha_scale = 1.00`, `repulsion_strength = 0.50 MeV`).
*   **Dynamic Bounds:** Implements a physically motivated, adaptive search space to prevent artificial confinement of nucleons, especially for heavier nuclei.
*   **Stability Mapping:** Scans the (Z, N) plane and computes the binding energy for each nuclide.
*   **Visualization:** Generates a heatmap of ΔE and overlays the predicted stable isotopes.

## Installation & Dependencies

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/RealQM-DeepSeek-NucleonStabilityMapper.git
    cd RealQM-DeepSeek-NucleonStabilityMapper
