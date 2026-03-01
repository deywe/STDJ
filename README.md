# 📡 STDJ Telemetry Generator & Analyzer (telemetria.py)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue.svg)](https://pandas.pydata.org/)

## 🚀 Overview
**telemetria.py** is the core component of the SPHY engine simulation framework. It generates synthetic telemetry data representing the stabilization of quantum fields within a **Sub-Planck Regime** using the **Fair Fold Temporal System (STDJ)** concept.

This script simulates the interaction between the number of qubits, vacuum penetration factors, and phase integrity (veracity) to calculate warp metrics that exceed conventional informational propagation limits ($>100c$).

## 🔒 Features
* **Warp Field Simulation:** Calculates `STDJ Time` and `Warp Velocity` based on quantum field inputs.
* **Secure Data Logging:** Saves telemetry to **Apache Parquet** format for efficient storage and analysis.
* **Cryptographic Integrity:** Each frame is signed using **SHA256** to ensure dataset immutability.
* **Progress Tracking:** Uses `tqdm` for interactive progress visualization.
* **Detailed Metrics Analysis:** Generates a comprehensive summary comparing stable and turbulent frames.
* **Visualization:** Automatically generates a heatmap of **Fidelity vs. Time** to analyze turbulence patterns.

## 🛠️ Prerequisites
You will need Python 3.8+ and the following libraries:

```bash
pip install numpy pandas pyarrow tqdm matplotlib seaborn
🖥️ Usage
Run the script directly from your terminal:

Bash
python3 telemetria.py
📋 Interaction Flow
The script will prompt you for the total number of frames to simulate (e.g., 1200).

It will run the simulation, showing a progress bar.

It will save the signed dataset to stdj_secure_telemetry.parquet.

It will print detailed metrics to the console.

It will generate and display a heatmap image (warp_fidelity_heatmap.png).

📊 Example Output
Plaintext
Enter the total number of frames to simulate: 1200
🚀 STARTING STDJ FAIR FOLD SIMULATOR - V22 (SECURE MODE)
📡 ACTIVE FIELD: 1200 Qubits | REGIME: SUB-PLANCK
----------------------------------------------------------------------
Simulating Warp Fields: 100%|██████████| 1200/1200 [00:24<00:00, 49.18it/s]
----------------------------------------------------------------------
✅ SIMULATION COMPLETED.
💾 Telemetry signed and saved to: stdj_secure_telemetry.parquet

============================== DETAILED METRICS ==============================
Total Frames: 1200
Stable Frames: 376 (31.3%)
Turbulent Frames: 824 (68.7%)
----------------------------------------------------------------------
Avg Warp Speed (Stable): 99.49e10 c
Min Fidelity (Stable): 0.990034
Avg Warp Speed (Turbulent): 97.99e10 c
Max Turbulence Dip (Veracity): 0.970011
==============================================================================
🎨 Generating heatmap of Veracity vs. Time...
💾 Heatmap saved as: warp_fidelity_heatmap.png
🖥️ Opening heatmap image...
🔒 Security Note
The dataset generated (.parquet) contains a signature column for each frame, ensuring that data integrity can be verified post-simulation.

Project initiated by Deywe Okabe - Black Swan Research
