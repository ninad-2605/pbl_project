# 👻 SPECTRE: From Signals to Motion: A Physics-Grounded Wi-fi Data Generator
### *High-Fidelity Synthetic WiFi CSI Generator for Human Activity Recognition*

[![Engine: SovereignRT](https://img.shields.io/badge/Engine-SovereignRT-red?style=for-the-badge)](https://github.com/)
[![Compute: NVIDIA Warp](https://img.shields.io/badge/Compute-NVIDIA_Warp-76B900?style=for-the-badge&logo=nvidia)](https://github.com/NVIDIA/warp)
[![Physics: ITU-R P.2040-1](https://img.shields.io/badge/Physics-ITU--R_P.2040--1-blue?style=for-the-badge)](https://www.itu.int/rec/R-REC-P.2040)

**Spectre** is a sophisticated simulation framework designed to bridge the gap between invisible radio waves and human biomechanics. By treating the indoor RF environment as a mathematically deterministic space, Spectre generates massive, high-fidelity datasets of **Channel State Information (CSI)**—allowing AI models to "see" human activity through the lens of radio signals without a single camera lens.

---

## 👁️ The Vision: Architecting the Unseen
The primary bottleneck in RF sensing is data. Real-world CSI collection is slow, hardware-dependent, and a nightmare to label accurately. **Spectre** solves this by:
* **Synthesizing Reality:** Utilizing the **AMASS** dataset to mirror actual human kinematics and biomechanics.
* **Physical Grounding:** A custom port of **NVIDIA Sionna** ray-tracing primitives to **NVIDIA Warp** for lightning-fast, GPU-accelerated simulation on Windows.
* **Perfect Labels:** Frame-perfect, 52-joint ground truth data that is automatically synchronized with the RF signal.

---

## 🛠️ Key Technical Pillars

### 1. SovereignRT: Diffraction-Aware Ray Tracing
The core engine, **SovereignRT**, leverages NVIDIA Warp kernels to simulate the literal Maxwellian interactions of 2.4/5GHz signals.
* **Multipath Propagation:** High-fidelity modeling of reflection, diffraction, and scattering.
* **Material Intelligence:** Frequency-dependent dielectric properties (permittivity/conductivity) for concrete, wood, glass, and human tissue per **ITU-R P.2040-1**.



### 2. The Bio-Digital Twin
Spectre doesn't just animate; it simulates biological presence.
* **Respiratory Mechanics:** Periodic mesh expansion modeling chest movement to simulate breathing-induced phase shifts.
* **Micro-Doppler Signatures:** Capturing subtle vibrations from heart rates and muscle tremors.
* **Procedural Kinematics:** 52-joint skeletal mapping with adjustable speed, gait variance, and distress levels.



### 3. Hardware Impairment Layer
To ensure "Sim-to-Real" transferability, Spectre injects real-world hardware artifacts found in commodity WiFi cards (e.g., Intel AX210):
* **Phase Drift:** Modeling oscillator instability and phase noise.
* **IQ Imbalance:** Simulating In-phase/Quadrature phase mismatches.
* **Thermal Noise Floor:** Realistic Signal-to-Noise Ratio (SNR) modeling.

---

## 🏗️ System Architecture

| Module | Responsibility | Technology |
| :--- | :--- | :--- |
| **Motion Engine** | Procedural IK & AMASS MoCap integration. | PyTorch / Python |
| **Physics Engine** | `SovereignRT` GPU-accelerated ray-scene interaction. | NVIDIA Warp / CUDA |
| **Radio Engine** | Synthesis of complex-valued CSI matrices ($H_{f,t}$). | NumPy / CuPy |

---

## 🖥️ Tech Stack
* **Compute:** NVIDIA Warp (Python-based CUDA kernels)
* **Motion:** AMASS (Mastering the Human Body Shape and Motion)
* **Standard:** ITU-R P.2040-1 (Material interaction models)
* **Linear Algebra:** PyTorch & NumPy for high-speed tensor operations.

---

## 🚩 Proof of Work
> *Current State: Sovereign V3 High-Fidelity Simulation. Demonstrating identity consistency across multi-human scenes with stable temporal evolution and physics-constrained motion.*

![Spectre Simulation Preview](your-video-link-here.gif)

---

## 🗺️ Roadmap
- [x] Port NVIDIA Sionna Ray Tracing to Windows via NVIDIA Warp.
- [x] Integrate AMASS biomechanical dataset for motion synthesis.
- [ ] **Phase 2:** Procedural generation of cluttered "Real-World" indoor environments (multipath fading).
- [ ] **Phase 3:** Public release of a 1000-hour synthetic CSI dataset on Kaggle.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
