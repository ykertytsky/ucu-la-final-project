# Voice Denoising via Linear Algebra (SVD on Hankel Matrices)

A course project studying how linear algebra can be applied to **speech enhancement**: recovering an intelligible speech signal from a noisy recording.

---

## 1. Introduction and Motivation

Voice denoising belongs to the broader research area of **speech enhancement**, whose goal is to recover an intelligible and perceptually clear speech signal from a recording with noise present. In practical settings, voice recordings are often shifted by environmental noise, microphone noise, or transmission artifacts. Denoising is an important part of telecommunication systems, assistive technologies, HRI (Human–Robot Interaction), and any pipeline in which speech is used as input for further processing.

From a mathematical perspective, the project is based on the assumption that an observed audio signal can be modeled as

$$
x(t) = s(t) + n(t),
$$

where $s(t)$ is the useful speech signal and $n(t)$ is the added noise. In this project, we observe only the mixed signal and want to recover the voice part as well as possible. The broader goal is to study how **linear algebra** can be applied to this problem. The main idea is that speech usually has more structure and repeated patterns than random noise; we can transform the signal into a matrix and analyze it using linear algebra methods.

---

## 2. Project Aim and Objectives

**Main aim:** study how linear algebra can be applied to the problem of voice denoising.

**Objectives:**

- Model noisy audio as a sum of voice and noise and describe the denoising task in mathematical form.
- Represent a one-dimensional audio signal as a matrix suitable for SVD.
- Decompose that matrix into principal components and identify the dominant components associated with speech.
- Reconstruct a denoised version of the signal by retaining only the most significant singular values.
- Convert the reconstructed matrix back into a one-dimensional audio waveform.
- Evaluate the resulting audio qualitatively and, where possible, quantitatively.
- Compare this method with other possible approaches in later stages of the project.

These objectives follow the workflow: mathematical model → matrix representation → keep dominant singular values → reconstruct the signal.

---

## 3. Existing Approaches and Chosen Method

### Classical filtering methods

Classical filtering passes the signal through a filter that removes some frequency ranges (low-pass, high-pass, band-pass). This can work when noise and speech occupy clearly different bands; often they overlap, so filters may remove useful speech together with noise.

### Spectral methods

Spectral methods work in the frequency domain: transform with the Fourier transform, estimate noise-dominated parts of the spectrum, attenuate them, then reconstruct. Examples include spectral subtraction and Wiener filtering. They are flexible but depend on good noise estimates; poor estimates cause distortions or artifacts.

### Machine learning and deep learning

Neural methods learn mappings from noisy to clean speech from data and can perform very well with large datasets and compute. For this project they serve as a **comparison point**; the focus is on a linear-algebra perspective rather than training large models.

### Matrix decomposition methods

Matrix methods transform the 1D signal into a matrix and use decompositions to separate dominant speech-like patterns from weaker, noise-like components. They are interpretable and do not require large labeled training sets.

**Chosen method (current stage):** a matrix decomposition based on **SVD**. The audio is embedded in a structured **Hankel matrix**, then

$$
H = U \Sigma V^{T}
$$

is computed. The singular values in $\Sigma$ are ordered by importance. We assume that **the largest singular values capture the main structure of speech**, while smaller values are more likely to represent noise.

**Why SVD + Hankel:** (1) aligns with course linear algebra; (2) no large labeled dataset required; (3) denoising strength is controlled by how many singular values we keep. Other approaches may be explored and compared later.

---

## 4. Proposed Methodology

**Input:** a noisy voice recording (microphone or a public dataset). The waveform is a sample vector

$$
x = (x_1, x_2, \dots, x_N),
$$

with sampling rate in a typical speech range (e.g. 16 000–44 100 Hz).

**Hankel embedding:** choose a window length $L$ and build

$$
H =
\begin{bmatrix}
x_1 & x_2 & \cdots & x_L \\
x_2 & x_3 & \cdots & x_{L+1} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N-L+1} & x_{N-L+2} & \cdots & x_N
\end{bmatrix}.
$$

**Truncated SVD:** keep the first $k$ singular values:

$$
H_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{T}.
$$

$k$ is chosen experimentally: too small loses speech; too large leaves noise.

**Reconstruction:** the rank-$k$ matrix may not be exactly Hankel. Recover a 1D signal by **anti-diagonal averaging (Hankelization)**—average entries on each anti-diagonal to form one sample per index.

---

## 5. Implementation Plan and Pseudocode

**Environment:** Python, with **NumPy** for arrays and matrix work, **SciPy** for linear algebra utilities, and **librosa** / **soundfile** for reading and writing audio. The priority is a clear experimental pipeline, not micro-optimization.

**Core pipeline (pseudocode):**

```text
Input noisy audio signal x
Load and normalize the waveform
Choose a window length L
Construct the Hankel matrix H from x
Compute the singular value decomposition H = UΣV^T
Select a truncation rank k
Form the low-rank approximation H_k = U_k Σ_k V_k^T
Reconstruct the one-dimensional signal by anti-diagonal averaging
Save the reconstructed waveform as denoised output
```

The main design knob is the truncation rank $k$ (and, jointly, the window length $L$).

---

## 6. Evaluation and Data

- **Listening:** compare noisy vs. denoised by ear.
- **Visualization:** waveforms and spectrograms to see noise reduction vs. preserved speech structure.
- **Quantitative metrics** (if clean references exist): e.g. SNR improvement or reconstruction error.

**Data sources (planned):** self-recorded microphone audio and public speech data online. Place raw/processed assets under `data/` as the project evolves.

---

## 7. Future Work and Timeline

Next steps are mostly experimental:

- Finalize the implementation end-to-end.
- Run experiments on multiple audio samples.
- Tune the number of retained singular values (and window length).
- Analyze results and compare with other linear-algebra or baseline methods where appropriate.
- Prepare the final report.

---

## 8. Challenges and Risks

| Challenge | Notes / mitigation |
|-----------|-------------------|
| Choosing $k$ (and $L$) | Affects denoising vs. speech preservation; sweep values and listen. |
| Approximate assumptions | Noise is not always in weak singular values; structured noise can appear in strong components. |
| Data limitations | Use several samples; document limitations. |
| Objective quality | Combine metrics (when possible) with listening and plots. |

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Python source for the denoising pipeline and experiments |
| `data/` | Audio files and derived outputs (gitignored or tracked as you prefer) |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
