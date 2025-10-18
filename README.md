# Quin Beta 21.ab1 (8B)
### [INTERNAL RESEARCH ARTIFACT] — Experimental Model v0.2.1

**Model Identifier:** `Quin-Beta-21.ab1`  
**Status:** `EXPERIMENTAL (v0.2.1)`  
**Developed by:** Dr. ChamYoung & Seo Park Jun  
**Primary Purpose:** Architectural validation and optimization testing  
**Parent Architecture:** Quin-91 (910B)

---

## Abstract

**Quin Beta 21.ab1 (8B)** is a pre-production internal research artifact, developed as a public testbed for experimental architectural hypotheses and optimization strategies.

It is not a general-purpose foundational model. The model serves as a compact 8B-parameter prototype designed to validate two novel components:

1. **Resonant Gating Units (RGUs)** — Frequency-modulated activation and selective neuron gating.  
2. **Causal State Tensors (CSTs)** — Predictive vector-field–based context management.

This version is intended for academic analysis and architectural ablation, not for deployment or production inference.

---

## Research Context

The Quin architecture diverges from standard transformers by treating latent space as a **predictive vector field**, rather than a static memory structure. This approach seeks to optimize representational efficiency and contextual foresight.

| Component | Description |
|------------|-------------|
| **RGU (Resonant Gating Unit)** | Frequency-based dynamic gating replacing standard FFN layers. |
| **CST (Causal State Tensor)** | Fourth-order tensor predicting future state transitions. |
| **APP (Asymmetric Parameter Projection)** | Downscaled parameter projection from Quin-91’s 910B latent map. |

---

## RGU Activation (Conceptual)

The RGU activation function **φ(x)** is governed by a resonance function **R**, modulating input **x** via a learned frequency matrix **Ω** and a gate **G**:

\[
ϕ(x) = G(x, W_g) · tanh(R(x, Ω) ⊙ (xW_v))
\]

where resonance **R(x, Ω)** is defined as:

\[
R(x, Ω) = cos^2(proj(x, W_k) · Ω)
\]

This structure enforces resonance-driven activation pathways, forming the theoretical core of **Quin-21 Beta**.

---

## Model Description

| Property | Specification |
|-----------|----------------|
| **Model Type** | quin (Custom Architecture) |
| **Variant** | 21.ab1 (Asymmetric Beta 1) |
| **Parameter Count** | ~8.1B (Projected from Quin-91) |
| **Parent Model** | Quin-91 (910B) |
| **Context Window** | 8192 Tokens (Truncated) |
| **Gating Mechanism** | Resonant Gating Unit (RGU) |
| **State Management** | Causal State Tensor (T_cs) |

---

## Architectural Innovations (Quin-Arch v0.2b)

| Innovation | Description |
|-------------|-------------|
| **Asymmetric Parameter Projection (APP)** | Projects a high-fidelity subspace from Quin-91’s 910B latent map, preserving conceptual gradients and scaling effects. |
| **Resonant Gating Units (RGU)** | Replaces FFN/MoE layers, activating only frequency-aligned neurons, reducing FLOPs per token. |
| **Causal State Tensors (CST)** | Eliminates KV cache. Predicts context trajectory via a 4th-order causal tensor field. |

---

## Development Status & Disclaimers

| Category | Notes |
|-----------|-------|
| **Objective** | Internal testing of architectural optimization by Dr. ChamYoung and Seo Park Jun. |
| **Stability** | Unstable – not tuned for production. |
| **Safety** | No instruction tuning or alignment applied. |
| **Purpose** | Architectural transparency and academic evaluation only. |
| **Usage** | Do not deploy for real-world or production tasks. |

---

## Intended Use (Theoretical)

Direct integration into `transformers` is **not supported**.  
Model loading requires a **custom QuinLoader class** capable of initializing CST and RGU components.

---

### Illustrative Loader (Pseudocode)

```python
# WARNING: Pseudocode. This loader is not part of the public library.

from quin_arch_loader import QuinLoader, QuinConfig

config = QuinConfig(
    model_type="quin",
    variant="21.ab1",
    n_layers=32,
    rgu_expansion_factor=2.75,
    cst_dimensions=1024,
    projection_parent_hash="quin-91/proj-map-v2.bin"
)

model = QuinLoader.from_pretrained(
    "drchamyoung/Quin-Beta-21.ab1",
    config=config,
    device_map="auto",
    precision="bf16"
)

# model.get_architecture_report()
# > [Quin-21.ab1]: 8.1B Params. 32 RGU layers. CST dim 1024.
```

---

## Research Links

| Resource | URL |
|-----------|-----|
| **Primary Repo** | [https://github.com/InboraStudio](https://github.com/InboraStudio) |
| **Contributor** | [https://github.com/SeoParkjun](https://github.com/SeoParkjun) |
| **Discord Research Server** | [https://discord.gg/pVA8NZ5mru](https://discord.gg/pVA8NZ5mru) |
| **Datasets** | [https://huggingface.co/datasets/DrChamyoung/Quinbeta5.2DataSets](https://huggingface.co/datasets/DrChamyoung/Quinbeta5.2DataSets) |

---

## Dataset API (Hugging Face)

**Retrieve Rows:**
```bash
curl -X GET "https://datasets-server.huggingface.co/rows?dataset=DrChamyoung%2FQuinbeta5.2DataSets&config=default&split=train&offset=0&length=100"
```

**List Splits:**
```bash
curl -X GET "https://datasets-server.huggingface.co/splits?dataset=DrChamyoung%2FQuinbeta5.2DataSets"
```

**List Parquet Files:**
```bash
curl -X GET "https://huggingface.co/api/datasets/DrChamyoung/Quinbeta5.2DataSets/parquet/default/train"
```

---

## Citation

If referencing this artifact in academic or technical research:

> Dr. ChamYoung, Seo Park Jun. *Quin Beta 21.ab1 (8B) — Experimental Architectural Model*. Inbora Studio Research Division, 2025.

---

## License

This model and documentation are released for **non-commercial academic research** only.  
All intellectual property rights reserved to **Inbora Studio Research Group**.

---

## Acknowledgments

- **Architectural Design:** Dr. Chamyoung AKA Alok  
- **Optimization & Validation:** Seo Park Jun  
- **Infrastructure:** Inbora Studio Labs  
- **Dataset Preparation:** DrChamyoung AKA Alok Research Division
