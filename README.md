# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech

Official implementation of *"Forgotten Words"*, submitted to BioNLP @ ACL. This study presents the **first NLP evaluation of dementia classification in conversational Filipino speech**, benchmarking TF-IDF + Logistic Regression, BERT-base-uncased, and NeoBERT across monolingual, cross-lingual zero-shot, and bilingual training configurations.

---

## Overview

Dementia detection from speech has shown strong performance in English, yet remains underexplored in low-resource, code-switched settings. Filipino conversational speech — characterized by Tagalog–English code-switching and the absence of native clinical corpora — presents a unique and underserved challenge for clinical NLP.

This work investigates:
1. Whether NLP-based dementia classifiers trained on English generalize to Filipino under zero-shot cross-lingual transfer.
2. Whether NeoBERT's advances over standard BERT translate into improved cross-lingual robustness in clinical settings.
3. The extent to which bilingual training can bridge the cross-lingual performance gap.

Our key finding: **training data coverage is a stronger determinant of cross-lingual robustness than architectural design**. Models trained monolingually achieve near-perfect in-domain F1 (≥ 0.98) but collapse under zero-shot transfer (F1 ≤ 0.40), primarily due to failure to detect dementia-class samples. Bilingual training reduces this gap to ∆F1 < 0.01 across all models.

---

## Repository Structure

```
Filipino-English-Dementia-Classification/
├── BERT/                          # BERT fine-tuning scripts
│   ├── bert_english.py            #   Monolingual English training
│   ├── bert_tagalog.py            #   Monolingual Tagalog training
│   └── bert_bilingual.py          #   Bilingual (EN + TL) training
├── NeoBERT/                       # NeoBERT fine-tuning scripts
│   ├── neobert_english.py         #   Monolingual English training
│   ├── neobert_tagalog.py         #   Monolingual Tagalog training
│   └── neobert_bilingual.py       #   Bilingual (EN + TL) training
├── Baseline/                      # Non-neural baseline
│   └── tf_idf_and_lr.py           #   Bigram TF-IDF + Logistic Regression
├── requirements_baseline.txt      # Dependencies for the baseline
├── requirements_transformers.txt  # Dependencies for BERT / NeoBERT
└── README.md
```

---

## Dataset

We construct a balanced bilingual dataset of **4,000 conversational transcripts** for binary dementia classification (0 = Healthy Control, 1 = Dementia), with 1,000 samples per class per language.

**English sources:**
- Dementia-positive: DementiaBank, DementiaNet (structured clinical elicitation tasks)
- Healthy controls: TalkBank-style conversational corpora and public discourse datasets

**Filipino sources:**
- Healthy controls: YouTube conversational transcripts and the ASR-SFDuSC Filipino Corpus
- Dementia-positive: Controlled manual translation of English dementia transcripts into Filipino by Filipino-speaking validators, with explicit preservation of clinically relevant disfluency patterns (repetitions, hesitations, syntactic degradation). Machine translation was explicitly avoided to prevent semantic normalization.

**Preprocessing:** Unicode normalization, whitespace normalization, lowercasing. Disfluencies are intentionally retained as clinical markers. No stemming, lemmatization, or syntactic parsing is applied. All inputs are truncated to 128 tokens.

> The dataset files (`english.xlsx`, `tagalog.xlsx`) are not included in this repository. Each file is a headerless two-column Excel sheet: `transcript text | label (0 or 1)`.

---

## Models

### TF-IDF + Logistic Regression (Baseline)
Bigram TF-IDF (unigrams + bigrams, max 20,000 features, sublinear TF scaling, `min_df=2`, `max_df=0.95`) with ℓ2-regularized Logistic Regression (`solver=liblinear`, `class_weight=balanced`, `max_iter=2000`). Provides a reference for lexical separability independent of contextual modeling.

### BERT (`bert-base-uncased`)
Standard 12-layer BERT encoder fine-tuned end-to-end for binary classification. The default `[CLS]` pooling is replaced with **attention-masked mean pooling** over all final hidden states, which aggregates distributed token-level information and improves robustness under domain shift.

### NeoBERT (`chandar-lab/NeoBERT`)
A modernized BERT-style encoder with rotary positional embeddings (RoPE), Pre-LayerNorm with RMSNorm, and a significantly larger curated English pretraining corpus. Fine-tuned with the same pipeline as BERT, enabling a controlled comparison of architectural and pretraining effects. Requires `trust_remote_code=True`.

Both transformer models are pretrained **exclusively on English**, ensuring that Filipino performance reflects cross-lingual transfer rather than multilingual pretraining exposure.

---

## Experimental Design

### Training Configurations
| Configuration | Training Data | Evaluation |
|---|---|---|
| Monolingual English | English only | English (in-domain), Filipino (zero-shot), Combined |
| Monolingual Filipino | Filipino only | Filipino (in-domain), English (zero-shot), Combined |
| Bilingual | English + Filipino | Combined, English subset, Filipino subset |

### Hyperparameters
All transformer models use AdamW (β₁=0.9, β₂=0.999, ε=1e-8), gradient clipping at norm 1.0, linear warmup (10% of steps) + linear decay, dropout p=0.1, and max 10 epochs. Hyperparameters are selected via grid search on a stratified 70–15–15 train/validation/test split.

| Model | Training Config | Batch Size | Learning Rate | Weight Decay |
|---|---|---|---|---|
| NeoBERT | English | 4 | 6×10⁻⁶ | 1×10⁻⁵ |
| NeoBERT | Filipino | 8 | 2×10⁻⁵ | 1×10⁻² |
| NeoBERT | Bilingual | 8 | 6×10⁻⁶ | 1×10⁻⁵ |
| BERT-base | English | 8 | 3×10⁻⁵ | 1×10⁻² |
| BERT-base | Filipino | 8 | 3×10⁻⁵ | 1×10⁻² |
| BERT-base | Bilingual | 4 | 3×10⁻⁵ | 1×10⁻⁵ |

### Evaluation Protocol
- **10-fold Stratified K-Fold Cross-Validation** (seed 42), results reported as mean ± std across folds
- **In-domain:** train and test within the same language
- **Cross-lingual zero-shot:** train on one language, evaluate on the full dataset of the other (never seen during training)
- **Bilingual:** train on combined corpus, evaluate on mixed-language held-out fold with per-language subsets
- **Primary metric:** Macro F1 (equal weighting of Healthy and Dementia classes)
- **Secondary metrics:** Accuracy, per-class F1, Dementia-class recall (sensitivity)
- **Cross-lingual generalization gap:** ∆F1 = |F1_in-domain − F1_cross-lingual|

---

## Results

### Overall Performance (Accuracy / Macro-F1)

| Model | Train | English | Filipino | Combined | ∆F1 |
|---|---|---|---|---|---|
| BERT | EN | 0.988 / 0.988 | 0.530 / 0.393 | 0.764 / 0.751 | 0.595 |
| BERT | TL | 0.601 / 0.523 | 0.981 / 0.981 | 0.783 / 0.783 | 0.458 |
| BERT | EN+TL | 0.990 / 0.990 | 0.975 / 0.975 | 0.983 / 0.983 | 0.015 |
| NeoBERT | EN | 0.990 / 0.990 | 0.508 / 0.352 | 0.751 / 0.735 | 0.639 |
| NeoBERT | TL | 0.643 / 0.588 | 0.988 / 0.988 | 0.815 / 0.809 | 0.400 |
| NeoBERT | EN+TL | 0.990 / 0.990 | 0.984 / 0.984 | 0.987 / 0.987 | 0.006 |
| TF-IDF + LR | EN | 0.944 / 0.942 | 0.502 / 0.009 | 0.743 / 0.654 | 0.933 |
| TF-IDF + LR | TL | 0.500 / 0.000 | 0.975 / 0.974 | 0.746 / 0.659 | 0.974 |
| TF-IDF + LR | EN+TL | 0.937 / 0.960 | 0.984 / 0.960 | 0.960 / 0.960 | 0.000 |

### Class-Level Performance (F1 Healthy / F1 Dementia / Dementia Recall)

| Model | Train | English | Filipino | Combined |
|---|---|---|---|---|
| BERT | EN | 0.988 / 0.987 / 0.988 | 0.680 / 0.106 / 0.060 | 0.809 / 0.693 / 0.535 |
| BERT | TL | 0.715 / 0.330 / 0.201 | 0.981 / 0.981 / 0.982 | 0.826 / 0.739 / 0.593 |
| BERT | EN+TL | 0.990 / 0.990 / 0.990 | 0.975 / 0.976 / 0.966 | 0.983 / 0.982 / 0.978 |
| NeoBERT | EN | 0.991 / 0.990 / 0.986 | 0.670 / 0.033 / 0.017 | 0.800 / 0.669 / 0.504 |
| NeoBERT | TL | 0.738 / 0.438 / 0.287 | 0.988 / 0.988 / 0.981 | 0.844 / 0.774 / 0.633 |
| NeoBERT | EN+TL | 0.990 / 0.990 / 0.985 | 0.984 / 0.984 / 0.980 | 0.987 / 0.987 / 0.982 |

**Key findings:**
- Monolingual models achieve near-perfect in-domain F1 (≥ 0.98) but collapse under zero-shot transfer, driven almost entirely by failure to detect dementia cases (e.g., NeoBERT EN→TL dementia recall = 0.017).
- NeoBERT matches or exceeds BERT in-domain but shows greater cross-lingual degradation under monolingual English training (∆F1 = 0.639 vs. 0.595), indicating stronger specialization to the training distribution.
- Bilingual training eliminates cross-lingual bias across all models (∆F1 < 0.015), with NeoBERT achieving the smallest gap (0.006) and highest combined F1 (0.987).
- TF-IDF + LR shows the most severe cross-lingual collapse (F1 = 0.009 on Filipino when trained on English), confirming that contextual representations are necessary for cross-lingual robustness.

---

## Installation

```bash
# Baseline
pip install -r requirements_baseline.txt

# BERT / NeoBERT (run on Google Colab with GPU recommended)
pip install -r requirements_transformers.txt
```

---

## Usage

Place `english.xlsx` and `tagalog.xlsx` in the same directory as the script, then run:

```bash
# Baseline
python Baseline/tf_idf_and_lr.py

# BERT
python BERT/bert_english.py
python BERT/bert_tagalog.py
python BERT/bert_bilingual.py

# NeoBERT
python NeoBERT/neobert_english.py
python NeoBERT/neobert_tagalog.py
python NeoBERT/neobert_bilingual.py
```

Each transformer script performs hyperparameter grid search, then runs 10-fold cross-validation with the best configuration. Checkpoints are saved after each seed so training can be resumed if interrupted.

---

## Citation

```bibtex
@inproceedings{forgottenwords2025,
  title     = {Forgotten Words: Benchmarking {NeoBERT} for Dementia Detection
               in Low-Resource Conversational {Filipino} and {English} Speech},
  author    = {Anonymous},
  booktitle = {NA},
  year      = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
