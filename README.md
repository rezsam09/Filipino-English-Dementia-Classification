# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech

Official implementation of the paper accepted at **BioNLP @ ACL 2026** — 25th Workshop on Biomedical Language Processing, July 3–4, 2026, San Diego, CA.

> **Paper:**

---

## Overview

Dementia detection from spontaneous speech offers a scalable path to cognitive screening, yet NLP research has remained predominantly English-centric. This work addresses that gap for the Philippines, where Filipino–English code-switching is pervasive in everyday and clinical speech, and no prior NLP-based dementia detection work has existed.

We present the first systematic evaluation of transformer-based dementia detection in Filipino speech, the first empirical assessment of NeoBERT in any clinical NLP setting, and a controlled parallel bilingual dataset of 4,000 DementiaBank-derived transcripts. Five model families are benchmarked under monolingual, zero-shot cross-lingual, and bilingual fine-tuning settings.

**Core finding:** Bilingual fine-tuning eliminates cross-lingual degradation across all transformer architectures, converging to Macro-F1 = 0.969–0.973. Multilingual clinical NLP performance is driven primarily by linguistic coverage during training, not model scale or architectural design.

---

## Results

| Model | Train | EN F1 | TL F1 | Combined F1 | ΔF1 Gap |
|---|---|---|---|---|---|
| TF-IDF + LR | EN | 0.930 | 0.649 | 0.836 | 0.281 |
| TF-IDF + LR | TL | 0.640 | 0.971 | 0.825 | 0.331 |
| TF-IDF + LR | EN+TL | 0.932 | 0.976 | 0.954 | 0.044 |
| BERT | EN | 0.952 | 0.455 | 0.744 | 0.497 |
| BERT | TL | 0.705 | 0.981 | 0.855 | 0.276 |
| BERT | EN+TL | 0.954 | 0.984 | **0.969** | 0.030 |
| XLM-RoBERTa | EN | 0.948 | 0.936 | 0.942 | 0.013 |
| XLM-RoBERTa | TL | 0.825 | 0.986 | 0.905 | 0.161 |
| XLM-RoBERTa | EN+TL | 0.953 | 0.990 | **0.972** | 0.037 |
| RoBERTa-Tagalog | EN | 0.951 | 0.934 | 0.942 | 0.017 |
| RoBERTa-Tagalog | TL | 0.769 | 0.987 | 0.882 | 0.218 |
| RoBERTa-Tagalog | EN+TL | 0.958 | 0.988 | **0.973** | 0.030 |
| NeoBERT | EN | 0.952 | 0.617 | 0.802 | 0.335 |
| NeoBERT | TL | 0.757 | 0.979 | 0.878 | 0.222 |
| NeoBERT | EN+TL | 0.956 | 0.983 | **0.970** | 0.027 |

Macro-F1 reported. ΔF1 = |F1_in-domain − F1_cross-lingual|; bilingual combined is excluded from gap computation.

---

## Repository Structure

```
Filipino-English-Dementia-Classification/
├── BERT/                          # BERT-base-uncased fine-tuning scripts
├── NeoBERT/                       # NeoBERT fine-tuning scripts
├── Baseline/                      # TF-IDF + Logistic Regression baseline
├── requirements_baseline.txt      # Baseline dependencies
├── requirements_transformers.txt  # Transformer model dependencies
└── README.md
```

XLM-RoBERTa and RoBERTa-Tagalog share the same fine-tuning structure as the BERT directory.

---

## Dataset

All transcripts originate from [DementiaBank](https://dementia.talkbank.org/) (Becker et al., 1994).

- **4,000 transcripts total** — 2,000 English, 2,000 Filipino; 1,000 samples per class per language
- English samples are drawn directly from DementiaBank's Cookie Theft picture description task
- Filipino samples are **manually translated** from the same English transcripts, with instructions to preserve discourse-level markers of cognitive decline: repetitions, hesitations, false starts, and syntactic degradation
- Machine translation was deliberately avoided, neural MT normalizes disfluent speech toward fluent output, erasing the features that distinguish dementia from healthy speech
- All samples are anonymized in accordance with standard clinical data handling protocols

> ⚠️ **DementiaBank requires a data use agreement.** The dataset is not redistributed here. Apply for access at [https://dementia.talkbank.org/](https://dementia.talkbank.org/).

---

## Models

| Model | Pretraining | HuggingFace ID |
|---|---|---|
| BERT-base-uncased | English | `bert-base-uncased` |
| NeoBERT | English — RefinedWeb, 600B tokens | `lolobroken/neobert` |
| XLM-RoBERTa | 100 languages | `xlm-roberta-base` |
| RoBERTa-Tagalog | Filipino — TLUnified corpus | `jcblaise/roberta-tagalog-base` |

---

## Installation

```bash
# Transformer models
pip install -r requirements_transformers.txt

# TF-IDF + Logistic Regression baseline
pip install -r requirements_baseline.txt
```

---

## Usage

Each model is trained under three configurations:

- **EN** — English-only training and evaluation
- **TL** — Filipino-only training and evaluation
- **EN+TL** — Bilingual training, evaluated on both languages

**Baseline**

```bash
cd Baseline
python baseline.py --lang EN
python baseline.py --lang TL
python baseline.py --lang EN+TL
```

**Transformer models**

```bash
cd BERT        # or NeoBERT
python train.py --lang EN
python train.py --lang TL
python train.py --lang EN+TL
```

---

## Experimental Setup

- **Data split:** Stratified 70–15–15 train / validation / test
- **Evaluation:** Stratified 10-fold cross-validation; results reported as mean ± std across folds
- **Pooling:** Attention-masked mean pooling over final hidden states (not CLS token)
- **Optimizer:** AdamW with linear warmup and decay; gradient clipping = 1.0
- **Max sequence length:** 128 tokens
- **Primary metric:** Macro-F1 with equal class weighting; Dementia-class recall reported separately

### Hyperparameters

| Model | Regime | Batch Size | Learning Rate | Weight Decay |
|---|---|---|---|---|
| BERT | English | 8 | 2×10⁻⁵ | 1×10⁻² |
| BERT | Filipino | 4 | 3×10⁻⁵ | 1×10⁻⁵ |
| BERT | Bilingual | 4 | 3×10⁻⁵ | 1×10⁻² |
| XLM-RoBERTa | English | 8 | 1×10⁻⁵ | 1×10⁻⁵ |
| XLM-RoBERTa | Filipino | 8 | 3×10⁻⁵ | 1×10⁻² |
| XLM-RoBERTa | Bilingual | 8 | 2×10⁻⁵ | 1×10⁻⁵ |
| RoBERTa-Tagalog | English | 8 | 3×10⁻⁵ | 1×10⁻² |
| RoBERTa-Tagalog | Filipino | 4 | 3×10⁻⁵ | 1×10⁻² |
| RoBERTa-Tagalog | Bilingual | 8 | 3×10⁻⁵ | 1×10⁻⁵ |
| NeoBERT | English | 4 | 2×10⁻⁵ | 1×10⁻⁵ |
| NeoBERT | Filipino | 4 | 3×10⁻⁵ | 1×10⁻⁵ |
| NeoBERT | Bilingual | 4 | 3×10⁻⁵ | 1×10⁻⁵ |

---

## Key Findings

**Cross-lingual transfer fails without bilingual training.** English-trained BERT drops from F1 = 0.952 to 0.455 on Filipino. Filipino-trained BERT drops from 0.981 to 0.705 on English. This degradation reflects representational misalignment from language-specific pretraining, not domain shift, the Filipino corpus was constructed from the same source transcripts under controlled conditions.

**Architectural modernization alone does not help.** NeoBERT matches BERT in monolingual performance but exhibits the highest cross-lingual variance of any model (σ = 0.109 on Filipino), suggesting that architectural improvements tighten English-side decision boundaries in ways that reduce tolerance to language shift.

**Multilingual and language-matched models transfer more stably.** XLM-RoBERTa achieves ΔF1 = 0.013 from English to Filipino, the smallest transfer gap under monolingual training. RoBERTa-Tagalog achieves nearly identical English-to-Filipino transfer (ΔF1 = 0.017) despite no explicit multilingual pretraining, likely because conversational Filipino contains extensive English lexical borrowing and code-switching.

**Bilingual fine-tuning resolves cross-lingual gaps across all architectures.** Combined Macro-F1 converges to 0.969–0.973, with dementia recall above 0.93 and consistently low variance across all transformer models. Linguistic coverage during task training is more influential than architecture or scale.

---

## Limitations

The Filipino dataset was produced via controlled manual translation rather than native clinical collection, which means it reflects the conversational structure and semantic content of English source documents. No large-scale native Filipino clinical dementia corpus currently exists. The dataset's scale of 4,000 samples also contributes to cross-validation variance. Future work should involve organically collected speech from Filipino patient cohorts in collaboration with local geriatricians.

This study is text-only by design, to isolate discourse-level language effects. Acoustic features, pause duration, pitch variance, phonation rate, provide independent diagnostic markers not captured here. Integrating speech encoders such as Wav2Vec 2.0 with transformer text models is a natural next step.

Model decision mechanisms in multilingual contexts remain opaque. Clinical deployment requires interpretability, and local feature attribution methods are a necessary step before these workflows can achieve clinical trust.

---

## Citation

```bibtex
@inproceedings{floresca-etal-2026-forgotten,
    title     = "Forgotten Words: Benchmarking {N}eo{BERT} for Dementia Detection
                 in Low-Resource Conversational {F}ilipino and {E}nglish Speech",
    author    = "Floresca, Rez Samantha Z.  and
                 Hao, Edric Castel C.  and
                 Bu{\~n}ales, Hannah Grachiella  and
                 Temprosa, Chelsea Dominique E.  and
                 Reyes, Georgianna Z.  and
                 Chua, Kervin Gabriel L.",
    booktitle = "Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)",
    month     = july,
    year      = "2026",
    address   = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
}
```

---

## License

Released under the [MIT License](LICENSE). The DementiaBank dataset is subject to its own data use agreement and is not redistributed here.
