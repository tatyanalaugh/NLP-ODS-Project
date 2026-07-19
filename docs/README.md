# NLP-ODS-Project

> **📄 Start here:** The complete project description, methodology, experimental setup, and discussion of results are available in **`Project_report.pdf`**. This report serves as the primary documentation for the project, while this README provides an overview of the repository and its contents.

## Project Overview

This project investigates the detection of **drug-related messages** in chatbot conversation logs using both traditional NLP techniques and modern Large Language Models (LLMs).

The project was completed as part of the **Open Data Science (ODS) NLP course** and explores different approaches to binary text classification, including:

- a rule-based keyword matching baseline;
- instruction-tuned open-source LLMs;
- prompt engineering;
- In-Context Learning (ICL);
- semantic retrieval of demonstration examples using embedding models.

The objective is to maximize **recall** while maintaining high overall classification quality, making the solution suitable for content moderation scenarios where missing illegal messages is particularly costly.

---

## Main Results

The project compares several approaches to drug-related message detection.

| Method | Macro F1 | Recall |
|---------|----------|---------|
| Rule-based keyword matching | **0.68** | 0.79 |
| YandexGPT-5-Lite-8B (best prompt) | **0.91** | 0.90 |
| ICL + semantic example retrieval (MMR) | **0.91** | **0.94** |

Key findings:

- Simple keyword matching provides a strong baseline but suffers from limited generalization.
- Prompt engineering significantly improves LLM performance.
- Instruction-tuned LLMs substantially outperform the rule-based approach.
- Retrieving semantically similar examples for In-Context Learning increases recall while preserving high precision.
- The best-performing configuration achieves a **Macro F1 score of 0.91** and a **Recall of 0.94**, making it the most suitable approach for moderation tasks.

Complete evaluation details, confusion matrices, prompt templates, and error analysis can be found in **`Project_report.pdf`**.

---

## Repository Structure

```
NLP-ODS-Project
│
├── data/
│   ├── illegal_terms_dictionary_edit.csv      # Drug-related keyword dictionary
│   └── train.parquet                          # Labeled dataset
│
├── docs/
│   ├── README.md                              # Additional documentation
│   └── requirements.txt                       # Python dependencies
│
├── embeddings_out_qwen3_8b/
│   └── embeddings_out_qwen3_8b/
│       ├── train_embeddings.npy
│       ├── val_embeddings.npy
│       ├── test_embeddings.npy
│       ├── *_labels.npy
│       └── *_ids.npy
│
├── src/
│   ├── dataset processing/
│   │   ├── 2026-04-05_create_labeled_dataset.ipynb
│   │   └── 2026-04-14_split_labeled_dataset.ipynb
│   │
│   ├── rule-based method/
│   │   ├── 2026-04-14_keyword_baseline.ipynb
│   │   └── extending_vocabulary.py
│   │
│   └── LLMs/
│       ├── 2026-05-26_model_api_refactored.ipynb
│       ├── 22_05_26_multi-model testing_fixed_.py
│       └── ICL/
│
└── Project_report.pdf                         # Full project report
```

### Repository Contents

#### `data/`

Contains the processed dataset used for experiments together with the manually curated dictionary of illegal drug-related terms used by the rule-based baseline.

#### `src/dataset processing/`

Notebooks for dataset preparation:

- creating the labeled dataset;
- train/validation/test splitting.

#### `src/rule-based method/`

Implementation of the keyword-based baseline, including dictionary expansion and evaluation.

#### `src/LLMs/`

Experiments with instruction-tuned LLMs, prompt engineering, API-based inference, and model evaluation.

The `ICL` directory contains experiments on **In-Context Learning**, including semantic retrieval of demonstrations using sentence embeddings.

#### `embeddings_out_qwen3_8b/`

Precomputed embeddings generated with **Qwen3-8B** that are used for semantic retrieval during ICL experiments.

---

## Project Workflow

1. Build a labeled dataset from raw chatbot conversations.
2. Split the dataset into train, validation, and test subsets.
3. Train and evaluate a rule-based keyword detector.
4. Evaluate multiple instruction-tuned LLMs using different prompts.
5. Improve performance with In-Context Learning.
6. Retrieve semantically similar examples using embedding-based search.
7. Compare all approaches using standard classification metrics.

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Hugging Face Transformers
- Sentence Transformers
- Qwen3-8B Embeddings
- YandexGPT
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tatyanalaugh/NLP-ODS-Project.git
cd NLP-ODS-Project
```

Install dependencies:

```bash
pip install -r docs/requirements.txt
```

---

## Reproducing the Experiments

The experiments can be reproduced in the following order:

1. Prepare the labeled dataset (`src/dataset processing/`).
2. Run the rule-based baseline.
3. Evaluate the LLM-based approaches.
4. Run the In-Context Learning experiments using the provided embeddings.
5. Compare the results reported in `Project_report.pdf`.

---

## Project Report

The complete description of the project, including motivation, dataset construction, methodology, experiments, prompt design, evaluation metrics, error analysis, and discussion, is available in **`Project_report.pdf`**.

```

### A few suggestions

I would also make a couple of small improvements to the repository itself:

- Rename **`Project_peport.pdf`** → **`Project_report.pdf`** (there's a typo).
- Rename **`dataset processing`** → **`dataset_processing`**.
- Rename **`rule-based method`** → **`rule_based_method`**.
- Rename **`22_05_26_multi-model testing_fixed_.py`** to something cleaner like `multi_model_evaluation.py`.
- Remove the `__MACOSX` directory—it is an artifact created by macOS and should not be in the repository.
- Consider moving `requirements.txt` to the repository root instead of `docs/`, which is the more common GitHub convention.

These changes would make the repository look more polished and consistent with typical open-source NLP research projects.
