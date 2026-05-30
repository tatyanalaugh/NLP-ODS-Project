
# Automatic Detection of Illegal Drug Trade in Chatbot Logs

A binary text classification project developed for a commercial company to automatically detect illegal drug trade content in chatbot dialogues, without exposing sensitive customer data to external services.

## 👥 Team

**Khakhaleva Tatyana** — raw dataset processing, dataset description, existing approaches survey, dictionary enlarging, DuckDB pipeline, LLM estimation pipeline, metrics, report preparation

**Odnoshivkina Victoria** — raw dataset processing, dictionary creation, metrics

**Mamochkina Ekaterina** — raw dataset processing, existing approaches survey, in-context learning, metrics

## 📌 Problem Statement

Social media and messenger platforms have become venues for illegal businesses, including drug trade operated through chatbots mimicking legitimate companies. This project addresses the absence of an established illegal content detection framework for the Russian language domain. All models are deployed locally on a 32 GB GPU to prevent leakage of sensitive customer data.

## 📊 Dataset

- **Source:** commercial chatbot logs (64 raw columns, reduced to 10 after cleaning)
- **Full dataset:** 7,816,579 messages
- **Labelled sample:** 1,392 messages across 216 sessions (929 legal / 463 illegal)
- **Class distribution in full data:** 99.99% legal, 0.01% illegal
- **Language:** predominantly Russian (detected via fasttext-langdetect)
- **Split:** session-level 70/15/15 to prevent data leakage

| Split | Sessions | Messages |
|---|---|---|
| Train | 150 | 748 |
| Validation | 33 | 336 |
| Test | 33 | 305 |

## 🔑 Metrics

Recall on the illegal class is prioritized — missing an illegal message is worse than a false alarm. Primary comparison metric is **Macro F1**.

## 📈 Results

### Rule-based baseline

| Strategy | Accuracy | Macro F1 |
|---|---|---|
| Naive (substring match) | 0.40 | 0.35 |
| Smart (word boundaries + weight regex) | 0.74 | **0.68** ← baseline |

### Reasoning LLMs on test set (selected results)

| Model | Prompt | Macro F1 |
|---|---|---|
| Qwen3.5-9B | prompt_c_with_dict | 0.78 |
| Gemma | prompt_c_with_dict | 0.76 |
| Mistral | prompt_b_no_dict | 0.76 |

### Best result — YandexGPT-5-Lite-8B-instruct (prompt_d, after error analysis)

| Metric | Score |
|---|---|
| Accuracy | 0.92 |
| Precision (illegal) | 0.92 |
| Recall (illegal) | 0.83 |
| F1 (illegal) | 0.88 |
| **Macro F1** | **0.91** |

## 🛠️ Methods

### 1. Rule-based keyword matching
An 840-term lexicon assembled from four sources: Russian Illegal Drugs Reviews Sentiment Dataset (Kaggle), Russian drug addict slang sites, Wiktionary drug slang appendix, and English DEA seed terms from JEDIS. Two matching strategies were implemented: naive substring search and smart whole-word boundary matching with an auxiliary weight-pattern regex (`\d+(?:\.\d+)?\s*[гг]`).

### 2. Instruction-tuned LLMs
Four locally hosted models were evaluated across three prompt variants (with/without dictionary) under reasoning mode:
- Qwen3.5-9B
- Gemma-4-E4B-it
- YandexGPT-5-Lite-8B-instruct
- Ministral-3-8B-Instruct-2512

Key inference parameters: `temperature=0.0`, `max_tokens=20000`, `max_concurrency=32`, `max_retries=5`. Outputs were constrained to a boolean schema `{ has_drug_mention: bool }`.

## 📚 References

1. Cascavilla et al. — Illicit Darkweb Classification via NLP, SECRYPT 2022
2. Li et al. — Machine Learning for Illicit Drug Dealer Detection on Instagram, JMIR 2019
3. Prado-Sánchez et al. — Zero-Shot Classification of Illicit Dark Web Content, Electronics 2025
4. Song et al. — JEDIS: Delexicalized Distant Supervision for Illicit Drug Jargon Detection, KDD 2025
5. Sonawane et al. — AI Detection of Drug Activity in Telegram, 2025
