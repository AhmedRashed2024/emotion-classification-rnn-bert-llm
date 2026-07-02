# Comparative Analysis of Learning Paradigms for Text Classification

### From-Scratch RNNs, Transfer Learning, and Large Language Models

**Course:** ICS603 — Advanced Machine Learning, Spring 2026, German International University — Assignment 3

## Team

| Name | ID |
|---|---|
| Ahmed Ramy | 13002184 |
| Saif Saad | 14004494 |
| Ahmed Amr | 13007323 |

## Overview

Six-class emotion classification on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`), solved three ways to directly measure what pretraining buys you:

1. **From scratch** — a bidirectional LSTM with embeddings learned entirely from the training data.
2. **Transfer learning** — DistilBERT fully fine-tuned with a classification head.
3. **LLM prompting** — Phi-3.5-mini-instruct (4-bit quantized), zero-shot / few-shot / chain-of-thought, no gradient updates at all.

## Dataset

16,000 train / 2,000 validation / 2,000 test examples, tweet-length text. The classes are imbalanced (train split): joy 33.5%, sadness 29.2%, anger 13.5%, fear 12.1%, love 8.2%, **surprise only 3.6%**. Mean length ~19 words (95th percentile 41 words), so short sequence limits lose very little information.

## Preprocessing

Two separate tokenization pipelines, one per paradigm:

| | RNN | Transformer |
|---|---|---|
| Tokenizer | Custom word-level, vocab capped at 15,000 | `distilbert-base-uncased` WordPiece (`AutoTokenizer`) |
| Lowercasing | Yes | Not needed (model is already uncased) |
| Max length | 64 tokens | 128 subword tokens |

Punctuation was deliberately **kept** — exclamation marks and ellipses carry emotional signal that helps separate classes like anger vs. sadness. No stemming/lemmatization, no URL/mention stripping (the dataset is already clean).

## Part 2 — From-Scratch RNN (BiLSTM)

**Architecture:** Embedding → Dropout → 2-layer Bidirectional LSTM (hidden dim 128) → Dropout → Linear(→6). Forward and backward final hidden states from the last layer are concatenated before the classification head. ~1.6M parameters.

**Training:** Adam, lr = 1e-3, 15 epochs, `ReduceLROnPlateau` on validation loss.

**Results:**

| Metric | Value |
|---|---|
| Test accuracy | 91.55% |
| Macro F1 | 0.8708 |
| Training time | 119.4s |
| Inference time (test set) | 0.59s |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| sadness | 0.95 | 0.96 | 0.96 |
| joy | 0.93 | 0.94 | 0.94 |
| love | 0.84 | 0.79 | 0.81 |
| anger | 0.92 | 0.91 | 0.92 |
| fear | 0.92 | 0.83 | 0.87 |
| surprise | 0.64 | 0.85 | 0.73 |

## Part 3 — Transfer Learning: Fine-Tuned DistilBERT

`distilbert-base-uncased` + classification head, **all 66,958,086 parameters** fine-tuned (full fine-tuning, not frozen-backbone), batch size 16, HuggingFace `Trainer` API.

DistilBERT was trained via knowledge distillation from BERT-base — it retains BERT's masked-language-modeling objective plus a cosine embedding loss matching BERT's hidden representations, ending up 40% smaller and 60% faster while keeping ~97% of BERT's performance. This pretrained linguistic knowledge is why fine-tuning only needs to adapt an existing representation rather than learn English from 16,000 examples.

**Results:**

| Metric | Value |
|---|---|
| Test accuracy | 92.50% |
| Macro F1 | 0.8880 |
| Training time | 369.7s |
| Inference time (test set) | 4.385s |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| sadness | 0.95 | 0.98 | 0.96 |
| joy | 0.95 | 0.94 | 0.94 |
| love | 0.82 | 0.83 | 0.82 |
| anger | 0.98 | 0.87 | 0.92 |
| fear | 0.90 | 0.89 | 0.89 |
| surprise | 0.69 | 0.89 | 0.78 |

## Part 4 — LLM Prompting (No Training)

**Model:** `microsoft/Phi-3.5-mini-instruct`, loaded 4-bit (NF4, `bnb_4bit_compute_dtype=float16`) via `bitsandbytes`. **Requires Linux + CUDA** (Kaggle T4) — the notebook auto-detects the environment and skips Part 4 on local Windows machines. Greedy decoding throughout.

| Setting | Accuracy | Macro F1 | Inference time | Unparseable outputs |
|---|---|---|---|---|
| Zero-shot | 51.21% | 0.4170 | 5,282.9s | 96 / 2000 |
| Few-shot (3) | 38.23% | 0.2622 | 5,420.8s | 4 / 2000 |
| Few-shot (8) | 43.72% | 0.3022 | 5,431.2s | 40 / 2000 |
| Chain-of-thought | 31.93% | 0.2770 | 16,702.5s | 816 / 2000 |

Counter-intuitively, adding examples **hurt** accuracy — the model tends to hallucinate continuation text (fabricating new "examples") after its answer, and the label parser sometimes picks up an incorrect label from that continuation. Chain-of-thought performed worst: unconstrained step-by-step reasoning ran the model out of its token budget before it produced a parseable label 816 times out of 2,000.

## Part 5 — Data Efficiency (500-example subset)

A balanced 498-example subset (83 per class) was used to retrain the RNN and re-fine-tune DistilBERT. The LLM needs no retraining, so its accuracy is unchanged.

| Approach | Full data | 500 examples | Drop |
|---|---|---|---|
| RNN (BiLSTM) | 91.55% | 26.45% | −65.1 pts |
| DistilBERT | 92.50% | 63.85% | −28.7 pts |
| LLM (zero-shot) | 51.21% | 51.21% | 0 pts (no training) |

The RNN collapses to barely above the 16.7% random-chance baseline for 6 classes — it must learn embeddings, language structure, and the task simultaneously from very little data. DistilBERT degrades but stays clearly useful, since fine-tuning on 500 examples only has to adapt an already-rich pretrained representation. Notably, DistilBERT at 500 examples (63.9%) still beats the LLM's best prompting result (51.2%) — a small amount of task-specific fine-tuning outperforms prompting alone.

## Part 6 — Comparative Summary

| Approach | Test Accuracy | Macro F1 | Training Time | Inference Time |
|---|---|---|---|---|
| RNN (BiLSTM) | 0.9155 | 0.8708 | 119.4s | 0.590s |
| DistilBERT (fine-tuned) | 0.9250 | 0.8880 | 369.7s | 4.385s |
| LLM zero-shot | 0.5121 | 0.4170 | N/A | 5,282.9s |
| LLM few-shot (3) | 0.3823 | 0.2622 | N/A | 5,420.8s |
| LLM few-shot (8) | 0.4372 | 0.3022 | N/A | 5,431.2s |
| LLM chain-of-thought | 0.3193 | 0.2770 | N/A | 16,702.5s |

**Failure modes:** all three models most often confuse **love ↔ joy** (both positive, similar vocabulary) and struggle hardest on **surprise**, the rarest class (66 test examples) — though DistilBERT's pretrained semantics handle it noticeably better than the RNN. The LLM's errors are less about understanding emotion and more about controlling output *format* without any fine-tuning.

**Compute profile:** the LLM trades zero training cost for inference that is roughly 1,200–3,800× slower than DistilBERT per example — classifying the 2,000-example test set took over 88 minutes per zero/few-shot setting and 4.6+ hours for chain-of-thought, which would be prohibitively expensive at production scale.

**When to choose each, in practice:**
- **From-scratch RNN** — abundant labeled data (thousands+ per class), need a lightweight model for edge/mobile deployment, or no pretrained model exists for the domain/language. Fails badly with limited data.
- **Fine-tuned transformer (DistilBERT)** — the default choice for most text classification: best accuracy-to-effort ratio, works even with moderate labeled data, a few minutes of fine-tuning.
- **LLM prompting** — zero labeled data, rapid prototyping with no training infrastructure, or a task definition that changes often (just edit the prompt). Costs more at inference, is less accurate on fine-grained classification, and is sensitive to prompt design and output parsing.

## Repository Contents

- `aml_project_3.ipynb` — full pipeline: data exploration, dual tokenization, BiLSTM baseline, DistilBERT fine-tuning, Phi-3.5-mini prompting (zero/few-shot/CoT), data-efficiency experiment, and final comparative analysis.

## Requirements

```
datasets transformers accelerate bitsandbytes
torch scikit-learn pandas matplotlib seaborn
```

> Part 4 (LLM prompting) requires a Linux environment with a CUDA GPU (`bitsandbytes` 4-bit quantization is not supported on Windows). The notebook detects this automatically and skips that section outside Kaggle/Linux.

