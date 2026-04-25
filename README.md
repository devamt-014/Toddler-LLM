# 🍼 Toddler LLM v2.0

A tiny next-word predictor built completely from scratch using Python and PyTorch.  
No libraries. No pretrained models. Just raw learning.

---

## 🧠 What Is This?

**Toddler LLM** is a minimal language model that predicts the next word in a sentence.  
It was built as a first step into understanding how large language models work under the hood —  
starting from vocabulary creation, tokenization, model design, training, all the way to inference.

Think of it as a baby version of GPT. It doesn't know much, but it built itself from nothing. 🐣

---

## ⚙️ How It Works

1. **Vocabulary** — All unique words from the training sentences are extracted and sorted
2. **Tokenization** — Words are mapped to integer indices with proper `<PAD>` and `<UNK>` tokens
3. **Training Data** — Sequences are created using a sliding window (input → next word)
4. **Model** — A custom `nn.Module` with Embedding + Dropout + LSTM + Linear layer
5. **Training** — 150 epochs using CrossEntropyLoss + Adam optimizer with train/val split
6. **Inference** — Given a prompt, predicts the most likely next word
7. **Export** — Predictions saved to an Excel file using `openpyxl`

---

## 🆕 What's New in v2.0

- ✅ Replaced flat Linear layer with **LSTM** — model now has sequence memory
- ✅ Added proper **`<PAD>` token** at index 0 — no more fake data during padding
- ✅ Added **Dropout (0.5)** — reduces overfitting
- ✅ Added **train/val split (90/10)** — now tracks generalization vs memorization
- ✅ Fixed **`optimizer.zero_grad()` order** — correct gradient clearing
- ✅ Added **input length guard** in `predict_next()` — no more silent crashes
- ✅ Added **save/load model weights** — no retraining every run
- ✅ Expanded dataset to **~100 sentences** — emotions, weather, affirmations and more
- ✅ Cleaner prediction output format — `input : predicted`

---

## 🗂️ Project Structure

```
TODDLER-LLM/
│
├── tod.py                    # Main script (full pipeline)
├── toddler_llm_v2.pth        # Saved model weights (auto-generated)
├── model_analysis.xlsx       # Output predictions (auto-generated)
├── requirements.txt          # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/devamt-014/Toddler-LLM.git
cd TODDLER-LLM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python tod.py
```

The script will train the model and print next-word predictions.  
A `toddler_llm_v2.pth` and `model_analysis.xlsx` will also be generated.

---

## 📦 Requirements

```
torch
openpyxl
```

---

## 🧪 Example Predictions

| Input | Predicted Next Word |
|-------|-------------------|
| `how are` | `you` |
| `i feel` | `lonely` |
| `hi` | `there` |
| `see you next` | `time` |
| `what time is` | `it` |
| `time flies` | `fast` |
| `of` | `course` |
| `i dont` | `know` |
| `i think` | `so` |
| `doing` | `great` |

---

## 📊 v1 vs v2

| | v1 | v2 |
|--|--|--|
| Architecture | Linear | LSTM |
| Val Loss | ~6.87 | ~2.71 |
| Dropout | ❌ | ✅ |
| Train/Val Split | ❌ | ✅ |
| PAD token | ❌ | ✅ |
| Save/Load weights | ❌ | ✅ |
| Good predictions | ~40% | ~80% |

---

## 📍 Roadmap

- [x] **v1.0** — Vocabulary, tokenization, Linear model, basic training
- [x] **v2.0** — LSTM, dropout, train/val split, PAD token, save/load
- [ ] **v3.0** — Full chatbot with `generate_response()` and `<EOS>` token
- [ ] **v3.0** — Beam search instead of argmax
- [ ] **v4.0** — Attention mechanism (baby Transformer 👀)

---

## 💡 What I Learned

- How vocabulary and tokenization work from scratch
- What `<PAD>` and `<UNK>` tokens are and why they matter
- How LSTM works — hidden state, forget/input/output gates
- How train/val split catches overfitting
- How dropout forces generalization
- The full ML pipeline: data → model → train → validate → infer → export

---

## 🙌 Acknowledgements

Built with curiosity, PyTorch, and a lot of `print()` statements.

---

*"Every expert was once a toddler." 🍼*
