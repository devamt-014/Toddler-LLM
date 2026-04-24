# 🍼 Toddler LLM

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
2. **Tokenization** — Words are mapped to integer indices
3. **Training Data** — Sequences are created using a sliding window (input → next word)
4. **Model** — A simple `nn.Module` with an Embedding layer + Linear layer
5. **Training** — 500 epochs using CrossEntropyLoss + Adam optimizer
6. **Inference** — Given a prompt, predicts the most likely next word
7. **Export** — Predictions saved to an Excel file using `openpyxl`

---

## 🗂️ Project Structure

```
toddler-llm/
│
├── toddler_llm.py       # Main script (full pipeline)
├── model_analysis.xlsx  # Output predictions (auto-generated)
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/toddler-llm.git
cd toddler-llm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python toddler_llm.py
```

The script will train the model and print next-word predictions.  
A `model_analysis.xlsx` file will also be generated with all test results.

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
| `i am` | `fine` |
| `hello` | `there` |
| `see you` | `next` |
| `what time` | `is` |

---

## 📍 Roadmap

This is **v1.0 — Toddler stage.** Future versions will grow up:

- [ ] **v2.0** — Add proper `<PAD>` token, fix padding conflict
- [ ] **v2.0** — Replace flat linear layer with an LSTM
- [ ] **v3.0** — Beam search instead of argmax
- [ ] **v3.0** — Larger, real-world dataset
- [ ] **v4.0** — Attention mechanism (baby Transformer 👀)

---

## 💡 What I Learned

- How vocabulary and tokenization work from scratch
- How to build and train a custom `nn.Module` in PyTorch
- How next-word prediction forms the basis of all LLMs
- The full ML pipeline: data → model → train → infer → export

---

## 🙌 Acknowledgements

Built with curiosity, PyTorch, and a lot of `print()` statements.

---

*"Every expert was once a toddler." 🍼*
