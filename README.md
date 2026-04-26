# 🍼 Toddler LLM v3.0

A tiny chatbot built completely from scratch using Python and PyTorch.  
No libraries. No pretrained models. Just raw learning.

---

## 🧠 What Is This?

**Toddler LLM** is a minimal language model that started as a next-word predictor and has now grown into a real chatbot.  
It was built as a step-by-step journey into understanding how large language models work under the hood —  
from vocabulary creation, tokenization, model design, training, all the way to live conversation.

Think of it as a baby version of GPT. It doesn't know much, but it built itself from nothing. 🐣

---

## 🆕 What's New in v3.0

- ✅ **Live chatbot** — real conversation instead of static predictions
- ✅ **`generate_response()`** — generates full sentences word by word
- ✅ **`<SOS>` and `<EOS>` tokens** — model knows when to start and stop
- ✅ **GPU support** — runs on CUDA (GTX 1650 tested)
- ✅ **External dataset** — moved to `datasets.py` as input/output pairs
- ✅ **`clean_input()`** — handles contractions and real world typing
- ✅ **Realistic data section** — handles casual responses like "cool", "lol", "wow"
- ✅ **Special token leak fix** — `<SOS>`, `<PAD>`, `<UNK>` filtered from responses
- ✅ **500 epochs** — best training run yet

---

## ⚙️ How It Works

1. **Vocabulary** — All unique words from pairs extracted with `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
2. **Tokenization** — Words mapped to integer indices
3. **Dataset** — Input/output conversation pairs stored in `datasets.py`
4. **Training Data** — Pairs encoded as full sequences with sliding window
5. **Model** — Embedding (128) + Dropout (0.5) + LSTM (256) + Linear
6. **Training** — 500 epochs, CrossEntropyLoss + Adam, train/val split, GPU
7. **Generation** — `generate_response()` predicts word by word until `<EOS>`
8. **Chat** — Live chat loop in terminal

---

## 🗂️ Project Structure

```
TODDLER-LLM/
│
├── tod.py                    # Main model + training + chat loop
├── datasets.py               # Conversation pairs dataset
├── toddler_llm_v3.pth        # Saved model weights (auto-generated)
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

The model will train and launch a live chat session in your terminal.

---

## 📦 Requirements

```
torch
openpyxl
```

---

## 💬 Example Chat

```
You: good morning !
Toddler: good morning ! how are you today ?

You: i am feeling sad
Toddler: i am sorry hope things get better !

You: what is your name ?
Toddler: i am toddler llm nice to meet you !

You: what can you do ?
Toddler: i can chat with you and predict next words !

You: cool !
Toddler: glad you think so !

You: bye
Toddler: goodbye take care !
```

---

## 📊 Version Comparison

| | v1 | v2 | v3 |
|--|--|--|--|
| Architecture | Linear | LSTM | LSTM + GPU |
| Val Loss | ~6.87 | ~2.71 | ~0.35 |
| Output | one word | one word | full sentence |
| Interaction | static | static | live chat |
| Dataset | sentences | sentences | conversation pairs |
| Tokens | PAD, UNK | PAD, UNK | PAD, UNK, SOS, EOS |
| GPU | ❌ | ❌ | ✅ |

---

## 📍 Roadmap

- [x] **v1.0** — Vocabulary, tokenization, Linear model, basic training
- [x] **v2.0** — LSTM, dropout, train/val split, PAD token, save/load
- [x] **v3.0** — Live chatbot, SOS/EOS tokens, GPU, generate_response()
- [ ] **v3.1** — Beam search instead of argmax
- [ ] **v4.0** — BiLSTM (research backed 🔥)
- [ ] **v5.0** — Streamlit UI + Hugging Face deployment 🌍
- [ ] **v6.0** — Custom tokenizer, subword tokenization
- [ ] **v7.0** — Attention mechanism
- [ ] **v8.0** — Baby Transformer
- [ ] **v9.0** — Fine tune on real data
- [ ] **v10.0** — Full grown LLM 👨

---

## 💡 What I Learned

- How `<SOS>` and `<EOS>` tokens control generation
- How `generate_response()` builds sentences word by word
- How to move tensors and models to GPU with CUDA
- How conversation pairs differ from single sentence training
- How `clean_input()` handles real world typing
- The difference between next word prediction and full response generation

---

## 🙌 Acknowledgements

Built with curiosity, PyTorch, and a lot of `print()` statements.  
Tested by real people who asked it about relationships and got coding advice. 😄

---

*"Every expert was once a toddler." 🍼*
