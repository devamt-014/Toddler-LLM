# 🍼 Toddler LLM v4.0

A tiny BiLSTM chatbot built completely from scratch using Python and PyTorch.  
No pretrained models. No shortcuts. Just raw learning.

---

## 🧠 What Is This?

**Toddler LLM** is a minimal language model that started as a next-word predictor and has grown into a BiLSTM powered chatbot.  
Built version by version to deeply understand how large language models work under the hood —  
from vocabulary creation, tokenization, model design, training, all the way to live conversation.

Think of it as a baby version of GPT. It doesn't know much, but it built itself from nothing. 🐣

---

## 🆕 What's New in v4.0

- ✅ **BiLSTM** — reads sentences both forward and backward for richer context
- ✅ **Hidden state concatenation** — forward + backward states combined (512 output)
- ✅ **ChatterBot Corpus** — 1600 real human conversation pairs
- ✅ **Mini batch training** — proper batched training (batch_size=32)
- ✅ **Separated train/chat** — `tod.py` trains, `run.py` chats
- ✅ **Self contained checkpoint** — vocab + model saved together in `.pth`
- ✅ **No retraining** — `run.py` loads and chats instantly

---

## ⚙️ How It Works

1. **Dataset** — ChatterBot English corpus loaded from greetings, emotions, conversations, food, health, humor
2. **Vocabulary** — All unique words extracted with `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
3. **Encoding** — Pairs encoded as full sequences with sliding window
4. **Model** — Embedding (128) + Dropout (0.5) + BiLSTM (256×2) + Linear (512)
5. **Training** — Mini batch training, CrossEntropyLoss + Adam, GPU
6. **Generation** — Beam search generates word by word until `<EOS>`
7. **Chat** — Live chat loop via `run.py`

---

## 🗂️ Project Structure

```
TODDLER-LLM/
│
├── tod.py                    # Training + model + saving
├── run.py                    # Load model + live chat
├── toddler_llm_v4.pth        # Saved model + vocab (auto-generated)
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

### 3. Train the model (first time only)
```bash
python tod.py
```

### 4. Chat (every time after)
```bash
python run.py
```

---

## 📦 Requirements

```
torch
chatterbot-corpus
pyyaml
```

---

## 💬 Example Chat

```
You: hello
Toddler: greetings!

You: hi there
Toddler: how are you doing?

You: i am good
Toddler: i'm doing well.

You: i am sad
Toddler: why do you feel that way?

You: what is good health ?
Toddler: your asking the wrong guy, however i always wanted to try a burger!

You: bye
Toddler: goodbye take care !
```

---

## 📊 Version Comparison

| | v1 | v2 | v3 | v4 |
|--|--|--|--|--|
| Architecture | Linear | LSTM | LSTM | BiLSTM |
| Val Loss | ~6.87 | ~2.71 | ~0.35 | ~1.37 |
| Output | one word | one word | full sentence | full sentence |
| Interaction | static | static | live chat | live chat |
| Dataset | custom | custom | custom pairs | ChatterBot corpus |
| Direction | → | → | → | ← → |
| Train file | single | single | single | tod.py |
| Chat file | single | single | single | run.py |
| GPU | ❌ | ❌ | ✅ | ✅ |

---

## 📍 Roadmap

- [x] **v1.0** — Vocabulary, tokenization, Linear model, basic training
- [x] **v2.0** — LSTM, dropout, train/val split, PAD token, save/load
- [x] **v3.0** — Live chatbot, SOS/EOS tokens, GPU, generate_response()
- [x] **v3.1** — Beam search instead of argmax
- [x] **v4.0** — BiLSTM, real corpus data, train/chat separation
- [ ] **v5.0** — Streamlit UI + Hugging Face deployment 🌍
- [ ] **v6.0** — Custom tokenizer, subword tokenization
- [ ] **v7.0** — Attention mechanism
- [ ] **v8.0** — Baby Transformer
- [ ] **v9.0** — Fine tune on real data
- [ ] **v10.0** — Full grown LLM 👨
- [ ] **v11.0** — Agent + tools
- [ ] **v14.0** — RAG + production 🚀

---

## 💡 What I Learned

- How BiLSTM reads sequences in both directions
- Why bidirectional context improves understanding
- How to concatenate forward and backward hidden states
- Mini batch training for memory efficiency
- How to separate training and inference into different files
- How to bundle vocab + model into a single checkpoint
- Why small models give philosophical answers to deep questions 😄

---

## 🔬 Research Background

This project is backed by real NLP research experience.  
The author has published work on **sarcasm detection using BiLSTM** achieving:
- **78% accuracy**
- **0.85 F1-Score**
- Outperforming DistilBERT baseline

The same BiLSTM concepts applied in that research are implemented here from scratch. 🔥

---

## 🙌 Acknowledgements

Built with curiosity, PyTorch, ChatterBot corpus, and a GTX 1650 that really didn't want to cooperate. 😄

---

*"Every expert was once a toddler." 🍼*
