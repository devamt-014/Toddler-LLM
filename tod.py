import torch
import torch.nn as nn
import random
from datasets import pairs

# ─────────────────────────────
# GPU SETUP
# ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────
# VOCABULARY
# ─────────────────────────────
words = set()

for input_s, output_s in pairs:
    for word in input_s.split():
        words.add(word)
    for word in output_s.split():
        words.add(word)

words.add("<PAD>")
words.add("<UNK>")
words.add("<SOS>")
words.add("<EOS>")

words = sorted(list(words))
words.remove("<PAD>")
words = ["<PAD>"] + words

# ─────────────────────────────
# MAPPINGS
# ─────────────────────────────
words_to_idx = {word: i for i, word in enumerate(words)}
idx_to_words = {i: word for i, word in enumerate(words)}

# ─────────────────────────────
# ENCODE PAIRS
# ─────────────────────────────
# Each pair becomes one full sequence:
# "how are you ? <SOS> i am fine <EOS>"
encoded_sequences = []

for input_s, output_s in pairs:
    input_enc  = [words_to_idx[w] for w in input_s.split()]
    output_enc = [words_to_idx.get(w, words_to_idx["<UNK>"]) 
                  for w in output_s.split()]
    
    sos = words_to_idx["<SOS>"]
    eos = words_to_idx["<EOS>"]
    
    full_sequence = input_enc + [sos] + output_enc + [eos]
    encoded_sequences.append(full_sequence)

# ─────────────────────────────
# TRAIN/VAL SPLIT
# ─────────────────────────────
random.seed(42)
random.shuffle(encoded_sequences)

split = int(1.0 * len(encoded_sequences))
train_sequences = encoded_sequences[:split]
val_sequences   = encoded_sequences[split:]

# ─────────────────────────────
# TRAINING DATA
# ─────────────────────────────
X_train, y_train = [], []
for seq in train_sequences:
    for i in range(1, len(seq)):
        X_train.append(seq[:i])
        y_train.append(seq[i])

X_val, y_val = [], []
for seq in val_sequences:
    for i in range(1, len(seq)):
        X_val.append(seq[:i])
        y_val.append(seq[i])

# ─────────────────────────────
# PADDING
# ─────────────────────────────
max_len = max(len(seq) for seq in X_train)

X_train_padded = [[0] * (max_len - len(s)) + s for s in X_train]
X_val_padded   = [[0] * (max_len - len(s)) + s for s in X_val]

# ─────────────────────────────
# TENSORS → GPU
# ─────────────────────────────
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

if len(X_val_padded) == 0:
        X_val_tensor = X_train_tensor[:5]
        y_val_tensor = y_train_tensor[:5]
else:
    X_val_tensor   = torch.tensor(X_val_padded, dtype=torch.long).to(device)
    y_val_tensor   = torch.tensor(y_val, dtype=torch.long).to(device)

# ─────────────────────────────
# MODEL
# ─────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.dropout   = nn.Dropout(0.5)
        self.lstm      = nn.LSTM(128, 256, batch_first=True)
        self.fc        = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        return x

# ─────────────────────────────
# INITIALIZE
# ─────────────────────────────
vocab_size = len(words)
model      = TinyLM(vocab_size).to(device)
loss_fn    = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────
# TRAINING LOOP
# ─────────────────────────────
for epoch in range(350):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss   = loss_fn(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss   = loss_fn(val_output, y_val_tensor)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

torch.save(model.state_dict(), "toddler_llm_v3.pth")
print("Model saved!")

# ─────────────────────────────
# GENERATE RESPONSE
# ─────────────────────────────
def generate_response(text, max_response_len=10):
    model.eval()
    
    tokens = [words_to_idx.get(w, words_to_idx["<UNK>"]) 
              for w in text.split()]
    tokens += [words_to_idx["<SOS>"]]
    
    response = []

    for _ in range(max_response_len):
        padded = tokens[-max_len:]
        padded = [0] * (max_len - len(padded)) + padded
        x = torch.tensor([padded]).to(device)

        with torch.no_grad():
            output    = model(x)
            predicted = torch.argmax(output, dim=1).item()

        word = idx_to_words[predicted]

        if word in ["<EOS>", "<SOS>", "<PAD>", "<UNK>"]:
            break

        response.append(word)
        tokens.append(predicted)

    return " ".join(response) if response else "i am not that capable enough yet to understand what are you saying sorry !"



# ─────────────────────────────
# INPUT CLEANER
# ─────────────────────────────
def clean_input(text):
    text = text.replace("i'm", "i am")
    text = text.replace("don't", "do not")
    text = text.replace("can't", "can not")
    text = text.replace("won't", "will not")
    text = text.replace("it's", "it is")
    text = text.replace("what's", "what is")
    text = text.replace("i've", "i have")
    text = text.replace("i'd", "i would")
    text = text.replace("i'll", "i will")
    text = text.replace("you're", "you are")
    text = text.replace("that's", "that is")
    text = text.replace("morning", "good morning")
    text = text.replace("afternoon", "good afternoon")
    text = text.replace("evening", "good evening")
    return text

# ─────────────────────────────
# LIVE CHAT
# ─────────────────────────────
print("\nToddler LLM v3.0 — Let's chat ! 🍼")
print("Type 'bye' to exit\n")

while True:
    user_input = clean_input(input("You: ").strip().lower())
    
    if user_input == "bye":
        print("Toddler: goodbye take care !")
        break
    
    if user_input == "":
        continue
    
    response = generate_response(user_input)
    print(f"Toddler: {response}\n")
