import torch
import torch.nn as nn
import random
import yaml
import chatterbot_corpus
import os

# ─────────────────────────────
# LOAD DATASET
# ─────────────────────────────
def load_pairs():
    path = os.path.dirname(chatterbot_corpus.__file__)
    english_path = os.path.join(path, "data", "english")
    
    files = [
        "greetings.yml",
        "conversations.yml",
        "emotion.yml",
        "ai.yml",
        "botprofile.yml",
        "humor.yml",
        "food.yml",
        "health.yml",
        "computers.yml",
        "tech_support.yml"
    ]
    
    pairs = []
    
    for filename in files:
        filepath = os.path.join(english_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        for convo in data["conversations"]:
            for i in range(len(convo) - 1):
                input_s  = str(convo[i]).strip().lower()
                output_s = str(convo[i+1]).strip().lower()
                
                if not input_s or not output_s:
                    continue
                if len(input_s.split()) > 30 or len(output_s.split()) > 30:
                    continue
                    
                pairs.append((input_s, output_s))
    
    print(f"Loaded {len(pairs)} conversation pairs")
    return pairs

pairs = load_pairs()

# ─────────────────────────────
# GPU SETUP
# ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ───────────────────────────── 
# VOCABULARY
# ─────────────────────────────
print("Building vocabulary...")
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

print(f"Vocabulary size: {len(words)}")

# ─────────────────────────────
# MAPPINGS
# ─────────────────────────────
words_to_idx = {word: i for i, word in enumerate(words)}
idx_to_words = {i: word for i, word in enumerate(words)}

# ─────────────────────────────
# ENCODE PAIRS
# ─────────────────────────────
print("Encoding pairs...")
encoded_sequences = []

for input_s, output_s in pairs:
    input_enc  = [words_to_idx.get(w, words_to_idx["<UNK>"])
                  for w in input_s.split()]
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

split = int(0.9 * len(encoded_sequences))
train_sequences = encoded_sequences[:split]
val_sequences   = encoded_sequences[split:]

# ─────────────────────────────
# TRAINING DATA
# ─────────────────────────────
print("Building training data...")
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
print("Padding sequences...")
max_len = max(len(seq) for seq in X_train)


X_train_padded = [[0] * (max_len - len(s[-max_len:])) + s[-max_len:] 
                  for s in X_train]
X_val_padded   = [[0] * (max_len - len(s[-max_len:])) + s[-max_len:] 
                  for s in X_val]

# ─────────────────────────────
# TENSORS → GPU
# ─────────────────────────────
print("Moving tensors to GPU...")
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train,        dtype=torch.long).to(device)

# guard — if val is empty use small slice of train
if len(X_val_padded) == 0:
    X_val_tensor = X_train_tensor[:5]
    y_val_tensor = y_train_tensor[:5]
else:
    X_val_tensor = torch.tensor(X_val_padded, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val,        dtype=torch.long)

# ─────────────────────────────
# BiLSTM MODEL
# ─────────────────────────────
class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.dropout   = nn.Dropout(0.5)
        self.lstm      = nn.LSTM(128, 256, batch_first=True,
                                 bidirectional=True)        # ✅ BiLSTM
        self.fc        = nn.Linear(512, vocab_size)         # ✅ 256*2

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, (hidden, cell) = self.lstm(x)
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)      # ✅ concat
        x = self.fc(x)
        return x

# ─────────────────────────────
# INITIALIZE
# ─────────────────────────────
vocab_size = len(words)
model      = TinyLM(vocab_size).to(device)
loss_fn    = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────
# TRAINING LOOP
# ─────────────────────────────
print("Training started...")

batch_size = 64

for epoch in range(300):
    model.train()
    
    # mini batches
    perm = torch.randperm(X_train_tensor.size(0))
    total_loss = 0
    
    for i in range(0, X_train_tensor.size(0), batch_size):
        idx = perm[i:i+batch_size]
        X_batch = X_train_tensor[idx]
        y_batch = y_train_tensor[idx]
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / (X_train_tensor.size(0) // batch_size)
    
    model.eval()
    with torch.no_grad():
        val_x = X_val_tensor[:256].to(device)
        val_y = y_val_tensor[:256].to(device)
        val_output = model(val_x)
        val_loss   = loss_fn(val_output, val_y)
    
    
    del val_x, val_y
    torch.cuda.empty_cache()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss.item():.4f}")

# ─────────────────────────────
# SAVE
# ─────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "words_to_idx": words_to_idx,
    "idx_to_words": idx_to_words,
    "max_len": max_len,
    "vocab_size": vocab_size
}, "toddler_llm_v4.pth")

print("Model saved! → toddler_llm_v4.pth")
