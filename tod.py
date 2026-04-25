import torch

#DATA INPUTS
sentences = [

    # 🔹 Basic questions
    "how are you ?",
    "how are you doing ?",
    "how are things ?",
    "what is your name ?",
    "where are you from ?",
    "how old are you ?",
    "what do you do ?",
    "can you help me ?",
    "do you understand ?",

    # 🔹 Responses
    "i am good",
    "i am fine",
    "i am better",
    "i am great",
    "i am great !",
    "i am feeling good today",
    "i am feeling great today",
    "i am happy",
    "i am sad",
    "i am tired",
    "i am bored",
    "i am excited",
    "i am angry",
    "i feel lonely",
    "i am great today!",
    "i am doing great !",

    # 🔹 Greetings
    "hello",
    "hello you",
    "hello you !",
    "hello there",
    "hello friend",
    "hi there",
    "hi buddy",
    "hi there friend",
    "hi there buddy",
    "hi how are you",

    # 🔹 Goodbyes
    "goodbye",
    "bye",
    "ok bye",
    "see you next time",
    "see you later",
    "bye bye",
    "goodbye friend",
    "ok goodbye",
    "bye for now",
    "bye take care",
    "ok bye bye",

    # 🔹 Time questions
    "what time is it ?",
    "what is the time ?",

    # 🔹 Time answers
    "it is morning",
    "it is evening",
    "it is night",

    # 🔹 Time variations
    "time to go out",
    "time flies fast",
    "time doesnt wait",
    "time doesnt wait for anyone",

    # 🔹 Weather (NEW)
    "it is raining",
    "it is sunny",
    "it is cold today",
    "it is hot outside",
    "the weather is nice",

    # 🔹 Short affirmations (NEW)
    "yes",
    "no",
    "maybe",
    "sure",
    "of course",
    "i think so",
    "i dont know",

    # 🔹 Short casual phrases
    "whats up",
    "not much",
    "all good",
    "doing great",

    # 🔹 Multi-sentence flows
    "how are you ? i am fine",
    "how are you ? i am great",
    "how are you doing ? i am good",

    # 🔹 Chat-style
    "user: how are you ?",
    "assistant: i am fine",

    "user: what time is it ?",
    "assistant: it is night",

    "user: hello",
    "assistant: hello there",

    "user: hi",
    "assistant: hi there",

    "user: what is your name ?",
    "assistant: i am toddler llm",

    "user: can you help me ?",
    "assistant: yes i can",

    "user: are you smart ?",
    "assistant: i am learning",

    "user: how are you feeling ?",
    "assistant: i am feeling great today",

    "user: what is the weather ?",
    "assistant: it is sunny",

    # 🔹 Noise / real-world typing
    "how r you",
    "im good",
    "im fine",
    "ok thanks",

    # 🔹 Edge / ambiguity
    "what ?",
    
    # 🔹 Emotion follow ups
    "i am sad today",
    "i am sad sometimes",
    "i am tired today",
    "i am tired now",
    "i am bored today",
    "i am bored now",
    "i am angry today",
    "i am angry now",
    "i am happy today",
    "i am happy now",
    "i am excited today",
    "i am excited now",
    
    # 🔹 Advanced Emotion follow ups
    "i am sad today",
    "i am sad always",
    "i am tired today",
    "i am bored today",
    "i am so bored",
    "i am a bit angry now",
    "i am so happy now",
    "i am really excited today",
    "i am really happy today",
    "i am really sad today"
]


#SPLITTING UNIQUELY ALL WORDS 
words = set()

for sentence in sentences:
    for word in sentence.split():
        words.add(word) 

words.add("<UNK>")
words.add("<PAD>")
words = sorted(list(words))
words.remove("<PAD>")
words = ["<PAD>"] + words

#MAPPINGS
words_to_idx = {word: i for i, word in enumerate(words)}
idx_to_words = {i: word for i, word in enumerate(words)}


#SENTENCES TO NUMBERS 
encoded_sentences = []

for sentence in sentences:
    encoded = [words_to_idx[word] for word in sentence.split()]
    encoded_sentences.append(encoded)
    
    
    
#TRAIN/VAL SPLIT 
import random
random.seed(42)
random.shuffle(encoded_sentences)
split = int(0.9 * len(encoded_sentences))
train_sentences = encoded_sentences[:split]
val_sentences = encoded_sentences[split:]

#TAINING DATA
X_train, y_train = [], []
for sentence in train_sentences:
    for i in range(1, len(sentence)):
        X_train.append(sentence[:i])
        y_train.append(sentence[i])

X_val, y_val = [], []
for sentence in val_sentences:
    for i in range(1, len(sentence)):
        X_val.append(sentence[:i])
        y_val.append(sentence[i])
        
        
#PADDING
max_len = max(len(seq) for seq in X_train)

X_train_padded = [[0] * (max_len - len(s)) + s for s in X_train]
X_val_padded = [[0] * (max_len - len(s)) + s for s in X_val]

# TENSORS
X_train_tensor = torch.tensor(X_train_padded)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val_padded)
y_val_tensor = torch.tensor(y_val)


#OWN TINY LLM MODEL (BRAIN)
import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(64,128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)
        
    def forward(self,x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        return x


#INITALIZING THE MODEL    
vocab_size = len(words)

model = TinyLM(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(150):
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = loss_fn(val_output, y_val_tensor)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

torch.save(model.state_dict(), "toddler_llm_v2.pth")
print("Model saved!")
        
        
def predict_next(text):
    tokens = [words_to_idx.get(word, words_to_idx["<UNK>"]) for word in text.split()]
    tokens = tokens[-max_len:]
    tokens = [0] * (max_len - len(tokens)) + tokens
    x = torch.tensor([tokens])

    with torch.no_grad():
        output = model(x)
        predicted = torch.argmax(output, dim=1).item()

    return idx_to_words[predicted] 


test_inputs = [
    # Basic questions
    "how", "how are", "how are you",
    "how are you doing", "how are things",

    # Responses
    "i", "i am", "i am great", "i am fine",
    "i am feeling", "i am feeling good",
    "i feel",

    # Emotions
    "i am happy", "i am sad", "i am tired",
    "i am bored", "i am excited", "i am angry",
    
    # Emotion follow ups (NEW)
    "i am sad today", "i am sad always",
    "i am so bored", "i am so tired",
    "i am a bit", "i am really",
    "i am really excited", "i am really happy",

    # Greetings
    "hello", "hello you", "hello there",
    "hello friend", "hi", "hi there",

    # Goodbyes
    "goodbye", "bye", "bye bye",
    "goodbye friend", "goodbye for",
    "ok", "ok bye", "ok goodbye",
    "see", "see you", "see you next",
    "see you later",

    # Time questions
    "what time", "what time is",
    "what is", "what is the",
    "it", "it is",

    # Time variations
    "time", "time to", "time flies",
    "time doesnt", "time doesnt wait",

    # Weather
    "the", "the weather",
    "it is raining", "it is cold",

    # Short affirmations
    "yes", "no", "maybe",
    "of", "i think", "i dont",

    # Casual
    "whats", "not", "all", "doing",

    # Chat style
    "user:", "user: how", "user: what",
    "user: hello", "user: hi",
    "assistant:", "assistant: i",
    "assistant: hello",

    # Noise
    "how r", "im", "ok",

    # Edge
    "what ?", "why", "unknown"
]

for inp in test_inputs:
    result = predict_next(inp)
    print(f"{inp} : {result}")
