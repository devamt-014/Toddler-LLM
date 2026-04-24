#DATA INPUTS
sentences = [

    # 🔹 Basic questions
    "how are you ?",
    "how are you doing ?",
    "how are things ?",

    # 🔹 Responses
    "i am good",
    "i am fine",
    "i am better",
    "i am great",
    "i am great !",
    "i am feeling good today",
    "i am feeling great today",

    # 🔹 Greetings
    "hello",
    "hello you",
    "hello you !",
    "hello there",
    "hello friend",
    "hi there",
    "hi buddy",

    # 🔹 Goodbyes
    "goodbye",
    "bye",
    "ok bye",
    "see you next time",
    "see you later",

    # 🔹 Time questions
    "what time is it ?",
    "what is the time ?",

    # 🔹 Time answers (NEW)
    "it is morning",
    "it is evening",
    "it is night",

    # 🔹 Time variations (your experiment expanded)
    "time to go out",
    "time flies fast",
    "time doesnt wait",
    "time doesnt wait for anyone",

    # 🔹 Short casual phrases
    "whats up",
    "not much",
    "all good",
    "doing great",

    # 🔹 Multi-sentence flows (VERY IMPORTANT)
    "how are you ? i am fine",
    "how are you ? i am great",
    "how are you doing ? i am good",

    # 🔹 Chat-style (BIG UPGRADE)
    "user: how are you ?",
    "assistant: i am fine",

    "user: what time is it ?",
    "assistant: it is night",

    "user: hello",
    "assistant: hello there",

    "user: hi",
    "assistant: hi there",

    # 🔹 Noise / real-world typing
    "how r you",
    "im good",
    "im fine",
    "ok thanks",

    # 🔹 Edge / ambiguity (your fun idea kept)
    "what ?"
]



#SPLITTING UNIQUELY ALL WORDS 
words = set()

for sentence in sentences:
    for word in sentence.split():
        words.add(word) 

words.add("<UNK>")
words = sorted(list(words))

#MAPPINGS
words_to_idx = {word: i for i, word in enumerate(words)}
idx_to_words = {i: word for i, word in enumerate(words)}


#SENTENCES TO NUMBERS 
encoded_sentences = []

for sentence in sentences:
    encoded = [words_to_idx[word] for word in sentence.split()]
    encoded_sentences.append(encoded)
    
    
    
#TAINING DATA
X = []
y = []

for sentence in encoded_sentences:
    for i in range(1,len(sentence)):
        X.append(sentence[:i])  #input sequence
        y.append(sentence[i])   #next word
        
max_len = max(len(seq) for seq in X)

X_padded = []

for seq in X:
    padded = [0] * (max_len - len(seq)) + seq
    X_padded.append(padded)
    
import torch 
X_tensor = torch.tensor(X_padded)
y_tensor = torch.tensor(y)



#OWN TINY LLM MODEL (BRAIN)
import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.fc = nn.Linear(32*max_len, vocab_size)
        
    def forward(self,x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#INITALIZING THE MODEL    
vocab_size = len(words)

model = TinyLM(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


for epoch in range(500):
    output = model(X_tensor)
    loss = loss_fn(output, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")
        
        
def predict_next(text):
    tokens = [words_to_idx.get(word, words_to_idx["<UNK>"]) for word in text.split()]
    tokens = [0] * (max_len - len(tokens)) + tokens
    x = torch.tensor([tokens])

    with torch.no_grad():
        output = model(x)
        predicted = torch.argmax(output, dim=1).item()

    return idx_to_words[predicted] 


print(predict_next("how"))
print(predict_next("how are"))
print(predict_next("how are you"))

print(predict_next("i"))
print(predict_next("i am"))
print(predict_next("i am great"))

print(predict_next("hello"))
print(predict_next("hello you"))
print(predict_next("hi"))

print(predict_next("see"))
print(predict_next("see you"))
print(predict_next("see you next"))

print(predict_next("what"))
print(predict_next("what time"))
print(predict_next("what time is"))
print(predict_next("what is"))
print(predict_next("what is the"))

print(predict_next("time"))
print(predict_next("time to"))
print(predict_next("time flies"))

print(predict_next("user:"))
print(predict_next("user: how"))

print(predict_next("assistant"))
print(predict_next("assistant:"))
print(predict_next("assistant: i"))

print(predict_next("what ?"))
print(predict_next("why"))
print(predict_next("how r"))

print(predict_next("unknown"))








from openpyxl import Workbook

# 🔹 Your test inputs
test_inputs = [
    "how", "how are", "how are you",
    "i", "i am", "i am great",
    "hello", "hello you",
    "see", "see you", "see you next",
    "what", "what time", "what time is",
    "what is", "what is the",
    "time", "time doesnt",
    "user:", "assistant:",
    "unknown"
]

# 🔹 Create workbook
wb = Workbook()
ws = wb.active
ws.title = "Model Analysis"

# 🔹 Headers
ws.append(["Input", "Output"])

# 🔹 Run predictions + store
for inp in test_inputs:
    try:
        output = predict_next(inp)
    except Exception as e:
        output = f"ERROR: {str(e)}"
    
    ws.append([inp, output])

# 🔹 Save file
wb.save("model_analysis.xlsx")

print("Excel file generated: model_analysis.xlsx")