import io
import json
import pandas as pd

#Load data from JSON file
with io.open('data.json', 'r', encoding="UTF-8") as file:
    data = json.load(file)


#Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Combine relevant fields to create text data
df['text'] = df['first_name'] + ' ' + df['last_name'] + ' ' + df['email'] + ' ' + df['dx_desc'] + ' ' + df['proc_desc']


# Drop rows with missing values in the 'text' column
df.dropna(subset=['text'], inplace=True)


# Extract the 'text' column as a list
texts = df['text'].tolist()

# Display the first 5 text samples
print(texts[:5])


from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = "[PAD]"
model = GPT2LMHeadModel.from_pretrained('gpt2')


import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoding['input_ids'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].flatten()
        }

dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


from transformers import AdamW, get_linear_schedule_with_warmup

# Set up training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(dataloader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    i = 0
    for batch in dataloader:
        i = i+ 1
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        print(str(i) + '. iteration run, total_loss ' + str(total_loss))
        
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

print('starting training loop')
# Training loop
epochs = 3
for epoch in range(epochs):
    avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

val_loss = evaluate(model, dataloader, device)
print(f'Validation Loss: {val_loss:.4f}')

print('saving model and tokens')
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')



