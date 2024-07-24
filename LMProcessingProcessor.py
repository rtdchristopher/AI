import io
import json
import re
import pickle

# Load JSON data from a file
with io.open('dd.json', 'r', encoding="ascii") as file:
    data = json.load(file)
    
# Extract text from each "Request" field
requests = [entry['Request'] for entry in data]    

# Clean and Normalize the Text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove ?
    text = text.replace('?', '')
    # Remove \r\n
    text = re.sub(r'\r\n', '', text)
    return text         

# Apply the cleaning function to each request
cleaned_requests = [clean_text(request) for request in requests]

# pip install transformers
#Tokenization
from transformers import GPT2Tokenizer

# Load a pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Tokenize the text data
tokenized_requests = [tokenizer.encode(request, add_special_tokens=True, max_length=1479) for request in cleaned_requests]

# Convert to Training Data Format
from torch.utils.data import Dataset

class PrayerRequestDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return self.tokenized_texts[idx]

# Create a dataset
dataset = PrayerRequestDataset(tokenized_requests)

# Save the Tokenized Data to a file
with open('tokenized_requests.pkl', 'wb') as file:
    pickle.dump(tokenized_requests, file)


# Define the Model and Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load a pre-trained model and tokenizer
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare the Dataset

# Load JSON data
with io.open('dd.json', 'r', encoding="ascii") as f:
    data = json.load(f)

# Extract text from JSON data
texts = [item["Request"] for item in data]

# Save the extracted texts to a file
with open('prayer_texts.txt', 'w') as f:
    for text in texts:
        f.write(text + '\n')

# Load the dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="prayer_texts.txt",
    block_size=128
)

# Define a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Fine Tune the Model
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the Fine Tuned Model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')







    

