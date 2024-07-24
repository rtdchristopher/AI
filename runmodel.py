from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Step 1: Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Step 2: Set up the text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Step 3: Define an input prompt
input_prompt = "Dear Lord, I pray that"

# Step 4: Generate text with default parameters
generated_text_default = text_generator(input_prompt, max_length=50)
print("Default Generation:")
print(generated_text_default[0]['generated_text'])

# Step 5: Generate text with custom parameters
generated_text_custom = text_generator(
    input_prompt,
    max_length=50,
    num_return_sequences=1,
    temperature=0.9,
    top_k=50,
    top_p=0.95
)
print("Custom Generation:")
print(generated_text_custom[0]['generated_text'])

dummy = 1

