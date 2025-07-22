from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./.hf_models/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained('./.hf_models/TinyStories-1M')


prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate continuation
output = model.generate(input_ids, max_length=1000, num_beams=1)

# Decode the generated tokens
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Save to file
output_file = "generated_dataset.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Generated text saved to {output_file}")
