from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
model = AutoModelForCausalLM.from_pretrained('./.hf_models/gpt2-medium')
tokenizer = AutoTokenizer.from_pretrained('./.hf_models/gpt2-medium')


output_file = "generated_dataset_gpt2-medium.txt"
random.seed(42)
prompts = [
    "Max had a secret.",
    "Lily found a strange key.",
    "Ben wanted to fly like a bird.",
    "Sara’s robot wouldn’t stop dancing.",
    "Jack built a castle in the backyard.",
    "Mia heard a whisper from her closet.",
    "Tom found a magic crayon.",
    "Ella’s toy dragon started breathing fire.",
    "Noah lost his voice in a thunderstorm.",
    "Emma followed a glowing butterfly.",
    "A cat named Socks wanted to be a chef.",
    "The sleepy bear missed the train.",
    "A tiny turtle raced a rocket.",
    "A bird with rainbow feathers sang a secret song.",
    "The elephant forgot how to trumpet.",
    "A curious fox discovered a mirror.",
    "A frog dreamed of touching the stars.",
    "The lion cub was scared of the dark.",
    "A whale followed a paper boat.",
    "The bunny baked cookies for the moon.",
    "A wizard lost her wand at school.",
    "The cloud turned into a sheep.",
    "A potion turned Alex into a squirrel.",
    "Time stopped when Max sneezed.",
    "A book began to speak.",
    "The stars fell into Olivia’s backyard.",
    "A staircase appeared in the sky.",
    "A snowflake whispered a warning.",
    "A mirror showed the future.",
    "The rainbow had a secret door.",
    "It rained glitter today.",
    "Jamie’s socks ran away.",
    "The fridge started singing.",
    "Sam’s shadow disappeared.",
    "Anna could only speak in rhymes.",
    "A sandwich came to life.",
    "The TV asked for a nap.",
    "The crayons argued all night.",
    "Alex’s bicycle told a story.",
    "The swings flew into space.",
    "The robot forgot how to beep.",
    "A space jellybean landed on Earth.",
    "The moon invited Mia for tea.",
    "Stars spelled out a message.",
    "An alien visited third grade.",
    "Leo’s toy rocket vanished.",
    "Earth turned blue and pink.",
    "Sam rode a comet to school.",
    "Gravity disappeared for one hour.",
    "A black hole opened under the bed."
]
sampled_prompts = random.choices(prompts, k=1000)

generated_stories = []

for i, prompt in enumerate(sampled_prompts):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate continuation
    output = model.generate(
        input_ids,
        max_length=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_stories.append({
        "prompt": prompt,
        "story": output_text
    })

    if i % 100 == 0:
        print(f"{i} stories generated...")

  

with open("generated_dataset_gpt2-medium.jsonl", "w", encoding="utf-8") as f:
    for entry in generated_stories:
        f.write(json.dumps(entry) + "\n")

print("Generated dataset saved to generated_dataset_gpt2-medium.jsonl")
