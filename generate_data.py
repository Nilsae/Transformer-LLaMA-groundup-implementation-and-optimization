from transformers import AutoModelForCausalLM, AutoTokenizer
import random
model = AutoModelForCausalLM.from_pretrained('./.hf_models/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained('./.hf_models/TinyStories-1M')


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

output_file = "generated_dataset.txt"

for i in range(1000):
    n = random.randint(0, 49)

    input_ids = tokenizer.encode(prompts[n], return_tensors="pt")

    # Generate continuation
    output = model.generate(input_ids, max_length=1024, num_beams=1)

    # Decode the generated tokens
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Save to file

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(output_text + "\n\n---\n\n")


print(f"Generated text saved to {output_file}")
