from unsloth import FastLanguageModel
import torch


from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset # Import load_dataset here


max_seq_length = 8192
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### label:
{}

### Reasoning:
{}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["label"]
    reasonings   = examples["Reasoning"]
    texts = []
    for instruction, input, output, reasoning in zip(instructions, inputs, outputs, reasonings):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        # Ensure 'output' is formatted as an integer for the prompt, then it will be a string in 'text'
        text = alpaca_prompt.format(instruction, input, int(output), reasoning) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts } # Return only the 'text' field

file_path = "alpaca_readmission30_notes.json"

dataset = load_dataset("json", data_files={"train": file_path}, split="train")

print("Original dataset structure:")
print(dataset)

# Apply the formatting function to create the 'text' column
dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=2) # Use num_proc for parallel processing

# Remove the original columns that are no longer needed, to prevent confusion for the trainer
# The 'text' column now contains the full formatted prompt, including instruction, input, label, and reasoning
dataset = dataset.remove_columns(["instruction", "input", "label", "Reasoning"])

print("\nDataset structure after formatting and column removal:")
print(dataset)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # This specifies that the 'text' column contains the input for training
    max_seq_length = max_seq_length,
    dataset_num_proc = 2, # Number of processes to use for data loading
    packing = False, # Can make training 5x faster for short sequences. Set to True if your sequences are often short.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 200, # For testing, increase this for actual training
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(), # Use fp16 if bf16 is not supported
        bf16 = torch.cuda.is_bf16_supported(), # Use bf16 if supported
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_steps = 10, # Uncomment and set to save checkpoints periodically
        save_total_limit = 6, # Uncomment and set to limit the number of saved checkpoints
    ),
)

print("\nStarting training...")
trainer_stats = trainer.train()
print("\nTraining complete.")


# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

# Save the finetuned model
save_path = "lora_model_readmission30_notes"
model.save_pretrained(save_path)
print(f"Model saved to {save_path}")

import json
# Load the original data again for inference example
with open('alpaca_readmission30_notes.json', 'r') as jsonfile:
    admission_data = json.load(jsonfile)
    
# Select an example for inference (e.g., the second entry)
example_for_inference = admission_data[1]

# Format the input for inference - leave label and Reasoning blank
inference_prompt = alpaca_prompt.format(
    example_for_inference["instruction"], # instruction
    example_for_inference["input"],     # input
    "",                                # label - leave this blank for generation!
    "",                                # Reasoning - leave this blank for generation!
)

print(f"\n--- Inference Prompt ---\n{inference_prompt}")

inputs = tokenizer(
    [inference_prompt],
    return_tensors = "pt"
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) # skip_prompt=True to only show generated tokens

print("\n--- Model Generating Response ---")
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 500)

print("\n--- End of Inference ---")