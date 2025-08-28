from unsloth import FastLanguageModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
from datasets import load_dataset
import pandas as pd
import numpy as np
import json


max_seq_length = 8500 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3  Jiaming using
] # More models at https://huggingface.co/unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-7b-it-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

FastLanguageModel.for_inference(model) 


def get_llm_response(prompt, model=model):
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    output_tokens = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return response.split("Extracted triples:")[-1].strip()
    

def extract_relationships(text, concepts):
    """Extract relationships from text using LLM."""
    example = """Example:
    Text:
    Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. Common symptoms include wheezing, coughing, shortness of breath, and chest tightness. Triggers can vary but often include allergens, air pollution, exercise, and respiratory infections. Management typically involves a combination of long-term control medications, such as inhaled corticosteroids, and quick-relief medications like short-acting beta-agonists. Recent research has focused on personalized treatment approaches, including biologics for severe asthma and the role of the microbiome in asthma development and progression. Proper inhaler technique and adherence to medication regimens are crucial for effective management. Asthma action plans, developed in partnership with healthcare providers, help patients manage symptoms and exacerbations.

    Concepts: [asthma, inflammation, airways, wheezing, coughing, inhaled corticosteroids, short-acting beta-agonists, allergens, respiratory infections]

    Extracted triples:
    [[asthma, is a, chronic respiratory condition], [asthma, characterized by, inflammation of airways], [inflammation, causes, narrowing of airways], [narrowing of airways, leads to, breathing difficulties], [wheezing, is a symptom of, asthma], [coughing, is a symptom of, asthma], [allergens, can trigger, asthma], [respiratory infections, can trigger, asthma], [inhaled corticosteroids, used for, long-term control of asthma], [short-acting beta-agonists, provide, quick relief in asthma]]
    """

    prompt = f"""Given a medical text and a list of important concepts, extract relevant relationships between the concepts from the text (if present). For each triple, if an entity matches one of the given concepts, replace the entity with the exact concept term.

    Focus on generating high-quality triples closely related to the provided concepts. Aim to extract at most 10 triples for each text. Each triple should follow this format: [ENTITY1, RELATIONSHIP, ENTITY2]. Ensure the triples are informative and logically sound.

    {example}

    Text:
    {text}

    Concepts: {concepts}

    Extracted triples:
    """

    response = get_llm_response(prompt=prompt)
    return response


def get_embedding(text, model=model, tokenizer=tokenizer):
    """
    Generate embeddings for a given text using the Unsloth Gemma model.
    
    Args:
        text (str): Input text to encode.
        model: Preloaded Unsloth model.
        tokenizer: Corresponding tokenizer.
    
    Returns:
        torch.Tensor: Embedding representation of the input text.
    """
    text = text.replace("\n", " ")  # Normalize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8500)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    embeddings = outputs.hidden_states[-1].mean(dim=1) 
    return embeddings

def get_unsloth_response(prompt, model=model, tokenizer=tokenizer, max_new_tokens=400, temperature=0.7, top_p=0.9):
    """
    Generate a response from the Unsloth Gemma model.

    Args:
        prompt (str): Input prompt for the model.
        model: Preloaded Unsloth model.
        tokenizer: Corresponding tokenizer.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness in sampling.
        top_p (float): Controls nucleus sampling.

    Returns:
        str: Generated response.
    """
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )

    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response

