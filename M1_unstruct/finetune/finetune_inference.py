from unsloth import FastLanguageModel
import torch
from peft import PeftModel
from transformers import TextStreamer
import json
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
# === SETTINGS ===
max_seq_length = 16384
dtype = None
load_in_4bit = True
lora_path = "lora_model_mortality_notes"
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

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

# === LOAD DATA ===
with open('alpaca_mortality_notes.json', 'r') as jsonfile:
    admission = json.load(jsonfile)

# === LOAD BASE MODEL ===
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# === LOAD LoRA ADAPTER ===
model = PeftModel.from_pretrained(
    base_model,
    model_id=lora_path,
    adapter_name="default"
)

# === PREPARE FOR INFERENCE ===
FastLanguageModel.for_inference(model)
model.eval()

import re
import os


# === INIT CSV ===
csv_file = "predictions_vs_ground_truth_train_mortality_notes.csv"
df = pd.read_csv(csv_file)
done_ids = list(df['patient_id'].unique())

fieldnames = ["patient_id", "ground_truth", "prediction", "match", "reasoning"]

# # # Write header first
# with open(csv_file, "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()

# === INFERENCE LOOP WITH PERIODIC SAVING ===
results = []
y_true = []
y_pred = []

for i, entry in enumerate(admission):
    anomaly = ''
    input_text = entry["input"]
    GT = entry["label"]
    prompt = alpaca_prompt.format(entry["instruction"], input_text, "", "")
    
    match = re.search(r"# Patient EHR Context #\n\nPatient ID:\s*([^\s\n]+)", input_text)
    patient_id = match.group(1) if match else "UNKNOWN"
    if patient_id in done_ids:
        continue

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    try:
        outputs = model.generate(**inputs, max_new_tokens=500)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = prediction.split("### Reasoning:")[-1].strip()
    except: 
        print('Skipped one')
        continue
 
    
    pred_label = response.split("### Prediction:")[-1].strip().lower()
    pred_label = re.sub(r'\d+\.', '.', pred_label)
    pred_label = re.sub(r'\d+\:', ':', pred_label)
    pred_label = re.sub(r'<n>\d+</n>', '', pred_label)
    pred_label = pred_label.replace("predict hospital readmission within 30 days", "")
    pred_label = re.sub(r'\d+_\d+', '', pred_label)
    print('\n PRED LABEL: \n------------', pred_label)
    
    if "no readmission" in pred_label or "0" in pred_label:
        pred_label = "0"
    
    elif ("readmission within 30 days" in pred_label or "readmission in 30 days" in pred_label or  "1" in pred_label or "likely" in pred_label or "higher risk" in pred_label or "high risk" in pred_label or "high likelihood" in pred_label):
        pred_label = "1"
    else:
        pred_label = "0"

    # pred_label = "1" if "1" in pred_label else "0"
    
    
    print('Predicted Label', pred_label)
    

    y_true.append(int(GT))
    y_pred.append(int(pred_label))

    result = {
        "patient_id": patient_id,
        "ground_truth": GT,
        "prediction": pred_label,
        "match": GT == pred_label,
        "reasoning": response
    }

    results.append(result)

    # === Save every 10 entries ===
    if (i + 1) % 10 == 0 or (i + 1) == len(admission):
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(results)
        results = []  # reset buffer

# === METRICS ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
try:
    roc_auc = roc_auc_score(y_true, y_pred)
except ValueError:
    roc_auc = "N/A (only one class present)"

# === PRINT METRICS ===
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc}")
print(f"Anomalies ({len(anomaly)}):", anomaly)

