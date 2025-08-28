import json
import re
dataset = "mimic3"
task = "readmission30"
output_path = "llm_finetune_data_multitask"
# mimic3_mortality_train_0_notes_checkpoint,

        
# ==========================================
# train data
ori_path = f"llm_finetune_data_ulti/{dataset}_{task}_train_0_notes_checkpoint.jsonl"

with open(ori_path, "r") as f:
    data = [json.loads(line) for line in f]
t = 0

    
instruction_prev_prev = "\nGiven the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context and relevant medical knowledge.\nAfter the reasoning process, provide the prediction label (0/1)."
instruction_prev = "\nGiven the following task description, patient EHR context, similar patients, and retrieved medical knowledge..."
instruction_new_reason = "\n[Reasoning] Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context and relevant medical knowledge.\nAfter the reasoning process, provide the prediction label (0/1)."
instruction_new_pred = "\n[Label Prediction] Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please directly predict the label (0/1).\n"


label_pred_data = []
reasoning_data = []
patient_id_pattern = re.compile(
    r"# Patient EHR Context #.*?Patient ID:\s*([^\s]+)",
    re.DOTALL
)

ori_path = f"llm_finetune_data_ulti/{dataset}_{task}_test_0_notes_checkpoint.jsonl"
context_path = f"patient_context/base_context/patient_contexts_{dataset}_{task}_notes.json"
patient_data_path = f"ehr_prepare/pateint_{dataset}_{task}_physician_summary.json"
patient_data = json.load(open(patient_data_path))

for item in data:
    input_new = item["input"].replace(instruction_prev, instruction_new_reason)
        
    match = patient_id_pattern.search(input_new)
    if match:
        patient_id = match.group(1)
        t+= 1
    else:
        print('couldnt find')

    output_new = "# Prediction # " + str(patient_data[patient_id]['label'])
    # print(output_new)
    label_pred_data.append({"input": input_new, "output": output_new})
    
    # input_new = item["input"].replace(instruction_prev, instruction_new_reason)
    output_new = item["output"]
    reasoning_data.append({"input": input_new, "output": output_new})
    
    
data = label_pred_data
with open(f"{output_path}/{dataset}_{task}_train_notes.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
        
        
# # ==========================================
# # test data



with open(ori_path, "r") as f:
    data = [json.loads(line) for line in f]
    
    
label_pred_data = []

for item in data:
    input_new = item["input"].replace(instruction_prev, instruction_new_pred)
    # output_new = item["output"][-1]
    match = patient_id_pattern.search(input_new)
    if match:
        patient_id = match.group(1)
        # print(patient_id)
        t+= 1
    else:
        print('couldnt find')

    output_new = "# Prediction # " +str(patient_data[patient_id]['label'])
    label_pred_data.append({"input": input_new, "output":output_new})
    
with open(f"{output_path}/{dataset}_{task}_test_notes.jsonl", "w") as f:
    for item in label_pred_data:
        f.write(json.dumps(item) + "\n")