import os
import sys
import json
import random
import pandas as pd
from tqdm import tqdm
import torch
sys.path.append(".")
from apis.gpt_api import get_gpt_response


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_knowledge(patient_context, knowledgeWhole, top_k=1):
    # Embed the patient_context
    context_embedding = model.encode(patient_context, convert_to_tensor=True)

    # Flatten knowledge entries (assuming a dict of {summary_id: text})
    knowledge_items = list(knowledgeWhole.keys())
    knowledge_embeddings = model.encode(knowledge_items, convert_to_tensor=True)
  

    # Compute similarity
    similarities = util.cos_sim(context_embedding, knowledge_embeddings)[0]
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(knowledge_items)))

    top_knowledge = [knowledge_items[i] for i in top_indices]
    # print(top_knowledge,'\n', patient_context)
    # print('\n\n\n--------------------------------------')
    return "\n".join(top_knowledge)

TASKS = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 1 = mortality, 0 = survival

Key Considerations:
1. Conditions:
   - Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)
   - Presence of multiple comorbidities
   - Acute vs. chronic nature of conditions

2. Procedures:
   - Invasiveness and complexity of recent procedures 
   - Emergency vs. elective procedures
   - Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)

3. Medications:
   - Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)
   - Multiple medication use indicating complex health issues
   - Presence of medications typically used in end-of-life care

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.
"""
    }
}

TASKS_ABBR = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit.
Labels: 1 = mortality, 0 = survival

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.
"""
    }
}

LABEL_MAPPING = {
    "mortality": {
        0: "0 (Survival in the subsequent visit)",
        1: "1 (Mortality in the subsequent visit)"
    }
}


TASKS = {
    "readmission30": {
        "description": """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 30 days of discharge based solely on conditions, procedures, and medications.
Labels: 1 = readmission within 30 days, 0 = no readmission within 15 days

Key Considerations:  
1. Conditions:
   - Chronic diseases with high risk of exacerbation (e.g., COPD, heart failure)
   - Conditions requiring close monitoring or frequent adjustments (e.g., diabetes)
   - Recent acute conditions with potential for complications

2. Procedures:
   - Recent major surgeries or interventions with high complication rates
   - Procedures that require extensive follow-up care
   - Incomplete or partially successful procedures

3. Medications:  
   - New medication regimens that may require adjustment
   - Medications with narrow therapeutic windows or high risk of side effects
   - Complex medication schedules that may lead to adherence issues

Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.
"""
    }
}

TASKS_ABBR = {
    "readmission30": {
        "description": """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 30 days of discharge.
Labels: 1 = readmission within 30 days, 0 = no readmission within 30 days

Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.
"""
    }
}

LABEL_MAPPING = {

    "readmission30": {
        0: "0 (No Readmission within 30 days)",
        1: "1 (Readmission within 30 days)"
    },
}


def generate_reasoning(patient_context, task, ground_truth, medical_knowledge, similar_patient=None):
    context = patient_context
    similar_patients = similar_patient['positive'] + similar_patient['negative']
    random.shuffle(similar_patients)

    prompt = f"""
Given the following task description, patient EHR context, similar patients, retrieved medical knowledge, and ground truth label, provide a step-by-step reasoning process that leads to the correct prediction, Note that, you must give your prediction based on the Ground Truth label, and the reasoning should be short within 300 tokens:

========================================
# Task #
{TASKS[task]['description']}

========================================
# Patient EHR Context #

{context}

========================================
# Similar Patients #

{" ".join(similar_patients)}

========================================
# Retrieved Medical Knowledge #

{medical_knowledge}

========================================
# Ground Truth #

{ground_truth}

========================================

Please provide a step-by-step reasoning process that leads to the correct prediction...

# Confidence #
[CONFIDENCE (choose one: "Very Confident", "Confident", "Neutral", "Not Confident", "Very Not Confident")]
"""
    response = get_gpt_response(prompt=prompt[:128000])
    print(response, '\nACTUAL LABEL:', ground_truth)
    return response

def construct_input_output_2(patient_context, task, reasoning, ground_truth, medical_knowledge, similar_patient=None):
    similar_patients = similar_patient['positive'] + similar_patient['negative']
    random.shuffle(similar_patients)

    input_ = f"""
Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge...

========================================
# Task #
{TASKS_ABBR[task]['description']}

========================================
# Patient EHR Context #

{patient_context}

========================================
# Similar Patients #

{" ".join(similar_patients)}

========================================
# Retrieved Medical Knowledge #

{medical_knowledge}
"""
    output_ = f"""
# Reasoning #
{reasoning}

# Prediction #
{ground_truth}
"""
    return input_, output_

def construct_input_output(patient_context, task, reasoning, ground_truth, medical_knowledge, similar_patient=None):
    similar_patients = similar_patient['positive'] + similar_patient['negative']
    random.shuffle(similar_patients)

    input_ = f"""
Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge...

========================================
# Task #
{TASKS_ABBR[task]['description']}

========================================
# Patient EHR Context #

{patient_context}
========================================
# Similar Patients #

{" ".join(similar_patients)}

========================================
# Retrieved Medical Knowledge #

{medical_knowledge}
"""
    output_ = f"""
# Reasoning #
{reasoning}

# Prediction #
{ground_truth}
"""
    return input_, output_

def process_patient(patient_id, patient_context, task, ground_truth, train_ids, val_ids, test_ids, medical_knowledge=None, similar_patient=None):
    print(f"Processing patient {patient_id}...")

    reasoning = generate_reasoning(patient_context, task, ground_truth, medical_knowledge, similar_patient)
    if not reasoning:
        return None

    if (patient_id in train_ids or patient_id in val_ids) and "\n# Confidence #\nNot Confident" in reasoning:
        return None

    reasoning = reasoning.split("\n# Confidence #\n")[0]
    reasoning = reasoning.replace("# Reasoning Chain #\n", "")
    input_, output_ = construct_input_output(patient_context, task, reasoning, ground_truth, medical_knowledge, similar_patient)

    item = {"input": input_, "output": output_}
    if patient_id in train_ids:
        return ("train", item)
    elif patient_id in val_ids:
        return ("val", item)
    elif patient_id in test_ids:
        return ("test", item)
    return None


def process_dataset(dataset, task, i):
    # Load data
    base_path = f"."
    save_path = f"{base_path}/llm_finetune_data_ulti"

    context_path = f"{base_path}/patient_context/base_context/patient_contexts_{dataset}_{task}_notes.json"
    similar_path = f"{base_path}/patient_context/similar_patient/patient_to_top_1_patient_contexts_{dataset}_{task}_notes.json"
    knowledge_path = f"{base_path}/indexing/community_summary_to_nodes_pubmed.json"
    patient_data_path = f"{base_path}/pateint_{dataset}_{task}.json"
    test_sample_path = f"{base_path}/ehr_data/{dataset}_{task}_samples_test_PN_summary.json"
    train_sample_path = f"{base_path}/ehr_data/{dataset}_{task}_samples_train_PN_summary.json"
    val_sample_path = f"{base_path}/ehr_data/{dataset}_{task}_samples_val_PN_summary.json"

    patient_contexts = json.load(open(context_path))
    patient_data = json.load(open(patient_data_path))
    similar_patients = json.load(open(similar_path))
    test_samples = json.load(open(test_sample_path))
    train_samples = json.load(open(train_sample_path))
    val_samples = json.load(open(val_sample_path))
    knowledgeWhole = json.load(open(knowledge_path))

    # Get patient IDs
    patient_id_test = [f"{s['patient_id']}_{s['visit_id']}" for s in test_samples]
    patient_id_train = [f"{s['patient_id']}_{s['visit_id']}" for s in train_samples]
    patient_id_val = [f"{s['patient_id']}_{s['visit_id']}" for s in val_samples]
    patient_id_all = patient_id_train + patient_id_val + patient_id_test
    random.shuffle(patient_id_all)

    train_data, val_data, test_data = [], [], []

    counter = 0  # Track patients processed

    for patient_id in tqdm(patient_id_all, desc="Processing patients"):
        # try:
        result = process_patient(
            patient_id,
            patient_contexts[patient_id],
            task,
            patient_data[patient_id]["label"],
            patient_id_train,
            patient_id_val,
            patient_id_test,
            get_relevant_knowledge(patient_contexts[patient_id], knowledgeWhole),
            similar_patients[patient_id]
        )
        if result:
            split, item = result
            if split == "train":
                train_data.append(item)
            elif split == "val":
                val_data.append(item)
            elif split == "test":
                test_data.append(item)
        # except Exception as e:
        #     print(f"{patient_id} failed: {e}")
        
        counter += 1
        if counter % 5 == 0:
            # Save checkpoint after every 5 patients
            with open(f"{save_path}/{dataset}_{task}_train_{i}_notes_checkpoint.jsonl", "a") as f:
                for item in train_data + val_data:
                    f.write(json.dumps(item) + "\n")
                train_data.clear()
                val_data.clear()

            with open(f"{save_path}/{dataset}_{task}_test_{i}_notes_checkpoint.jsonl", "a") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")
                test_data.clear()

    # Final save
    with open(f"{save_path}/{dataset}_{task}_train_{i}_notes.jsonl", "w") as f:
        for item in train_data + val_data:
            f.write(json.dumps(item) + "\n")

    with open(f"{save_path}/{dataset}_{task}_test_{i}_notes.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    return train_data, val_data, test_data


# === MAIN ===
if __name__ == "__main__":
    for dataset in ["mimic3"]:
        for task in ["readmission30", "mortality"]:
            all_train, all_val, all_test = [], [], []
            for i in range(1):
                print(f"Starting {dataset}_{task}_{i}...")
                train, val, test = process_dataset(dataset, task, i)
                all_train.extend(train)
                all_val.extend(val)
                all_test.extend(test)

            final_path = "llm_finetune_data_ulti"
            with open(f"{final_path}/{dataset}_{task}_train_notes.jsonl", "w") as f:
                for item in all_train + all_val:
                    f.write(json.dumps(item) + "\n")

            with open(f"{final_path}/{dataset}_{task}_test_notes.jsonl", "w") as f:
                for item in all_test:
                    f.write(json.dumps(item) + "\n")

            print(f"✅ Finished {dataset}_{task}")
            