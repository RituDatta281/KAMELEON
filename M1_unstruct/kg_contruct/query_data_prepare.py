import json
from collections import defaultdict
from itertools import chain
from tqdm import tqdm

TASKS = ["mortality"]
DATASETS = ["mimic3","readmission30"]
SAVE_DIR = "/project/biocomplexity/hht9zt/KARE/kg_construct"

all_visit_concepts = []
all_concept_coexistence = defaultdict(lambda: defaultdict(int))

import spacy
nlp = spacy.load("en_core_sci_sm")

def extract_medical_terms(text):
    if not text.strip():
        return []
    doc = nlp(text)
    terms = set()
    for ent in doc.ents:
        if ent.label_ in {"DISEASE", "CHEMICAL", "PROCEDURE", "MEDICATION", "CONDITION", "SYMPTOM"}:
            terms.add(ent.text.lower())
        else:
            # Include any noun chunk if no label is provided (scispaCy is light)
            if len(ent.text) > 3:
                terms.add(ent.text.lower())
    return list(terms)


print("Processing datasets...")
for DATASET in DATASETS:
    for TASK in TASKS:
        agg_samples_path =f"/ehr_prepare/patient_{DATASET}_{TASK}_physician.json"
        agg_samples = json.load(open(agg_samples_path, "r"))

        # Task 1: Get a list of sets of concepts for each visit
        visit_concepts = []
        for patient_id, patient_data in tqdm(agg_samples.items()):
            for visit_id, visit_data in patient_data.items():
                if visit_id.startswith("visit"):
                    structured_codes = list(set(chain(
                        visit_data.get("conditions", []),
                        visit_data.get("procedures", []),
                        visit_data.get("drugs", [])
                    )))

                    discharge_notes = visit_data.get("physician_notes", "") or ""
                    note_concepts = extract_medical_terms(discharge_notes[:900000])

                    combined_concepts = structured_codes + note_concepts
                    visit_concepts.append(combined_concepts)


        # Task 2: Get top 20 co-existing concepts for each concept
        concept_coexistence = defaultdict(lambda: defaultdict(int))
        for visit_concept_set in visit_concepts:
            for concept1 in visit_concept_set:
                for concept2 in visit_concept_set:
                    if concept1 != concept2:
                        concept_coexistence[concept1][concept2] += 1
                        all_concept_coexistence[concept1][concept2] += 1

        top_coexisting_concepts = {}
        for concept, coexistence_counts in concept_coexistence.items():
            top_coexisting_concepts[concept] = [item[0] for item in sorted(coexistence_counts.items(), key=lambda x: x[1], reverse=True)[:20]]

        # Save the results as JSON
        with open(f"{SAVE_DIR}/{DATASET}_{TASK}_visit_concepts_PN.json", "w") as f:
            json.dump(visit_concepts, f, indent=4)
        
        with open(f"{SAVE_DIR}/{DATASET}_{TASK}_top_coexisting_concepts_PN.json", "w") as f:
            json.dump(top_coexisting_concepts, f, indent=4)

        all_visit_concepts.extend(visit_concepts)

# Aggregate results for task 2
all_top_coexisting_concepts = {}
for concept, coexistence_counts in all_concept_coexistence.items():
    all_top_coexisting_concepts[concept] = [item[0] for item in sorted(coexistence_counts.items(), key=lambda x: x[1], reverse=True)[:20]]

# Save the aggregate results as JSON
with open(f"{SAVE_DIR}/all_visit_concepts_PN.json", "w") as f:
    json.dump(all_visit_concepts, f, indent=4)

with open(f"{SAVE_DIR}/all_top_coexisting_concepts_PN.json", "w") as f:
    json.dump(all_top_coexisting_concepts, f, indent=4)