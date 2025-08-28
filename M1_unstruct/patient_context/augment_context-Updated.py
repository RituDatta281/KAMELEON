import os
import json
import pickle
import csv
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np

import openai


client = openai.OpenAI(api_key="_")  ## add api key here


def get_embedding(text, model="text-embedding-3-large"):
    # print(text)
    if isinstance(text, list):
        text = " ".join(map(str, text))
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


MORTALITY_THEME_TERMS = [
    "terminal condition", "critical diagnosis", "end-stage", "life-threatening",
    "critical interventions", "life-saving measures", "resuscitation", "emergency procedure",
    "palliative medication", "end-of-life drugs", "life support drugs", "emergency meds"
]

READMISSION_THEME_TERMS = [
    "chronic ailment", "postoperative complication", "recurrent", "readmission-prone",
    "follow-up procedure", "secondary intervention", "post-treatment", "treatment review",
    "maintenance medication", "postoperative drugs", "treatment continuation", "follow-up meds"
]


def load_mappings():
    condition_mapping_file = "./resources/CCSCM.csv"
    procedure_mapping_file = "./resources/CCSPROC.csv"
    drug_file = "./resources/ATC.csv"

    condition_dict = {}
    condition_dict_inv = {}
    with open(condition_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()
            condition_dict_inv[row['name'].lower()] = row['code']

    procedure_dict = {}
    procedure_dict_inv = {}
    with open(procedure_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()
            procedure_dict_inv[row['name'].lower()] = row['code']

    drug_dict = {}
    drug_dict_inv = {}
    with open(drug_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()
                drug_dict_inv[row['name'].lower()] = row['code']

    return condition_dict, procedure_dict, drug_dict, condition_dict_inv, procedure_dict_inv, drug_dict_inv


def community_summary_to_nodes():
    summary_to_nodes = {}
    path = "/project/biocomplexity/hht9zt/KARE/indexing"
    for file in os.listdir(path):
        if file.endswith(".json") and file.startswith("communities_Pubmed"):
            community_data = json.load(open(os.path.join(path, file), "r"))
            for d in community_data:
                if d['summary'] != "The community is too large":
                    node_set = set()
                    triples = d['triples']
                    for triple in triples:
                        entity_1, relationship, entity_2 = triple
                        node_set.add(entity_1)
                        node_set.add(entity_2)
                    summary_to_nodes[d['summary']] = list(node_set)
    return summary_to_nodes
    

def load_entity_mappings():
    entity_mapping_path = "/clustering/original_to_new_entity_mapping_0.14.json"
    entity_mapping = json.load(open(entity_mapping_path, "r"))
    return entity_mapping


def read_indirect_nodes(code, main_concept, entity_mapping, mode):
    graph_path = f"../kg_construct/graph_PPID/{mode}/{code}.txt"
    indirect_nodes = set()
    with open(graph_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        entity1, relationship, entity2 = line.strip().split("\t")
        if entity1 != main_concept:
            indirect_nodes.add(entity_mapping[entity1])
        if entity2 != main_concept:
            indirect_nodes.add(entity_mapping[entity2])
            
    return indirect_nodes
    
def filter_func(data, visit_id, entity_mapping, mode, concept_dict_inv):
    concepts = []
    code_map = {}
    for concept in data[visit_id][mode]:
        if concept in entity_mapping:
            concepts.append(concept)
            code_map[concept] = concept_dict_inv[concept]
        else:
            if ";" in concept:
                for c in concept.split(";"):
                    if c in entity_mapping:
                        concepts.append(c)
                        code_map[c] = concept_dict_inv[concept]
    return concepts, code_map
    
    
def patient_graph_construct(data, entity_mapping, condition_dict_inv, procedure_dict_inv, drug_dict_inv):
    num_visit = len(data) - 1
    direct_nodes, indirect_nodes = set(), set()
    for i in range(num_visit):
        visit_id = f'visit {i}'

        conditions, condition_code_map = filter_func(data, visit_id, entity_mapping, "conditions", condition_dict_inv)
        procedures, procedure_code_map = filter_func(data, visit_id, entity_mapping, "procedures", procedure_dict_inv)
        drugs, drug_code_map = filter_func(data, visit_id, entity_mapping, "drugs", drug_dict_inv)
        
        for condition in conditions:
            dirent_node = entity_mapping[condition]
            direct_nodes.add(dirent_node)
            code = condition_code_map[condition]
            indirect_nodes_condition = read_indirect_nodes(code, condition, entity_mapping, "condition")
            indirect_nodes.update(indirect_nodes_condition)
            
        for procedure in procedures:
            direct_node = entity_mapping[procedure]
            direct_nodes.add(direct_node)
            code = procedure_code_map[procedure]
            indirect_nodes_procedure = read_indirect_nodes(code, procedure, entity_mapping, "procedure")
            indirect_nodes.update(indirect_nodes_procedure)
            
        for drug in drugs:
            direct_node = entity_mapping[drug]
            direct_nodes.add(direct_node)
            code = drug_code_map[drug]
            indirect_nodes_drug = read_indirect_nodes(code, drug, entity_mapping, "drug")
            indirect_nodes.update(indirect_nodes_drug)
            
    return direct_nodes, indirect_nodes     


def theme_relevance(summary, task, lambda_3=0.1):
    theme_terms = MORTALITY_THEME_TERMS if task == "mortality" else READMISSION_THEME_TERMS
    theme_embeddings = get_embedding(theme_terms)
    summary_embedding = list(get_embedding([summary]).values())[0]

    max_similarity = max(np.dot(summary_embedding, theme_embedding) for theme_embedding in theme_embeddings.values())
    return 1 + lambda_3 * max_similarity

def dynamic_graph_retrieval_augmentation(patient_nodes, summary_to_nodes, summary_embeddings, patient_base_context_embedding, task, patient_node_visit_idx=None, N=10, alpha=0.8, beta=0.5, lambda_1=0.1, lambda_2=0.1, lambda_3=0.1):
    candidate_communities = list(summary_to_nodes.keys())
    selected_summaries = []
    node_hit_count = defaultdict(int)
    patient_direct_nodes = set(patient_nodes["direct"])
    patient_indirect_nodes = set(patient_nodes["indirect"])
    
    for _ in range(N):
        if not candidate_communities:
            break

        relevance_scores = {}
        for summary in candidate_communities:
            community_nodes = set(summary_to_nodes[summary])

            h_direct = len(patient_direct_nodes.intersection(community_nodes)) / len(patient_direct_nodes)

            h_indirect = len(patient_indirect_nodes.intersection(community_nodes)) / len(patient_indirect_nodes)

            decay = np.prod([beta ** node_hit_count[node] for node in community_nodes])

            coherence = 1 + lambda_1 * np.dot(summary_embeddings[summary], patient_base_context_embedding)
            
            if patient_node_visit_idx:
                hit_direct_nodes = patient_direct_nodes.intersection(community_nodes)
                recency = 1 + lambda_2 * np.mean([patient_node_visit_idx.get(node, 0) for node in hit_direct_nodes])
            else:
                recency = 1 
                
            theme_rel = theme_relevance(summary, task, lambda_3)

            relevance_score = (h_direct + alpha * h_indirect) * decay * coherence * recency * theme_rel
            relevance_scores[summary] = relevance_score

        best_summary = max(relevance_scores, key=relevance_scores.get)
        selected_summaries.append(best_summary)

        for node in summary_to_nodes[best_summary]:
            node_hit_count[node] += 1

        candidate_communities.remove(best_summary)

    return selected_summaries

def main():
    if not os.path.exists("indexing/community_summary_to_nodes_pubmed.json"):
        print("mapping community summaries to nodes")
        summary_to_nodes = community_summary_to_nodes()
        with open("indexing/community_summary_to_nodes_pubmed.json", "w") as f:
            json.dump(summary_to_nodes, f, indent=4)
        print("generating embeddings for community summaries")
        summary_embeddings = get_embedding(list(summary_to_nodes.keys()))
        with open("indexing/community_summary_embeddings_pubmed.pkl", "wb") as f:
            pickle.dump(summary_embeddings, f)
            
    else:
        print("loading community summaries to nodes mapping")
        with open("indexing/community_summary_to_nodes_pubmed.json", "r") as f:
            summary_to_nodes = json.load(f)
        print("loading community summary embeddings")
        with open("/indexing/community_summary_embeddings_pubmed.pkl", "rb") as f:
            summary_embeddings = pickle.load(f)
            
    
    datasets = ["mimic3", 
                ]
    tasks = ["readmission30", "mortality"]

    ehr_dir = "ehr_prepare"
    context_dir = "patient_context"
    
    condition_dict, procedure_dict, drug_dict, condition_dict_inv, procedure_dict_inv, drug_dict_inv = load_mappings()
    entity_mapping = load_entity_mappings()
  
    for dataset in datasets:
        for task in tasks:
            print(f"Processing dataset: {dataset}, task: {task}")
            if os.path.exists(f"{ehr_dir}/pateint_{dataset}_{task}_physician_summary.json"):
                print(f"Patient data already exists for dataset: {dataset}, task: {task}, loading from file")
                patient_data = json.load(open(f"{ehr_dir}/pateint_{dataset}_{task}_physician_summary.json", "r"))
                base_context = json.load(open(f"{context_dir}/base_context/patient_contexts_{dataset}_{task}_notes.json", "r"))
            else:
                print("Please run base_context.py to generate patient data and context first.")

            ## Patient graph construction
            if os.path.exists(f"indexing/patient_nodes_{dataset}_{task}_notes.pkl"):
                print(f"Patient nodes already exists for dataset: {dataset}, task: {task}, loading from file")
                with open(f"indexing/patient_nodes_{dataset}_{task}_notes.pkl", "rb") as f:
                    patient_nodes = pickle.load(f)
            else:
                print("Constructing patient graphs ...")
                patient_nodes = defaultdict(dict)
                for patient in tqdm(patient_data.keys()):
                    data = patient_data[patient]
                    direct_nodes, indirect_nodes = patient_graph_construct(data, entity_mapping, condition_dict_inv, procedure_dict_inv, drug_dict_inv)
                    patient_nodes[patient] = {"direct": list(direct_nodes), "indirect": list(indirect_nodes)}
                    
                print("saving patient nodes ...")
                with open(f"indexing/patient_nodes_{dataset}_{task}_notes.pkl", "wb") as f:
                    pickle.dump(patient_nodes, f)
            
            ## Patient Context Embeddings
            if os.path.exists(f"indexing/context_embeddings_{dataset}_{task}_notes.pkl"):
                print(f"Patient context embeddings already exists for dataset: {dataset}, task: {task}, loading from file")
                with open(f"indexing/context_embeddings_{dataset}_{task}_notes.pkl", "rb") as f:
                    patient_context_embs = pickle.load(f)
            else:
                print("Generating patient context embeddings ...")
                patient_context_embs = get_embedding(list(base_context.values()))
                with open(f"indexing/context_embeddings_{dataset}_{task}_notes.pkl", "wb") as f:
                    pickle.dump(patient_context_embs, f)
                    
            
            # Patient node visit index
            if os.path.exists(f"{ehr_dir}/pateint_{dataset}_{task}_node_visit_idx_notes.json"):
                print(f"Patient node visit index already exists for dataset: {dataset}, task: {task}, loading from file")
                with open(f"{ehr_dir}/pateint_{dataset}_{task}_node_visit_idx_notes.json", "r") as f:
                    patient_node_visit_idx = json.load(f)
            else:
                patient_node_visit_idx = None
          

if __name__ == "__main__":
    main()