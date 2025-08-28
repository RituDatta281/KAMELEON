Please the codes for the following folders sequentially.

# ehr_prepare
	1. run ehr_prepare.ipynb
		- Fix the directory first.
		- It will create patient info for both task

	2. run sample_prepare.py
		- fix the directory
		- running this file will create train test val set with physician notes

# kg_construct
	1. python query_data_prepare.py
	2. python download_pubmed.py
	3. python embed_pubmed.py
	4. python convert_dat.py
	5. run all cells of kg_from_pubmed.ipynb for generating triples **before that please set up the apis in apis/ folder**
	6. python llm_source.py **this is will generate KG using llm**
	7. python combine-Updated.py **will combine the KGs**
	8. python structure_partition_leiden-PPID.py

# patient_context
	1. python base_context.py
	2. python get_emb.py
	3. python sim_patient_ret_faiss.py
	4. python augment_context-Updated.py

# prediction
	1. python data_prepare_KG.py

# finetune
	1. Processing.ipynb has 3 sections
		- the first few cells under markdown "## Preprocessing" needs to run first
	2. python finetuning.py  **for finetuning**
	3. Processing.ipynb [last 2 sections]
		- cells under "## Inference" for inferencing and saving result
		- run the cells under "## Performance Analaysis" for checking accuracy and other metric
		- save the final inference result and reasoning for structured analysis

# checkhospice.ipynb
	- this one is for sanity check, to ensure in physican notes clear indication of mortality is not mentioned




