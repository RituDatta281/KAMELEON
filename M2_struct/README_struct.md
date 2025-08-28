1.extracting lab vitals from mimic-III dataset (hourly data of visit)

    - python -m mimic3benchmark.scripts.extract_subjects {mimic-iii-root} .   ##extracting mimic3patient data
    - python -m mimic3benchmark.scripts.extract_subjects-ReadmissionInfo {mimic-iii-root} .
    - python -m mimic3benchmark.scripts.validate_events .
    - python -m mimic3benchmark.scripts.extract_episodes_from_subjects .
    - python -m mimic3benchmark.scripts.split_train_and_test .
    - python -m mimic3benchmark.scripts.create_in_hospital_mortality . data/in-hospital-mortality/
    - python -m mimic3benchmark.scripts.create_readmission . data/readmission/


2. LSTM performance on struct data

    - python -m mimic3models.split_train_val {dataset-directory}
    - python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality
    - python -um mimic3models.readmission.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/readmission


3. Feature Engineering and ML Model Evaluation

    - Run Preprocessing_ML_models.ipynb to perform the following:

    - Extract and prepare lab vitals

    -Merge with:
        Demographic features
        Other structured data: ICD codes, medications, comorbidities
        Unstructured model outputs (e.g., clinical note embeddings)

    - Apply preprocessing:
        Imputation of missing values
        Standardization and discretization
        Dimensionality reduction using PCA

    - Final dataset: train_X, train_y, test_X, test_y

    - Train and evaluate various traditional ML models

    - Compare model performance across classifiers


