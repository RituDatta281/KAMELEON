import os
import argparse
import pandas as pd
from mimic3benchmark.mimic3csv import *

parser = argparse.ArgumentParser(description='Extract ICU stays and annotate 30-day readmission.')
parser.add_argument('mimic3_path', type=str, help='Path to MIMIC-III CSVs.')
parser.add_argument('output_path', type=str, help='Where to save stays.csv.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except FileExistsError:
    pass

# Load necessary tables
patients = read_patients_table(args.mimic3_path)
admits = read_admissions_table(args.mimic3_path)
stays = read_icustays_table(args.mimic3_path)

# Clean and filter
stays = remove_icustays_with_transfers(stays)
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)
stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)

# --- Add 30-day readmission label ---


stays['DISCHTIME'] = pd.to_datetime(stays['DISCHTIME'])
admits['ADMITTIME'] = pd.to_datetime(admits['ADMITTIME'])
admits['DISCHTIME'] = pd.to_datetime(admits['DISCHTIME'])

admits_sorted = admits.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
admits_sorted['NEXT_ADMITTIME'] = admits_sorted.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
admits_sorted['DAYS_TO_NEXT_ADMIT'] = (admits_sorted['NEXT_ADMITTIME'] - admits_sorted['ADMITTIME']).dt.days
admits_sorted['Readmission30'] = (admits_sorted['DAYS_TO_NEXT_ADMIT'] <= 30).astype(int)




# admits_sorted['CUR_DISCHTIME'] = admits_sorted['DISCHTIME']
# admits_sorted['READMIT_GAP_DAYS'] = (admits_sorted['NEXT_ADMITTIME'] - admits_sorted['CUR_DISCHTIME']).dt.total_seconds() / (3600 * 24)
# admits_sorted['Readmission30'] = (admits_sorted['READMIT_GAP_DAYS'] <= 60).astype(int).fillna(0)

# Merge into stays
stays = stays.merge(admits_sorted[['HADM_ID', 'Readmission30']], on='HADM_ID', how='left')

# Save result
stays.to_csv(os.path.join(args.output_path, 'all_stays_new30.csv'), index=False)
