from src.pubchem_download import download_pubchem_data as pcd
from src.structure_automation import generate_slurm_scripts as gss
from src.structure_automation import generate_slurm_scripts_from_csv as gss_csv
from src.csv_conversion import csv_validator as cvd
from src.dft_parser import parse_log as pl
from src.ml_analysis import analysis
from src.ml_analysis import cv_check as cv_check
import pandas as pd
import random

default_user = {"uniqname": "xaustin", "DFT_folder": "AX_DFT"}

def generate_gjf_file_from_input():
    molecules = input("Enter a list of molecules separated by commas: ").split(",")
    molecules = [molecule.strip() for molecule in molecules]
    for_research = input("Is this for research purposes? (y/n): ").strip().lower() == "y"
    user = None
    dft_folder = None
    default = True
    if for_research:
        experiment_id = input("Enter the experiment ID: ").strip()
        default = input("Default user? (y/n): ").strip().lower() == "y"
    if default or not for_research:
        user = default_user["uniqname"]
        dft_folder = default_user["DFT_folder"]
    else:
        user, dft_folder = input("Enter your uniqname and DFT folder name separated by a comma: ").replace(" ", "").split(",")
    correction_geo = input("Do you want to apply geometric correction? (y/n): ").strip().lower() == "y"
    #molecules = ["ammonia"]
    #iteration = input("Enter an iteration number: ").strip()
    iteration = 0
    valid_df = pcd(molecules, iteration, validate = False, classifier = "name")
    gss(valid_df, correction_geo, experiment_id if for_research else None, son = False, user = user, dft_folder = dft_folder)

def generate_gjf_file_from_random_molecules():
    samples = int(input("Enter the number of random molecules to generate: ").strip())
    CIDs = random.sample(range(1, 100000), samples)  # Generate samples random CIDs
    for_research = input("Is this for research purposes? (y/n): ").strip().lower() == "y"
    son = False
    ID = None
    if for_research:
        experiment_id = input("Enter the experiment ID: ").strip()
    else:
        son = input("Son, enter the inputson: ").strip().lower() == "y"
        ID = input("ID svp: ").strip()
    correction_geo = input("Do you want to apply geometric correction? (y/n): ").strip().lower() == "y"
    iteration = 0
    valid_df = pcd(CIDs, iteration, validate = True, classifier = "cid")
    gss(valid_df, correction_geo, ID, son, for_research)

def generate_gjf_file_from_csv():
    csv_file = input("Enter the path to the CSV file: ").strip()
    for_research = input("Is this for research purposes? (y/n): ").strip().lower() == "y"
    ID = None
    if for_research:
        ID = input("Enter the experiment ID: ").strip()
    else:
        son = input("Son, enter the inputson: ").strip().lower() == "y"
        ID = input("ID svp: ").strip()
    randomize = input("Do you want a subset of the CSV file? (Press enter if no, otherwise enter the number of molecules you want to randomly select): ").strip()
    if randomize:
        randomize = int(randomize)
    correction_geo = input("Do you want to apply geometric correction? (y/n): ").strip().lower() == "y"
    output = input("Enter the output file name (without extension): ").strip()
    # csv_file = "molecules/csv_files/PubChem_compound_MW___to_200.csv"
    # son = True
    # ID = "data_3"
    # correction_geo = False
    iteration = 0
    df = pd.read_csv(csv_file)
    valid_df = cvd(csv_file, iteration, output)
    gss_csv(valid_df, correction_geo, ID, son, randomize)


def generate_gjf_file_from_random_molecules(data_path = "molecules/csv_files/mega_data_86400.csv"):
    samples = int(input("Enter the number of random molecules to generate: ").strip())
    indexes = random.sample(range(1, 1048575), samples)  # Generate samples random CIDs
    # for_research = input("Is this for research purposes? (y/n): ").strip().lower() == "y"
    # if for_research:
    #     experiment_id = input("Enter the experiment ID: ").strip()
    # else:
    #     son = input("Son, enter the inputson: ").strip().lower() == "y"
    #     ID = input("ID svp: ").strip()
    # correction_geo = input("Do you want to apply geometric correction? (y/n): ").strip().lower() == "y"
    correction_geo = False
    iteration = 0
    for_research = False
    son = True
    ID = "AX_10k_rand_" + str(iteration)
    valid_df = cvd(data_path, iteration, "AX_10k_rand_", indexes)
    gss(valid_df, correction_geo, ID, son, None, None)
#def generate_slurm_scripts(df, correction, ID, son, user, dft_folder):

def ml_analysis(log_data_path, ID):
    pl(log_data_path, ID)


def main():
    parse = True
    if parse:
        print("Hello from dft automation project!")
        print("For now, we can only generate organic molecules. Please do not input salts, metals, or radicals.")
        choice = input("Choose an option:\n1. Generate GJF files from input molecules\n2. Generate GJF files from random molecules\n3. Generate GJF files from CSV\n Enter your choice (1-3):").strip()
        
        if choice == "1":
            generate_gjf_file_from_input()
        elif choice == "2":
            generate_gjf_file_from_random_molecules()
        elif choice == "3":
            generate_gjf_file_from_csv()
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
        generate_gjf_file_from_random_molecules()
    else:
        # df = pl("data/log_files/data_3", "data_3", "molecules/csv_files/data_3.csv")
        # df.to_csv("data/log_data/data_3.csv")
        #add descriptors
        for fs in ["base", "conj", "conj_fp"]:
            analysis("data/descriptors/test_3_extra.csv", "data_3", fs)
        print("Analysis done!")
        frames = []
        for fs in ["base", "conj", "conj_fp"]:
            t = pd.read_csv(f"results/data_3_{fs}_model_comparison.csv"); t["FeatureSet"] = fs
            frames.append(t)
        print(pd.concat(frames).sort_values(["Model", "FeatureSet"]).to_string(index=False))
        # analysis("data/descriptors/test_3_extra.csv", "data_3_ml")
        cv_df = cv_check("data/descriptors/test_3_extra.csv")
        print(cv_df.to_string(index=False))
        cv = KFold(n_splits=5, shuffle=True, random_state=42)    # use the same settings as your other models
        scores = -cross_val_score(DummyRegressor(strategy="mean"), X, y, cv=cv,
                                scoring="neg_mean_absolute_error")
        print(f"Mean predictor MAE = {scores.mean():.3f} ± {scores.std():.3f} eV")
    # generate_gjf_file_from_input()
    # generate_gjf_file_from_random_molecules()
    #generate_gjf_file_from_csv()

if __name__ == "__main__":
    main()
