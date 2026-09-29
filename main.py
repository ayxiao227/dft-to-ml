from src.pubchem_download import download_pubchem_data as pcd
from src.structure_automation import generate_slurm_scripts as sa


def generate_gjf_file_from_input():
    molecules = input("Enter a list of molecules separated by commas: ").split(",")
    molecules = [molecule.strip() for molecule in molecules]
    for_research = input("Is this for research purposes? (y/n): ").strip().lower() == "y"
    if for_research:
        experiment_id = input("Enter the experiment ID: ").strip()
    correction_geo = input("Do you want to apply geometric correction? (y/n): ").strip().lower() == "y"
    #molecules = ["ammonia"]
    #iteration = input("Enter an iteration number: ").strip()
    iteration = 0
    valid_df = pcd(molecules, iteration, validate = False)
    sa(valid_df, correction_geo, experiment_id if for_research else None)

def main():
    print("Hello from dft automation project!")
    generate_gjf_file_from_input()

if __name__ == "__main__":
    main()
