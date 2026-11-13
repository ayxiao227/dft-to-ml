"""Command-line entry point for validated molecular input generation."""
import argparse
from src.pubchem_download import download_pubchem_data
from src.csv_conversion import csv_validator
from src.structure_automation import generate_slurm_scripts_from_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--names", nargs="+")
    parser.add_argument("--id", required=True)
    parser.add_argument("--user", default="xaustin")
    parser.add_argument("--remote-folder", default="dft_ml")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    df = (csv_validator(args.csv, output=args.id) if args.csv else
          download_pubchem_data(args.names, 0, validate=True, classifier="name"))
    if df.empty:
        parser.error("No valid molecules")
    result = generate_slurm_scripts_from_csv(df, ID=args.id, randomize=args.sample,
        user=args.user, dft_folder=args.remote_folder, seed=args.seed, resume=args.resume,
        gjf_options={"nproc": args.cpus, "mem": f"{args.memory_gb}GB"})
    print(f"Generated {result.generated.sum()} of {len(result)} inputs; inspect manifest.csv")
    if not result.generated.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
