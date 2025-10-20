# -*- coding: utf-8 -*-
import pandas as pd
import os
import argparse
from tqdm import tqdm
import glob


def process_generation_results(args):
    """
    Merges ARES classifier outputs with original prompts (extracting the profession)
    to create the input file for IFS_Gen and RFS_Gen calculations.
    """
    print("Starting generation data preparation...")

    ares_files = glob.glob(os.path.join(args.raw_results_dir, "**", args.ares_report_filename), recursive=True)

    if not ares_files:
        print(f"Error: No ARES reports named '{args.ares_report_filename}' found in '{args.raw_results_dir}'.")
        return

    for ares_file_path in tqdm(ares_files, desc="Processing Models"):
        try:
            model_name = os.path.basename(os.path.dirname(ares_file_path))
            print(f"\nProcessing model: {model_name}")

            df_ares = pd.read_csv(ares_file_path)

            # --- Logic to extract profession ---
            # This assumes your image filenames are structured like 'profession_id.png',
            # e.g., 'doctor_001.png'. You MUST adapt this logic if your filenames are different.
            def extract_profession_from_filename(filename):
                if not isinstance(filename, str): return None
                return filename.split('_')[0]

            df_ares['profession'] = df_ares['image_path'].apply(extract_profession_from_filename)

            # Filter out rows where profession could not be determined
            df_ares.dropna(subset=['profession'], inplace=True)
            if df_ares.empty:
                print(f"Warning: Could not extract any professions for {model_name}. Check filename format.")
                continue

            output_dir = os.path.join(args.processed_data_dir, model_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, args.output_filename)

            df_ares.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Successfully created '{args.output_filename}' for model {model_name} at {output_path}")

        except Exception as e:
            print(f"Failed to process {ares_file_path}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Generation task metric calculations.")
    parser.add_argument('--raw_results_dir', type=str, default='raw_results/generation',
                        help="Directory with raw ARES classifier outputs.")
    parser.add_argument('--processed_data_dir', type=str, default='processed_data/generation',
                        help="Directory to save processed files.")
    parser.add_argument('--ares_report_filename', type=str, default='ares_prediction_report.csv',
                        help="Filename of the ARES output.")
    parser.add_argument('--output_filename', type=str, default='final_classified_report_with_profession.csv',
                        help="Standardized output filename.")
    args = parser.parse_args()
    process_generation_results(args)


