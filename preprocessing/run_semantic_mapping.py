# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
import torch
import os
import glob
import argparse
from tqdm import tqdm


# --- Prerequisites ---
# 1. Run this in a Python interpreter once to download necessary data:
#    import nltk
#    nltk.download('wordnet')
#    nltk.download('omw-1.4')

class SemanticMapper:
    """A rigorous, three-tiered funnel-style semantic mapper."""

    def __init__(self, official_occupations, similarity_threshold=0.6):
        print("Initializing SemanticMapper...")
        self.official_occupations = [occ.lower() for occ in official_occupations]
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.occupation_embeddings = self.model.encode(self.official_occupations, convert_to_tensor=True)
        self.alias_map = self._generate_aliases()
        print("SemanticMapper initialized successfully.")

    def _generate_aliases(self):
        alias_map = {}
        for occupation in self.official_occupations:
            aliases = set()
            processed_occ = occupation.replace('_', ' ')
            for syn in wordnet.synsets(processed_occ):
                for lemma in syn.lemmas():
                    aliases.add(lemma.name().replace('_', ' '))
            aliases.add(processed_occ)
            alias_map[occupation] = list(aliases)
        return alias_map

    def map_occupation(self, raw_answer):
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            return "unmappable", "no_answer", 0.0

        clean_answer = raw_answer.lower().strip()

        if clean_answer in self.official_occupations:
            return clean_answer, "direct_match", 1.0

        for official_occ, aliases in self.alias_map.items():
            if clean_answer in aliases:
                return official_occ, "alias_match", 0.99

        answer_embedding = self.model.encode(clean_answer, convert_to_tensor=True)
        cosine_scores = util.cos_sim(answer_embedding, self.occupation_embeddings)
        top_score, top_idx = torch.max(cosine_scores, dim=1)
        top_score, top_idx = top_score.item(), top_idx.item()

        if top_score >= self.similarity_threshold:
            mapped_occupation = self.official_occupations[top_idx]
            return mapped_occupation, "semantic_match", top_score
        else:
            return "unmappable", "low_confidence", top_score


def main(args):
    OFFICIAL_OCCUPATIONS = [
        'astronaut', 'backpacker', 'ballplayer', 'bartender', 'basketball player', 'boatman',
        'carpenter', 'cheerleader', 'climber', 'computer user', 'craftsman', 'dancer',
        'disk jockey', 'doctor', 'drummer', 'electrician', 'farmer', 'fireman', 'flutist',
        'gardener', 'guard', 'guitarist', 'gymnast', 'hairdresser', 'horseman', 'judge',
        'laborer', 'lawman', 'lifeguard', 'machinist', 'motorcyclist', 'nurse', 'painter',
        'patient', 'prayer', 'referee', 'repairman', 'reporter', 'retailer', 'runner',
        'sculptor', 'seller', 'singer', 'skateboarder', 'soccer player', 'soldier',
        'speaker', 'student', 'teacher', 'tennis player', 'trumpeter', 'waiter'
    ]

    mapper = SemanticMapper(OFFICIAL_OCCUPATIONS, similarity_threshold=args.threshold)

    input_files = glob.glob(os.path.join(args.input_dir, "**", args.raw_vqa_filename), recursive=True)
    if not input_files:
        print(f"Error: No files named '{args.raw_vqa_filename}' found in '{args.input_dir}'.")
        return

    print(f"Found {len(input_files)} raw VQA result files to process...")

    for input_path in input_files:
        try:
            model_name = os.path.basename(os.path.dirname(input_path))
            print(f"\nProcessing model: {model_name}")
            df = pd.read_csv(input_path)

            if args.raw_answer_col not in df.columns:
                print(f"Warning: Column '{args.raw_answer_col}' not found in {input_path}. Skipping.")
                continue

            results = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Mapping answers for {model_name}"):
                raw_answer = row[args.raw_answer_col]
                mapped_occ, method, score = mapper.map_occupation(raw_answer)
                result_row = row.to_dict()
                result_row.update({
                    'mapped_occupation': mapped_occ,
                    'mapping_method': method,
                    'confidence_score': score
                })
                results.append(result_row)

            results_df = pd.DataFrame(results)

            output_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, args.output_filename)

            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Mapping complete. Results saved to: {output_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map raw VQA occupation answers to a standardized list.")
    parser.add_argument('--input_dir', type=str, default='raw_results/vqa',
                        help="Root directory containing raw model predictions.")
    parser.add_argument('--output_dir', type=str, default='processed_data/vqa',
                        help="Directory to save the processed files.")
    parser.add_argument('--raw_vqa_filename', type=str, default='raw_vqa_results.csv',
                        help="The name of the CSV file with raw text answers.")
    parser.add_argument('--raw_answer_col', type=str, default='raw_answer',
                        help="Column name containing the model's raw text answer.")
    parser.add_argument('--output_filename', type=str, default='mapped_vqa_results.csv',
                        help="Name for the output file with mapped results.")
    parser.add_argument('--threshold', type=float, default=0.6, help="Semantic similarity threshold for mapping.")
    args = parser.parse_args()
    main(args)


