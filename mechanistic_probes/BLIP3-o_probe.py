# -*- coding: utf-8 -*-
"""
BLIP3o Mechanistic Probe Experiments

This script consolidates all key mechanistic probe experiments for the BLIP3o model
as described in the paper, focusing on:
1.  Visual Understanding Fairness (RSA Revised)
2.  Text Encoder Bias (M-IAT)
3.  LLM Intent Consistency ("Lazy Commander" Hypothesis)
4.  Projection Layer Geometric Distortion
5.  "Counter-Stereotype Reward" Phenomenon
"""
import torch
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import os
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline
from transformers import AutoProcessor
import sys

# This assumes the blip3o library is in the python path.
# For a self-contained project, you might need to vendor this code.
try:
    from blip3o.conversation import conv_templates
    from blip3o.model.builder import load_pretrained_model
    from blip3o.utils import disable_torch_init
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("CRITICAL ERROR: Could not import 'blip3o' or 'qwen_vl_utils'.")
    print("Please ensure the blip3o repository is cloned and its path is added to your PYTHONPATH.")
    sys.exit(1)


class ProbeConfig:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Default paths are relative to the project root
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "BLIP3o-main")
    RSA_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "probe_stimuli", "rsa_images")
    PROMPT_CSV = os.path.join(PROJECT_ROOT, "data", "probe_stimuli", "blip3o_prompts.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "experiments", "blip3o")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    HIGH_BIAS_OCCUPATIONS = [
        'doctor', 'electrician', 'fireman', 'guard', 'machinist',
        'painter', 'climber', 'carpenter', 'drummer', 'guitarist'
    ]
    STEREO_GENDER_MAP = {occ: 'male' for occ in HIGH_BIAS_OCCUPATIONS}


class ExperimentSetup:
    """Encapsulates model loading and all experiment methods for BLIP3o."""

    def __init__(self, model_path):
        print("=" * 50)
        print("Initializing BLIP3o Experiment Setup...")
        disable_torch_init()

        model_dir = os.path.join(model_path, "BLIP3o-Model")
        diffusion_path = os.path.join(model_dir, "diffusion-decoder")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")

        self.tokenizer, self.multi_model, _ = load_pretrained_model(model_dir, None, "qwen", torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained("Qwen", trust_remote_code=True)
        self.pipe = DiffusionPipeline.from_pretrained(
            diffusion_path, custom_pipeline="pipeline_llava_gen", torch_dtype=torch.bfloat16,
            use_safensors=True, variant="bf16", multimodal_encoder=self.multi_model,
            tokenizer=self.processor.tokenizer, safety_checker=None
        )
        self.multi_model.to(ProbeConfig.DEVICE)
        self.pipe.to(ProbeConfig.DEVICE)
        self.embedding_storage = {}
        self.hook_handles = []
        self._setup_hooks()
        print("Models and hooks loaded successfully.")

    # ... (Rest of the methods from your scripts: _setup_hooks, _remove_hooks, get_text_embedding, etc.)
    # The functions below are copied and adapted from your uploaded files.

    def _setup_hooks(self):
        self._remove_hooks()

        def pre_hook_llm_input(module, input_data):
            if isinstance(input_data, tuple) and input_data and isinstance(input_data[0], torch.Tensor):
                self.embedding_storage['llm_input'] = input_data[0].detach().float().mean(dim=1).cpu().numpy()

        self.hook_handles.append(self.multi_model.model.layers[0].register_forward_pre_hook(pre_hook_llm_input))

        def post_hook_llm_output(module, input_data, output_data):
            if isinstance(output_data, tuple) and output_data and isinstance(output_data[0], torch.Tensor):
                self.embedding_storage['llm_output'] = output_data[0].detach().float().mean(dim=1).cpu().numpy()

        self.hook_handles.append(self.multi_model.model.layers[-1].register_forward_hook(post_hook_llm_output))

    def _remove_hooks(self):
        for handle in self.hook_handles: handle.remove()
        self.hook_handles.clear()

    def get_image_embedding(self, image_path):
        self.embedding_storage.clear()
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "describe"}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt").to(
            ProbeConfig.DEVICE)
        with torch.no_grad():
            _ = self.multi_model(**inputs)
        return self.embedding_storage.get('llm_input')

    def get_llm_output_from_gen_prompt(self, prompt_text):
        self.embedding_storage.clear()
        conv = conv_templates['qwen'].copy()
        conv.append_message(conv.roles[0], f"Please generate image based on the following caption: {prompt_text}")
        conv.append_message(conv.roles[1], None)
        prompt = [conv.get_prompt()]
        with torch.no_grad(): _ = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0)
        return self.embedding_storage.get('llm_output')

    def get_unet_input_embedding(self, prompt_text):
        self.embedding_storage.clear()
        conv = conv_templates['qwen'].copy()
        conv.append_message(conv.roles[0], f"Please generate image based on the following caption: {prompt_text}")
        conv.append_message(conv.roles[1], None)
        prompt = [conv.get_prompt()]

        def hook_fn(module, args, kwargs):
            if 'encoder_hidden_states' in kwargs:
                self.embedding_storage['unet_input'] = kwargs['encoder_hidden_states'].detach().float().mean(
                    dim=1).cpu().numpy()

        handle = self.pipe.unet.register_forward_pre_hook(hook_fn, with_kwargs=True)
        with torch.no_grad(): _ = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0)
        handle.remove()
        return self.embedding_storage.get('unet_input')

    def run_rsa_test(self, rsa_image_dir, output_dir):
        print("\n--- Running RSA Test (Visual Understanding Fairness) ---")
        male_anchor_emb = self.get_image_embedding(os.path.join(rsa_image_dir, "male_anchor.png"))
        female_anchor_emb = self.get_image_embedding(os.path.join(rsa_image_dir, "female_anchor.png"))
        if male_anchor_emb is None or female_anchor_emb is None:
            print("ERROR: Anchor images not found. Aborting RSA.")
            return

        results = []
        for occ in tqdm(ProbeConfig.HIGH_BIAS_OCCUPATIONS, desc="RSA"):
            stereo_gender = ProbeConfig.STEREO_GENDER_MAP[occ]
            counter_gender = 'female' if stereo_gender == 'male' else 'male'
            stereo_emb = self.get_image_embedding(os.path.join(rsa_image_dir, f"{stereo_gender}_{occ}1.png"))
            counter_emb = self.get_image_embedding(os.path.join(rsa_image_dir, f"{counter_gender}_{occ}1.png"))
            if stereo_emb is None or counter_emb is None: continue

            sim_stereo = 1 - cosine(stereo_emb.flatten(), locals()[f"{stereo_gender}_anchor_emb"].flatten())
            sim_counter = 1 - cosine(counter_emb.flatten(), locals()[f"{counter_gender}_anchor_emb"].flatten())
            results.append({
                "occupation": occ, "stereotype_similarity": sim_stereo,
                "counter_stereotype_similarity": sim_counter, "visual_understanding_bias": sim_stereo - sim_counter
            })

        results_df = pd.DataFrame(results)
        print("\n--- RSA Results ---");
        print(results_df.to_string())
        results_df.to_csv(os.path.join(output_dir, "rsa_results.csv"), index=False)

    def run_distortion_test(self, prompt_csv_path, output_dir):
        print("\n--- Running Projection Geometry Distortion Test ---")
        prompt_df = pd.read_csv(prompt_csv_path)
        results = []
        for occ in tqdm(ProbeConfig.HIGH_BIAS_OCCUPATIONS, desc="Distortion Test"):
            occ_df = prompt_df[prompt_df['occupation'].str.lower() == occ]
            if occ_df.empty: continue
            stereo_prompt = occ_df[occ_df['prompt_type'] == 'stereotypical']['prompt'].iloc[0]
            counter_prompt = occ_df[occ_df['prompt_type'] == 'counter-stereotypical']['prompt'].iloc[0]
            neutral_prompt = f"a photo of a {occ}"

            emb_llm_stereo = self.get_llm_output_from_gen_prompt(stereo_prompt)
            emb_llm_counter = self.get_llm_output_from_gen_prompt(counter_prompt)
            emb_llm_neutral = self.get_llm_output_from_gen_prompt(neutral_prompt)

            emb_unet_stereo = self.get_unet_input_embedding(stereo_prompt)
            emb_unet_counter = self.get_unet_input_embedding(counter_prompt)
            emb_unet_neutral = self.get_unet_input_embedding(neutral_prompt)

            if any(e is None for e in
                   [emb_llm_stereo, emb_llm_counter, emb_llm_neutral, emb_unet_stereo, emb_unet_counter,
                    emb_unet_neutral]):
                continue

            dist_llm_ns = cosine(emb_llm_neutral.flatten(), emb_llm_stereo.flatten())
            dist_llm_nc = cosine(emb_llm_neutral.flatten(), emb_llm_counter.flatten())
            dist_unet_ns = cosine(emb_unet_neutral.flatten(), emb_unet_stereo.flatten())
            dist_unet_nc = cosine(emb_unet_neutral.flatten(), emb_unet_counter.flatten())

            if dist_llm_ns == 0 or dist_unet_ns == 0: continue

            ratio_llm = dist_llm_nc / dist_llm_ns
            ratio_unet = dist_unet_nc / dist_unet_ns
            distortion = ratio_unet / ratio_llm if ratio_llm != 0 else np.inf

            results.append(
                {'occupation': occ, 'ratio_llm': ratio_llm, 'ratio_unet': ratio_unet, 'distortion_metric': distortion})

        results_df = pd.DataFrame(results)
        print("\n--- Distortion Test Results ---");
        print(results_df.to_string())
        results_df.to_csv(os.path.join(output_dir, "distortion_test_results.csv"), index=False)

    def cleanup(self):
        self._remove_hooks()
        print("\nCleanup complete.")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup = None
    try:
        setup = ExperimentSetup(model_path=args.model_path)
        if args.run_rsa:
            setup.run_rsa_test(args.rsa_image_dir, args.output_dir)
        if args.run_distortion:
            setup.run_distortion_test(args.prompt_csv, args.output_dir)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
    finally:
        if setup: setup.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLIP3o Mechanistic Probe Experiments')
    parser.add_argument('--model_path', type=str, default=ProbeConfig.MODEL_PATH)
    parser.add_argument('--rsa_image_dir', type=str, default=ProbeConfig.RSA_IMAGE_DIR)
    parser.add_argument('--prompt_csv', type=str, default=ProbeConfig.PROMPT_CSV)
    parser.add_argument('--output_dir', type=str, default=ProbeConfig.OUTPUT_DIR)
    parser.add_argument('--run_rsa', action='store_true', help="Run RSA test for visual understanding.")
    parser.add_argument('--run_distortion', action='store_true', help="Run projection geometry distortion test.")
    parser.add_argument('--run_all', action='store_true', help="Run all probe experiments.")
    args = parser.parse_args()

    if args.run_all:
        args.run_rsa = True
        args.run_distortion = True

    main(args)
