import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, AutoImageProcessor, CLIPProcessor, AutoTokenizer, AutoModel
import logging
from collections import Counter
import json
import joblib
import numpy as np
from sklearn.preprocessing import normalize
import warnings
import re
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time

# Import model definitions from the dedicated models file
from ares_classifier.models.definitions import (
    FusionMLP,
    DINOv2Classifier,
    ConvNeXtClassifier,
    CLIPClassifier,
    create_finetune_router_model
)

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')


# =====================================================================================
# --- 1. 全局配置 (Global Configuration) ---
# =====================================================================================
class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # 项目根目录 iris-benchmark/
    IMAGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "your_image_folder_here")  # 示例输入文件夹
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "ares_predictions")
    PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "ares_classifier", "pretrained")

    STAGE1_CONFIDENCE_THRESHOLD = 0.9
    STAGE2_AG_ESCALATION_THRESHOLD = 0.7

    SKIN_TONE_REGRESSION_MODEL_PATH = os.path.join(PRETRAINED_DIR, "st_fusion_regressor.pt")
    SKIN_TONE_THRESHOLDS = {"light_middle": 0.48, "middle_dark": 1.24}

    STAGE1_AG_ROUTER_PATH = os.path.join(PRETRAINED_DIR, "router_stage1_ag.pth")
    STAGE2_AG_ROUTER_PATH = os.path.join(PRETRAINED_DIR, "router_stage2_ag.joblib")

    L1_AG_DINO_PATH = os.path.join(PRETRAINED_DIR, "l1_ag_dino.pth")
    L1_AG_CONVNEXT_PATH = os.path.join(PRETRAINED_DIR, "l1_ag_convnext.pth")
    L1_AG_CLIP_PATH = os.path.join(PRETRAINED_DIR, "l1_ag_clip.pth")

    L1_ST_DINO_PATH = os.path.join(PRETRAINED_DIR, "l1_st_dino_features.pth")
    L1_ST_CONVNEXT_PATH = os.path.join(PRETRAINED_DIR, "l1_st_convnext_features.pth")
    L1_ST_CLIP_PATH = os.path.join(PRETRAINED_DIR, "l1_st_clip_features.pth")

    L2_HEAVYWEIGHT_MODEL_ID = "OpenGVLab/InternVL3-1B"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPERTS = ['dino', 'conv', 'clip']


# =====================================================================================
# --- 2. 日志设置 (Logging Setup) ---
# =====================================================================================
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.OUTPUT_DIR, "ares_prediction_log.txt"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================================
# --- 3. L2 重量级专家辅助函数 (L2 Heavyweight Expert Helper Functions) ---
# =====================================================================================
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio, area = float('inf'), (1, 1), width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff, best_ratio = ratio_diff, ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num), key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * target_aspect_ratio[0], image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = [resized_img.crop(((i % (target_width // image_size)) * image_size,
                                          (i // (target_width // image_size)) * image_size,
                                          ((i % (target_width // image_size)) + 1) * image_size,
                                          ((i // (target_width // image_size)) + 1) * image_size)) for i in
                        range(blocks)]
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_for_heavyweight(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(image) for image in images])


# =====================================================================================
# --- 4. 核心逻辑函数 (Core Logic Functions) ---
# =====================================================================================
AGE_GENDER_LABELS = ["young male", "young female", "middle male", "middle female", "older male", "older female"]
SKIN_TONE_REGRESSION_MAP = ['light', 'middle', 'dark']


@torch.no_grad()
def predict_with_stage1_router(router_model, image_path, device):
    try:
        transform = T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image = Image.open(image_path).convert('RGB')
        inputs = transform(image).unsqueeze(0).to(device)
        outputs = router_model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        return pred_idx.item(), confidence.item()
    except Exception as e:
        logger.warning(f"Stage 1 Router prediction failed for {os.path.basename(image_path)}: {e}")
        return None, None


@torch.no_grad()
def predict_with_l1_expert(expert_model, processor, image, device):
    try:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        logits = expert_model(inputs['pixel_values'])
        pred_idx = torch.argmax(logits, 1)
        return pred_idx.item()
    except Exception as e:
        logger.warning(f"L1 Expert prediction failed on image: {e}")
        return None


@torch.no_grad()
def run_all_l1_experts_for_ag(l1_models, l1_processors, image, device):
    results = {}
    for expert_name in Config.EXPERTS:
        model_key, proc_key = f'ag_{expert_name}', expert_name
        model, processor = l1_models[model_key], l1_processors[proc_key]
        try:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            logits = model(inputs['pixel_values'])
            probabilities = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
            label = AGE_GENDER_LABELS[pred_idx.item()]
            results[f'{model_key}_pred'] = label
            results[f'{model_key}_conf'] = confidence.item()
        except Exception as e:
            logger.warning(f"Error in run_all_l1_experts_for_ag for {model_key}: {e}")
            results[f'{model_key}_pred'], results[f'{model_key}_conf'] = "Failed", 0.0
    return results


@torch.no_grad()
def get_features_for_skin_tone_fusion(l1_models, l1_processors, image, device):
    try:
        dino_inputs = l1_processors['dino'](images=image.convert("RGB"), return_tensors="pt").to(device)
        dino_feat = l1_models['st_dino'].dinov2(pixel_values=dino_inputs['pixel_values']).pooler_output
        conv_inputs = l1_processors['conv'](images=image.convert("RGB"), return_tensors="pt").to(device)
        conv_feat = l1_models['st_conv'].convnext(pixel_values=conv_inputs['pixel_values']).pooler_output
        clip_inputs = l1_processors['clip'](images=image.convert("RGB"), return_tensors="pt").to(device)
        clip_feat = l1_models['st_clip'].clip_model.get_image_features(pixel_values=clip_inputs['pixel_values'])
        return torch.cat((dino_feat, conv_feat, clip_feat), dim=1)
    except Exception as e:
        logger.error(f"Failed to get features for skin tone fusion: {e}")
        return None


def create_meta_features_for_ag(l1_results):
    df_single_row = pd.DataFrame([l1_results])
    pred_cols = [f'ag_{expert}_pred' for expert in Config.EXPERTS]
    conf_cols = [f'ag_{expert}_conf' for expert in Config.EXPERTS]
    confidences = df_single_row[conf_cols].values
    normalized_confidences = normalize(confidences, norm='l1', axis=1)
    df_norm_conf = pd.DataFrame(normalized_confidences, index=df_single_row.index,
                                columns=[f'norm_{c}' for c in conf_cols])
    df_single_row[pred_cols] = df_single_row[pred_cols].apply(lambda x: pd.Categorical(x, categories=AGE_GENDER_LABELS))
    pred_dummies = pd.get_dummies(df_single_row[pred_cols], prefix='pred_ag')
    uniqueness = pd.DataFrame({'uniqueness': df_single_row[pred_cols].nunique(axis=1)})
    return pd.concat([pred_dummies, df_norm_conf, uniqueness], axis=1)


def predict_with_heavyweight_expert(model, tokenizer, image_path, device):
    try:
        pixel_values = load_image_for_heavyweight(image_path).to(device).to(torch.bfloat16)
        prompt = f"""<image>
Please act as an expert image analyst. Your task is to determine the age group, gender, and skin tone from the provided image.
Based on your expert analysis of the image, provide a final classification.

**Instructions:**
1.  **Age/Gender options**: {', '.join(AGE_GENDER_LABELS)}.
2.  **Skin Tone options**: {', '.join(SKIN_TONE_REGRESSION_MAP)}.
3.  Respond with a single, valid JSON object only. Do not add any explanatory text before or after the JSON.

**JSON format:**
{{
  "age_gender": "...",
  "skin_tone": "..."
}}"""
        response_text = model.chat(tokenizer, pixel_values, prompt,
                                   generation_config=dict(max_new_tokens=128, do_sample=False))
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            logger.warning(f"L2 expert did not return valid JSON for {os.path.basename(image_path)}.")
            return None
    except Exception as e:
        logger.error(f"L2 expert prediction failed for {os.path.basename(image_path)}: {e}")
        return None


# =====================================================================================
# --- 5. 主工作流 (Main Workflow) ---
# =====================================================================================
def main_workflow():
    logger.info("--- Starting ARES Classifier Workflow (Skin Tone via Regression) ---")
    logger.info("--- Phase 1: Initializing all models ---")

    # Load L1/L2 models and processors
    s1_ag_router = create_finetune_router_model(len(Config.EXPERTS)).to(Config.DEVICE)
    s1_ag_router.load_state_dict(torch.load(Config.STAGE1_AG_ROUTER_PATH, map_location=Config.DEVICE))
    s1_ag_router.eval()
    s2_ag_router = joblib.load(Config.STAGE2_AG_ROUTER_PATH)

    l1_processors = {
        'dino': AutoImageProcessor.from_pretrained("facebook/dinov2-base"),
        'conv': AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k"),
        'clip': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    }
    l1_models = {
        'ag_dino': DINOv2Classifier(len(AGE_GENDER_LABELS)).to(Config.DEVICE),
        'ag_conv': ConvNeXtClassifier(len(AGE_GENDER_LABELS)).to(Config.DEVICE),
        'st_dino': DINOv2Classifier(3, unfreeze_last_n_layers=0).to(Config.DEVICE),  # Feature extractor
        'st_conv': ConvNeXtClassifier(3, unfreeze_last_n_stages=0).to(Config.DEVICE),  # Feature extractor
    }
    l1_models['ag_dino'].load_state_dict(torch.load(Config.L1_AG_DINO_PATH, map_location=Config.DEVICE))
    l1_models['ag_conv'].load_state_dict(torch.load(Config.L1_AG_CONVNEXT_PATH, map_location=Config.DEVICE))
    l1_models['st_dino'].load_state_dict(torch.load(Config.L1_ST_DINO_PATH, map_location=Config.DEVICE))
    l1_models['st_conv'].load_state_dict(torch.load(Config.L1_ST_CONVNEXT_PATH, map_location=Config.DEVICE))

    ag_clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    l1_models['ag_clip'] = CLIPClassifier(ag_clip_base, AGE_GENDER_LABELS).to(Config.DEVICE)
    ag_checkpoint = torch.load(Config.L1_AG_CLIP_PATH, map_location=Config.DEVICE)
    l1_models['ag_clip'].load_state_dict(ag_checkpoint)
    l1_models['ag_clip'].compute_text_prototypes(l1_processors['clip'], Config.DEVICE)

    st_clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    l1_models['st_clip'] = CLIPClassifier(st_clip_base, ["light skin tone", "middle skin tone", "dark skin tone"]).to(
        Config.DEVICE)
    st_checkpoint = torch.load(Config.L1_ST_CLIP_PATH, map_location=Config.DEVICE)
    l1_models['st_clip'].load_state_dict(st_checkpoint)
    l1_models['st_clip'].compute_text_prototypes(l1_processors['clip'], Config.DEVICE)

    for m in l1_models.values(): m.eval()

    fusion_regression_model = FusionMLP().to(Config.DEVICE)
    fusion_regression_model.load_state_dict(
        torch.load(Config.SKIN_TONE_REGRESSION_MODEL_PATH, map_location=Config.DEVICE))
    fusion_regression_model.eval()

    heavyweight_model = AutoModel.from_pretrained(
        Config.L2_HEAVYWEIGHT_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(Config.DEVICE).eval()
    heavyweight_tokenizer = AutoTokenizer.from_pretrained(Config.L2_HEAVYWEIGHT_MODEL_ID, trust_remote_code=True)
    logger.info("--- All models initialized successfully ---")

    logger.info(f"--- Phase 2: Starting image processing in {Config.IMAGE_SOURCE_DIR} ---")
    if not os.path.isdir(Config.IMAGE_SOURCE_DIR):
        logger.error(f"Image source directory not found: {Config.IMAGE_SOURCE_DIR}")
        return

    all_images = [os.path.join(Config.IMAGE_SOURCE_DIR, f) for f in os.listdir(Config.IMAGE_SOURCE_DIR) if
                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    final_results = []
    start_time = time.time()

    for image_path in tqdm(all_images, desc="ARES Classification Workflow"):
        result_row = {'image_path': os.path.basename(image_path)}
        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Cannot load image {os.path.basename(image_path)}: {e}")
            result_row.update({'final_age_gender': "Image Load Failed", 'final_skin_tone': "Image Load Failed"})
            final_results.append(result_row)
            continue

        # Age/Gender Task Workflow
        ag_s1_pred_idx, ag_s1_conf = predict_with_stage1_router(s1_ag_router, image_path, Config.DEVICE)
        ag_on_fast_path = ag_s1_conf is not None and ag_s1_conf > Config.STAGE1_CONFIDENCE_THRESHOLD
        if ag_on_fast_path:
            chosen_expert = Config.EXPERTS[ag_s1_pred_idx]
            pred_idx = predict_with_l1_expert(l1_models[f'ag_{chosen_expert}'], l1_processors[chosen_expert],
                                              original_image, Config.DEVICE)
            result_row['final_age_gender'] = AGE_GENDER_LABELS[pred_idx] if pred_idx is not None else "S1 Failed"
            result_row['age_gender_source'] = f"Stage1-FastPath ({chosen_expert})"
        else:
            l1_ag_results = run_all_l1_experts_for_ag(l1_models, l1_processors, original_image, Config.DEVICE)
            meta_features_ag = create_meta_features_for_ag(l1_ag_results)
            s2_ag_router_features = s2_ag_router.get_booster().feature_names
            meta_features_ag_aligned = meta_features_ag.reindex(columns=s2_ag_router_features, fill_value=0)
            s2_ag_escalation_prob = s2_ag_router.predict_proba(meta_features_ag_aligned.values)[0][1]
            if s2_ag_escalation_prob <= Config.STAGE2_AG_ESCALATION_THRESHOLD:
                vote_options = [res for res in [l1_ag_results.get(f'ag_{exp}_pred') for exp in Config.EXPERTS] if
                                res and res != "Failed"]
                if not vote_options:
                    vote = "Vote Failed"
                else:
                    vote = Counter(vote_options).most_common(1)[0][0]
                result_row['final_age_gender'] = vote
                result_row['age_gender_source'] = "Stage2-Vote"
            else:
                heavyweight_result = predict_with_heavyweight_expert(heavyweight_model, heavyweight_tokenizer,
                                                                     image_path, Config.DEVICE)
                result_row['final_age_gender'] = heavyweight_result.get('age_gender',
                                                                        'L2 Failed') if heavyweight_result else 'L2 Failed'
                result_row['age_gender_source'] = "Stage2-Heavyweight"

        # Skin Tone Task Workflow (Regression)
        fusion_features = get_features_for_skin_tone_fusion(l1_models, l1_processors, original_image, Config.DEVICE)
        if fusion_features is not None:
            with torch.no_grad():
                regression_output = fusion_regression_model(fusion_features).item()
            result_row['skin_tone_regression_score'] = round(regression_output, 4)
            if regression_output < Config.SKIN_TONE_THRESHOLDS["light_middle"]:
                final_skin_tone = SKIN_TONE_REGRESSION_MAP[0]  # light
            elif regression_output < Config.SKIN_TONE_THRESHOLDS["middle_dark"]:
                final_skin_tone = SKIN_TONE_REGRESSION_MAP[1]  # middle
            else:
                final_skin_tone = SKIN_TONE_REGRESSION_MAP[2]  # dark
            result_row['final_skin_tone'] = final_skin_tone
            result_row['skin_tone_source'] = "RegressionMLP"
        else:
            result_row.update({'final_skin_tone': "ST Feature Extraction Failed", 'skin_tone_source': "Error"})

        final_results.append(result_row)

    total_time = time.time() - start_time
    num_images = len(all_images)
    avg_time_per_image = total_time / num_images if num_images > 0 else 0

    logger.info("--- Phase 3: Saving final report ---")
    pd.DataFrame(final_results).to_csv(os.path.join(Config.OUTPUT_DIR, "ares_prediction_report.csv"), index=False,
                                       encoding='utf-8-sig')
    logger.info(f"Workflow finished. Report saved to {Config.OUTPUT_DIR}")
    logger.info(f"--- Time Statistics ---")
    logger.info(f"Total images processed: {num_images}")
    logger.info(f"Total time taken: {total_time:.2f} seconds")
    logger.info(f"Average time per image: {avg_time_per_image:.2f} seconds")


if __name__ == "__main__":
    try:
        main_workflow()
    except Exception as e:
        logger.critical(f"A critical error occurred in the main workflow: {e}", exc_info=True)

