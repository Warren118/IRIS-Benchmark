#!/bin/bash

# =====================================================================================
# ARES Classifier - Weight Download Script
#
# This script automatically downloads all the necessary pre-trained weight files for the
# ARES classifier from the Hugging Face Hub.
#
# Usage:
# 1. Make sure you have huggingface_hub installed: pip install 'huggingface_hub[cli]'
# 2. IMPORTANT: Modify the HF_REPO_ID variable below to your repository.
# 3. In your terminal, navigate to the ares_classifier/pretrained/ directory.
# 4. Run the script: bash download_weights.sh
# =====================================================================================

# --- IMPORTANT: Modify this to point to your Hugging Face repository ---
# Example: "my-username/iris-ares-weights"
HF_REPO_ID="warrenlvlmgo/IRIS_Benchmark"


# --- List of weight files to download ---
# This list must be consistent with the paths in ares_classifier/scripts/predict.py
FILES_TO_DOWNLOAD=(
  "l1_ag_clip.pth"
  "l1_ag_convnext.pth"
  "l1_ag_dino.pth"
  "l1_st_clip_features.pth"
  "l1_st_convnext_features.pth"
  "l1_st_dino_features.pth"
  "router_stage1_ag.pth"
  "router_stage2_ag.joblib"
  "st_fusion_regressor.pt"
)

# --- Main Download Logic ---
echo "================================================="
echo "Starting ARES weight download from Hugging Face Hub..."
echo "Repository: $HF_REPO_ID"
echo "================================================="

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null
then
    echo "Error: 'huggingface-cli' command not found."
    echo "Please install it first: pip install 'huggingface_hub[cli]'"
    exit 1
fi

# Check if the user has modified the placeholder repository ID
if [[ $HF_REPO_ID == "YOUR_USERNAME/YOUR_REPO_NAME" ]]; then
    echo "Error: Please edit this script and replace the HF_REPO_ID variable with your actual Hugging Face repository ID."
    exit 1
fi


# Loop through and download each file
for filename in "${FILES_TO_DOWNLOAD[@]}"
do
  echo ""
  echo "Downloading: $filename ..."
  
  # Use huggingface-cli to download the file
  # --repo-type model: Assumes you created a "Model" type repository on the Hub.
  # --local-dir .: Downloads the file to the current directory (pretrained/).
  # --local-dir-use-symlinks False: Ensures the actual file is downloaded, not a symlink.
  huggingface-cli download $HF_REPO_ID $filename --repo-type model --local-dir . --local-dir-use-symlinks False
  
  if [ $? -eq 0 ]; then
    echo "✅ Download successful: $filename"
  else
    echo "❌ Download failed: $filename"
    echo "Please check if your repository ID is correct and if the file '$filename' exists in the repository."
    exit 1
  fi
done

echo ""
echo "================================================="
echo "🎉 All weight files have been successfully downloaded!"
echo "================================================="

exit 0

