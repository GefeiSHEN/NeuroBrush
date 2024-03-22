#!/bin/bash

# Define variables
input_image="Data/Input/afgrl.jpg"
depth_map="Data/Input/afgrl_depth_16bit.png"
n_segments=3
data_folder="./Data"
segmenter_folder="./ImageSegmenter"
paint_transformer_folder="./PaintTransformer/inference"
model_path="${paint_transformer_folder}/model.pth"
output_dir="${data_folder}/output"
segment_output="${data_folder}/segments"
ml_output="${data_folder}/ml_output"

# Ensure output directories exist
mkdir -p "${output_dir}"
mkdir -p "${segment_output}"
mkdir -p "${ml_output}"

# Step 1: Segment the image
python "${segmenter_folder}/segmenter.py" "${input_image}" "${depth_map}" "${n_segments}" "${segment_output}"

# Step 2: Process each segment through the ML model
for segment in "${segment_output}"/*.png; do
    filename=$(basename "${segment}")
    
    # Call the ML model script
    python "${paint_transformer_folder}/inference.py" \
        --input_path "${segment}" \
        --model_path "${model_path}" \
        --output_dir "${ml_output}" \
        --brush_dir "${paint_transformer_folder}" \
        --need_animation False \
        --resize_h None \
        --resize_w None \
        --serial False \
        --K None
done

# Step 3: Stack the processed images and masks
# Assuming the ML model outputs images and masks in a specific format
# Modify the stacker.py call according to its requirements
python "${segmenter_folder}/stacker.py" "${ml_output}" "${output_dir}/final_result.png"

echo "Workflow completed. Check ${output_dir} for the final result."
