input_image="Data/Input/afgrl.jpg"
depth_map="Data/Input/afgrl_depth_16bit.png"
n_segments=3
expand=10                                                                                                                                                                                                                                                                                                                                                                                                           
data_folder="./Data"
segmenter_folder="./ImageSegmenter"
paint_transformer_folder="./PaintTransformer/inference"
model_path="${paint_transformer_folder}/painter500alpha50.pth"
output_dir="${data_folder}/output"
segment_output="${data_folder}/segments"
ml_output="${data_folder}/ml_output"
stacking="${data_folder}/stacking"

mkdir -p "${output_dir}"
mkdir -p "${segment_output}"
mkdir -p "${ml_output}"
mkdir -p "${stacking}"

# Step 1: Segment the image
python "${segmenter_folder}/preprocess.py" "${input_image}" "${depth_map}" "${n_segments}" "${expand}" "${segment_output}"

mapfile -t depth_values < "${segment_output}/depth.txt"
depth_index=0

# Step 2: ML inference
for segment in "${segment_output}"/*.png; do
    filename=$(basename "${segment}")
    relative_K=${depth_values[$depth_index]}
    
    python "${paint_transformer_folder}/inference.py" \
        --input_path "${segment}" \
        --model_path "${model_path}" \
        --output_dir "${ml_output}" \
        --brush_dir "${paint_transformer_folder}" \
        --need_animation False \
        --resize_h None \
        --resize_w None \
        --serial True \
        --K None \
        --relative_K "${relative_K}"
    
    ((depth_index++))
done

# Step 3: Move results
python "${segmenter_folder}/mover.py" "${ml_output}" "${segment_output}" "${stacking}"

# Step 4: Stack the processed images and masks
python "${segmenter_folder}/stacker.py" "${stacking}" "${output_dir}/final_result.png"

echo "Workflow completed. Check ${output_dir} for the final result."