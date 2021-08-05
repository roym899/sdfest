# Render evaluation, normalized size, cup
python -m sdf_estimation.scripts.rendering_evaluation \
    --data_path ./data/modelnet/cup/ \
    --config sdf_estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_mug.yaml \
    --out_folder ./results \
    --run_name modelnet_cup

# Render evaluation, normalized size, bowl
python -m sdf_estimation.scripts.rendering_evaluation \
    --data_path ./data/modelnet/bowl/ \
    --config sdf_estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_bowl.yaml \
    --out_folder ./results \
    --run_name modelnet_bowl

# Render evaluation, normalized size, bottle
python -m sdf_estimation.scripts.rendering_evaluation \
    --data_path ./data/modelnet/bottle/ \
    --config sdf_estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_bottle.yaml \
    --out_folder ./results \
    --run_name modelnet_bottle
