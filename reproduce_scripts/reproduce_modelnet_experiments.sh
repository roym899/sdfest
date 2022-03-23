# Render evaluation, normalized size, cup
python -m sdfest.estimation.scripts.rendering_evaluation \
    --data_path ./data/modelnet/cup/ \
    --config sdfest.estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_mug.yaml \
    --out_folder ./results \
    --run_name modelnet_cup

# Render evaluation, normalized size, bowl
python -m sdfest.estimation.scripts.rendering_evaluation \
    --data_path ./data/modelnet/bowl/ \
    --config sdfest.estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_bowl.yaml \
    --out_folder ./results \
    --run_name modelnet_bowl

# Render evaluation, normalized size, bottle
python -m sdfest.estimation.scripts.rendering_evaluation \
    --data_path ./data/modelnet/bottle/ \
    --config sdfest.estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_bottle.yaml \
    --out_folder ./results \
    --run_name modelnet_bottle
