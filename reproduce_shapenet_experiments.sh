# # Render evaluation, normalized size, mug
# python -m sdf_estimation.scripts.rendering_evaluation \
#     --data_path ./data/shapenet/03797390 \
#     --config sdf_estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_mug.yaml \
#     --out_folder ./results \
#     --run_name shapenet_mug

# # Render evaluation, normalized size, bowl
# python -m sdf_estimation.scripts.rendering_evaluation \
#     --data_path ./data/shapenet/02880940 \
#     --config sdf_estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_bowl.yaml \
#     --out_folder ./results \
#     --run_name shapenet_bowl

# # Render evaluation, normalized size, bottle
# python -m sdf_estimation.scripts.rendering_evaluation \
#     --data_path ./data/shapenet/02876657 \
#     --config sdf_estimation/configs/rendering_evaluation_normalized_size/rendering_evaluation_normalized_bottle.yaml \
#     --out_folder ./results \
#     --run_name shapenet_bottle

# Render evaluation, NodeSLAM setup, mug
python -m sdf_estimation.scripts.rendering_evaluation \
    --data_path ./data/shapenet/03797390 \
    --config sdf_estimation/configs/rendering_evaluation.yaml \
    --out_folder ./results \
    --run_name nodeslam

# # Ablation study
# python -m sdf_estimation.scripts.rendering_evaluation \
#     --config sdf_estimation/configs/ablation_study.yaml \
#     --out_folder ./results/ \
#     --data_path data/shapenet/03797390
