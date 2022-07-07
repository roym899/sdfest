# Render evaluation, NodeSLAM setup, mug
python -m sdfest.estimation.scripts.rendering_evaluation \
    --data_path ./data/shapenet/03797390 \
    --config sdfest/estimation/configs/rendering_evaluation.yaml \
    --out_folder ./results \
    --run_name nodeslam

# Ablation study
python -m sdfest.estimation.scripts.rendering_evaluation \
    --config sdfest/estimation/configs/ablation_study.yaml \
    --out_folder ./results/ \
    --data_path data/shapenet/03797390 \
    --device cuda:0
