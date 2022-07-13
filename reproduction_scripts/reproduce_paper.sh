# Reproduce preprocessing
source ./preprocess_shapenet.sh

# Train VAEs
source ./train_vaes.sh

# Train init networks
source ./train_init_networks.sh

# Evaluation
## Render evaluation, NodeSLAM setup, mug (on ShapeNet)
python -m sdfest.estimation.scripts.rendering_evaluation \
    --data_path ./data/shapenet/03797390 \
    --config estimation/configs/rendering_evaluation.yaml \
    --out_folder ./results \
    --run_name nodeslam

## Ablation study (on ShapeNet)
python -m sdfest.estimation.scripts.rendering_evaluation \
    --config estimation/configs/ablation_study.yaml \
    --out_folder ./results \
    --data_path data/shapenet/03797390 \
    --device cuda:0

## Runtime analyis (on Redwood)
python -m sdfest.estimation.scripts.real_data \
    --config estimation/configs/runtime_analysis.yaml \
    --out_folder ./results

## REAL275 (using cpas_toolbox)
# coming soon, all code available in icaps_eval branch

## REDWOOD75 (using cpas_toolbox)
# coming soon, all code available in icaps_eval branch
