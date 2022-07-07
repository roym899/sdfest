# Reproduce preprocessing
python -m sdfest.vae.scripts.process_shapenet --inpath ./data/shapenet/03642806/ --outpath ./data/shapenet_processed/laptop_filtered/ --resolution 64 --padding 2
python -m sdfest.vae.scripts.process_shapenet --inpath ./data/shapenet/02946921/ --outpath ./data/shapenet_processed/can_filtered/ --resolution 64 --padding 2
python -m sdfest.vae.scripts.process_shapenet --inpath ./data/shapenet/02942699/ --outpath ./data/shapenet_processed/camera_filtered/ --resolution 64 --padding 2
python -m sdfest.vae.scripts.process_shapenet --inpath ./data/shapenet/03797390/ --outpath ./data/shapenet_processed/mug_filtered/ --resolution 64 --padding 2
python -m sdfest.vae.scripts.process_shapenet --inpath ./data/shapenet/02880940/ --outpath ./data/shapenet_processed/bowl_filtered/ --resolution 64 --padding 2
python -m sdfest.vae.scripts.process_shapenet --inpath ./data/shapenet/02876657/ --outpath ./data/shapenet_processed/bottle_filtered/ --resolution 64 --padding 2

# Train VAEs
python -m sdfest.vae.scripts.train --config initialization/vae_models/bowl.yaml --dataset_path ./data/shapenet_processed/bowl_filtered/
python -m sdfest.vae.scripts.train --config initialization/vae_models/bottle.yaml --dataset_path ./data/shapenet_processed/bottle_filtered/
python -m sdfest.vae.scripts.train --config initialization/vae_models/mug.yaml --dataset_path ./data/shapenet_processed/mug_filtered/
python -m sdfest.vae.scripts.train --config initialization/vae_models/camera.yaml --dataset_path ./data/shapenet_processed/camera_filtered/
python -m sdfest.vae.scripts.train --config initialization/vae_models/laptop.yaml --dataset_path ./data/shapenet_processed/laptop_filtered/
python -m sdfest.vae.scripts.train --config initialization/vae_models/can.yaml --dataset_path ./data/shapenet_processed/can_filtered/

# Train init networks
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_bottle.yaml 
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_bowl.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_can.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_camera.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_laptop.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_mug.yaml

# Train extra init networks for ablation study
# python -m sdfest.initialization.scripts.train --config initialization/configs/quaternion_mug.yaml
# python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_mug.yaml --orientation_grid_resolution 2

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
