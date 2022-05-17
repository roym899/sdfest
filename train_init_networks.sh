# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_bottle.yaml --device cuda:0 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_bowl.yaml --device cuda:1 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_can.yaml --device cuda:2 &
# wait
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_camera.yaml --device cuda:0 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_laptop.yaml --device cuda:1 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/quaternion_mug.yaml --device cuda:2 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_mug.yaml --device cuda:0 --orientation_grid_resolution 2 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_mug.yaml --device cuda:1 --orientation_grid_resolution 3 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_mug.yaml --device cuda:0 --orientation_grid_resolution 4 --orientation_weight 50 &
# python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_mug.yaml --device cuda:1 --orientation_grid_resolution 4 --orientation_weight 20 &
python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_mug.yaml --device cuda:2 --orientation_grid_resolution 4 --position_weight 0.0  --scale_weight 0.0 --latent_weight 0.0 --orientation_weight 100.0
