python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_bottle.yaml --device cuda:0 &
python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_bowl.yaml --device cuda:1 &
python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_can.yaml --device cuda:2 &
wait
python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_camera.yaml --device cuda:0 &
python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_laptop.yaml --device cuda:1 &
python -m sdf_single_shot.scripts.train --config sdf_single_shot/configs/discretized_mug.yaml --device cuda:2 &
wait

