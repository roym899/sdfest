# Train init networks
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_bottle.yaml 
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_bowl.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_can.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_camera.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_laptop.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_mug.yaml

# Train extra init networks for ablation study
python -m sdfest.initialization.scripts.train --config initialization/configs/quaternion_mug.yaml
python -m sdfest.initialization.scripts.train --config initialization/configs/discretized_mug.yaml --orientation_grid_resolution 2
