python -m sdfest.initialization.scripts.train --config sdfest.initialization/configs/discretized_bottle.yaml --device cuda:0 &
python -m sdfest.initialization.scripts.train --config sdfest.initialization/configs/discretized_bowl.yaml --device cuda:1 &
python -m sdfest.initialization.scripts.train --config sdfest.initialization/configs/discretized_can.yaml --device cuda:2 &
wait
python -m sdfest.initialization.scripts.train --config sdfest.initialization/configs/discretized_camera.yaml --device cuda:0 &
python -m sdfest.initialization.scripts.train --config sdfest.initialization/configs/discretized_laptop.yaml --device cuda:1 &
python -m sdfest.initialization.scripts.train --config sdfest.initialization/configs/discretized_mug.yaml --device cuda:2 &
wait

