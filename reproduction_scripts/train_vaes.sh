# Train VAEs for 6 categories
python -m sdfest.vae.scripts.train --config initialization/configs/vae_models/bowl.yaml --dataset_path ./data/shapenet_processed/bowl_filtered/
python -m sdfest.vae.scripts.train --config initialization/configs/vae_models/bottle.yaml --dataset_path ./data/shapenet_processed/bottle_filtered/
python -m sdfest.vae.scripts.train --config initialization/configs/vae_models/mug.yaml --dataset_path ./data/shapenet_processed/mug_filtered/
python -m sdfest.vae.scripts.train --config initialization/configs/vae_models/camera.yaml --dataset_path ./data/shapenet_processed/camera_filtered/
python -m sdfest.vae.scripts.train --config initialization/configs/vae_models/laptop.yaml --dataset_path ./data/shapenet_processed/laptop_filtered/
python -m sdfest.vae.scripts.train --config initialization/configs/vae_models/can.yaml --dataset_path ./data/shapenet_processed/can_filtered/
