# try smaller decoder architectures
python sdf_vae/scripts/train.py --config configs/default.yaml configs/decoder_1.yaml --device cuda:0 &
python sdf_vae/scripts/train.py --config configs/default.yaml configs/decoder_2.yaml --device cuda:0 &
python sdf_vae/scripts/train.py --config configs/default.yaml configs/decoder_3.yaml --device cuda:0 &
wait
