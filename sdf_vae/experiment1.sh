# try different latent size
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 5 --kld_weight 1.0 --learning_rate 1e-3 --device cuda:0 &
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 10 --kld_weight 1.0 --learning_rate 1e-3 --device cuda:0 &
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 20 --kld_weight 1.0 --learning_rate 1e-3 --device cuda:1 &
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 30 --kld_weight 1.0 --learning_rate 1e-3 --device cuda:1 &
wait

# try different KL weighing
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 10 --kld_weight 0.05 --learning_rate 1e-3 --device cuda:0 &
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 10 --kld_weight 0.1 --learning_rate 1e-3 --device cuda:0 &
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 10 --kld_weight 0.2 --learning_rate 1e-3 --device cuda:1 &
python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 10 --kld_weight 0.4 --learning_rate 1e-3 --device cuda:1 &
wait

python sdf_vae/scripts/train.py --iterations 1e6 --batch_size 8 --latent_size 10 --kld_weight 0.8 --learning_rate 1e-3
