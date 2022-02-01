# Need to be in folder spd for this script to work

# Create expected folder structure
cd data
ln -s ../../../data/nocs/gts/ ./data/
ln -s ../../../data/nocs/obj_models/ ./data/
mkdir Real
cd Real
ln -s ../../../../data/nocs/real_test/ ./test
ln -s ../../../data/nocs/real_train/ ./train
cd ../..

# Preprocess data (this won't modify original data)
python -m preprocess.shape_data
python -m preprocess.pose_data
python evaluate.py --date real_test --model ./results/real/model_50.pth
