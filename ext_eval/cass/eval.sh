# Reproduce CASS results
# echo "EVAL CASS ..."
python -m tools.eval --resume_model cass_best.pth --dataset_dir ./cass_data/ --cuda --save_dir ./cass_result --eval --mode cass --draw

# echo "EVAL CASS ..."
# python ./tools/eval.py --save_dir ./cass_result --mode cass --dataset_dir ../../data/nocs/ --cuda


# echo "EVAL NOCS ..."
# python ./tools/eval.py --save_dir ./cass_result --mode nocs --dataset_dir ../../data/nocs/ --cuda
