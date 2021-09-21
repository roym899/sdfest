# Quantitative evaluation on annotated Redwood data
python -m sdf_estimation.scripts.redwood_evaluation \
    --ann_path ./data/redwood_annotations/ \
    --data_path ./data/redwood/ \
    --config sdf_estimation/configs/redwood_evaluation.yaml \
    --out_folder ./results \
    --run_name redwood

# Qualtitative evaluation (will keep picking random frames)

# Runtime analyis
python -m sdf_estimation.scripts.real_data \
    --config sdf_estimation/configs/runtime_analysis.yaml \
    --out_folder ./results
