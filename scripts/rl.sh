ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
src/train/dapo.py \
--config recipes/DAPO_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/QC-DAPO-1.5B.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
src/train/dapo.py \
--config recipes/GDPO_student_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/QC-GDPO-1.5B.log 2>&1