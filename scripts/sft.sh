ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 src/train/sft.py \
--config recipes/SFT_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/Qcen2.5-1.5B_sampling.log 2>&1

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 src/train/sft.py \
--config recipes/Distill_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/Distill-1.5B_sampling.log 2>&1