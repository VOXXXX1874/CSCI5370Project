CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_base/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_base/benchmark_sampling_midterm.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qcen2.5-1.5B/checkpoint-120' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_Qcen_v2/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_Qcen_v2/benchmark_sampling_midterm.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/QC-DAPO-1.5B/checkpoint-560' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/benchmark_sampling_midterm.log 2>&1