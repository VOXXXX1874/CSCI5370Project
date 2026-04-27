### Researcher Benchmark

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_base/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_base/benchmark_sampling_midterm.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/Qcen2.5-1.5B/checkpoint-120' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_Qcen_v2/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_Qcen_v2/benchmark_sampling_midterm.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/QC-DAPO-1.5B/checkpoint-560' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/benchmark_sampling_midterm.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/QC-student-GDPO-1.5B/checkpoint-1040' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_GDPO_v1/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_QC_GDPO_v1/benchmark_sampling_midterm.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/Distill-1.5B/checkpoint-180' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_distill_v1/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_distill_v1/benchmark_sampling_midterm.log 2>&1

### Student benchmark

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='data/Benchmark/Student' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_base/result_benchmark_student'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_base/benchmark_sampling_student.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='data/Benchmark/Student_ood' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_base/result_benchmark_student_ood'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_base/benchmark_sampling_student_ood.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/Qcen2.5-1.5B/checkpoint-120' \
    --dataset_name='data/Benchmark/Student' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_Qcen_v2/result_benchmark_student'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_Qcen_v2/benchmark_sampling_student.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/Qcen2.5-1.5B/checkpoint-120' \
    --dataset_name='data/Benchmark/Student_ood' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_Qcen_v2/result_benchmark_student_ood'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_Qcen_v2/benchmark_sampling_student_ood.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/QC-DAPO-1.5B/checkpoint-560' \
    --dataset_name='data/Benchmark/Student' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/result_benchmark_student'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/benchmark_sampling_student.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/QC-DAPO-1.5B/checkpoint-560' \
    --dataset_name='data/Benchmark/Student_ood' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/result_benchmark_student_ood'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_QC_DAPO_v1/benchmark_sampling_student_ood.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/QC-student-GDPO-1.5B/checkpoint-1040' \
    --dataset_name='data/Benchmark/Student' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_GDPO_v1/result_benchmark_student'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_QC_GDPO_v1/benchmark_sampling_student.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/QC-student-GDPO-1.5B/checkpoint-1040' \
    --dataset_name='data/Benchmark/Student_ood' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_QC_GDPO_v1/result_benchmark_student_ood'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_QC_GDPO_v1/benchmark_sampling_student_ood.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/Distill-1.5B/checkpoint-180' \
    --dataset_name='data/Benchmark/Student' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_distill_v1/result_benchmark_student'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_distill_v1/benchmark_sampling_student.log 2>&1

CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='output/Qwen2.5-1.5B/Distill-1.5B/checkpoint-180' \
    --dataset_name='data/Benchmark/Student_ood' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_distill_v1/result_benchmark_student_ood'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--reward_types='text' \
	--num_generation=4 > output/Qwen2.5-1.5B/benchmark_distill_v1/benchmark_sampling_student_ood.log 2>&1