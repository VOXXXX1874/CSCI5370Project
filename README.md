# LLM QC Specialist

Course project of CSCI5370. A toy example on training LLM to be a quantum computing specialist through instruction tuning and reinforcement learning.

## Environment

### conda & pip

required: cuda >= 12.4

```bash
conda create -n rl python=3.12
conda activate rl
```

and

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Files Structure

Create ```data``` folder and ```output``` folder like:

```
| -- data
|	| -- Benchmark
|	| -- InstructionTuning
|		| -- final_data
|			| -- instruction_tuning_all_v1
|		| -- middleware
|			| -- all_v1
|		| -- raw_materials
|	| -- Reinforcement Learning
|		| -- final_data
|			| -- rlvr_sample1500
|		| -- middleware
|			| -- sample1500
|		| -- raw_materials
| -- output
|	| -- Qwen2.5-1.5B
```

## Experiment Result

### 0427

Experiment results about distillation and gdpo is updated in the same report of 0409.

### 0409

First set of experiment results that contain instruction tuning on 500 and 3000 samples. RL on 1500 samples.

https://wandb.ai/zizhechen-the-chinese-university-of-hong-kong/QC/reports/QC-Instruction-Tuning-RL--VmlldzoxNjQ2NjA2NQ?accessToken=3bqur9hmj15fcf3fvcmb1gd8838eczayfd9gnr2bgklb641jt3z61xe7aanactkg

## Instruction Tuning

We use the data from ```HuggingFaceFW/finepdfs-edu``` to generate instruction tuning samples because of its high quality and informaiton density. First we process data stream and use a heuristic filter to choose QC related materials.

```sh
python src/data_preparation/raw_material_from_finepdfsedu.py
```

The data will be chunked and saved at ```data/InstructionTuning/raw_materials```. Each chunk contain 100 rows. You can use following command to merge them into one large file before synthesize instruction tuning samples.

```sh
python src/data_preparation/merge_chunks.py --begin_index 100 --end_index 3100 --interval 100 --input_dir data/InstructionTuning/raw_materials --output_dir data/InstructionTuning/raw_materials
```

Then, we will use ```deepseek-v3.2-reasoner``` to generate the instruction tuning samples based on the raw materials. Please set your ```export OPENAI_API_KEY=``` and download deepseek tokenizer from https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip and put it in ```data/deepseek_tokenizer```. After that, you can generate the middleware of instruction tuning sample

```sh
python src/data_preparation/instruction_sample_generation.py --tokenizer_dir data/deepseek_tokenizer --input_file data/InstructionTuning/raw_materials/QC_corpus_merged_100_3100.jsonl --output_path data/InstructionTuning/middleware/all_v1
```

Finally, ensemble the middleware into training set and validation set through

```sh
python src/data_preparation/ensemble_instruction_dataset.py --input_file data/InstructionTuning/middleware/all_v1/result.json --output_path data/InstructionTuning/final_data/instruction_tuning_all_v1
```

After all the data is prepared, use following command to start instruction tuning. The minimum requirement is 4 RTX3090 GPUs.

```sh
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 src/train/sft.py \
--config recipes/SFT_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/Qcen2.5-1.5B_sampling.log 2>&1
```

## Reinforcement Learning

### DAPO

We use the data from ```HuggingFaceFW/fineweb-edu``` since it includes more samples (incomparison, the whole ```HuggingFaceFW/finepdfs-edu``` only contains 3000+ rows strongly related to QC). Same as instruction tuning, we first process the data stream and use a heuristic filter to choose QC related materials.

```sh
python src/data_preparation/raw_material_from_fineweb.py
```

The data will be chunked and saved at ```data/ReinforcementLearning/raw_materials```. Each chunk contain 100 rows. You can use following command to merge them into one large file before synthesize reinforcement learning samples.

```sh
python src/data_preparation/merge_chunks.py --begin_index 100 --end_index 1600 --interval 100 --input_dir data/ReinforcementLearning/raw_materials --output_dir data/ReinforcementLearning/raw_materials
```

Then, we will use ```deepseek-v3.2-reasoner``` to generate the RL samples based on the raw materials. Please set your ```export OPENAI_API_KEY=``` and download deepseek tokenizer from https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip and put it in ```data/deepseek_tokenizer```. After that, you can generate the middleware of instruction tuning sample

```sh
python src/data_preparation/RLVR_sample_generation.py --tokenizer_dir data/deepseek_tokenizer --input_file data/ReinforcementLearning/raw_materials/QC_corpus_merged_100_1600.jsonl --output_path data/ReinforcementLearning/middleware/sample1500
```

Finally, ensemble the middleware into training set and validation set through

```sh
python src/data_preparation/ensemble_rlvr_dataset.py --input_file data/ReinforcementLearning/middleware/sample1500/result.json --output_path data/ReinforcementLearning/final_data/rlvr_sample1500
```

After all the data is prepared, use following command to start Reinforcement Learning with DAPO algorithm. The minimum requirement is 4 RTX3090 GPUs.

```sh
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
src/train/dapo.py \
--config recipes/DAPO_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/QC-DAPO-1.5B.log 2>&1
```

### GDPO

We use the data from ```HuggingFaceFW/fineweb-edu``` in the last section and use samples indexed from 1500 to 3000 synthesize another set of GDPO samples (use BLEU, ROGUE, and DISTINCT as reward).

```sh
python src/data_preparation/merge_chunks.py --begin_index 1600 --end_index 3100 --interval 100 --input_dir data/ReinforcementLearning/raw_materials --output_dir data/ReinforcementLearning/raw_materials
```

Then, we will use ```deepseek-v3.2-reasoner``` to generate the GDPO samples based on the raw materials. Please set your ```export OPENAI_API_KEY=``` and download deepseek tokenizer from https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip and put it in ```data/deepseek_tokenizer```. After that, you can generate the middleware of GDPO sample.

```sh
python src/data_preparation/instruction_sample_generation.py --tokenizer_dir data/deepseek_tokenizer --input_file data/ReinforcementLearning/raw_materials/QC_corpus_merged_1600_3100.jsonl --output_path data/ReinforcementLearning/middleware/sample1600_3100
```

Finally, ensemble the middleware into training set and validation set through

```sh
python src/data_preparation/ensemble_instruction_dataset.py --input_file data/ReinforcementLearning/middleware/sample1600_3100/result.json --output_path data/ReinforcementLearning/final_data/rlvr_student
```

After all the data is prepared, use following command to start Reinforcement Learning with DAPO algorithm. The minimum requirement is 4 RTX3090 GPUs.

```sh
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=4 \
src/train/dapo.py \
--config recipes/GDPO_student_1.5B_config.yaml \
> ./output/Qwen2.5-1.5B/QC-GDPO-1.5B.log 2>&1
```

## Benchmarking

We scan the sample midterm exam questions as our benchmark to verify the effectiveness of our training, and some of the questions are rewrite into forms that is easy to parse and verify. In case the rule-base verifier does not capture the correct answer, we also use ```deepseek-v3.2``` to verify whether the model's response is correct. To run the benchmark with base model ```Qwen2.5-1.5B-Instruct```, please prepare one RTX3090 GPU and run following command:

```sh
### Researcher Benchmark
CUDA_VISIBLE_DEVICES=0 python src/train/benchmark.py \
	--model_name='Qwen/Qwen2.5-1.5B-Instruct' \
    --dataset_name='data/Benchmark/MidtermExample' \
	--output_name='./output/Qwen2.5-1.5B/benchmark_base/result_benchmark_midterm'  \
	--temperature=0.5 \
	--max_output_tokens=4096 \
	--num_gpus=1 \
	--num_generation=40 > output/Qwen2.5-1.5B/benchmark_base/benchmark_sampling_midterm.log 2>&1

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
```

The result will be saved at ```output/Qwen2.5-1.5B/benchmark_base/```. To run the evaluation with checkpoint from instruction tuning or reinforcement learning, please modify the model name like

```sh
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
```

