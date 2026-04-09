from datasets import load_dataset, Dataset, DatasetDict
from vllm import LLM, SamplingParams
import argparse
import json
from rewards import eval_answer_reward
# import torch
import re

SYSTEM_PROMPT = "You are a helpful and precise assistant with expertise in Quantum Computing."

def format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    matches = re.match(pattern, completion)
    rewards = 1.0 if matches else 0.0 
    return rewards


def create_dataset(dataset_name):
    dataset = load_dataset(dataset_name, split='test')

    def make_conversation(example):
        return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }

    dataset = dataset.map(make_conversation)

    def make_solution(example):
        if "solution" in example:
            example["solution"] = example["solution"]
        elif "answer" in example:
            example["solution"] = example["answer"]
        return example

    dataset = dataset.map(make_solution)

    def make_latex(example):
        if example.get("verifier", None) and ( "code" in example.get("verifier", None) or "general" in example.get("verifier", None)):
            pass
        else:
            example["solution"] = '$' + str(example["solution"]) + '$'
        return example

    dataset = dataset.map(make_latex)
        
    return dataset


def vllm_generate(model_name, output_name, dataset_name, num_gpus, max_output_tokens):

    # evaluation dataset
    dataset = create_dataset(dataset_name)
    print(dataset[0])

    solutions = []
    prompts = []
    processes = []
    problems = []
    verifiers = []
    count = 0
    for data in dataset:
        solutions.append(data['solution'])
        prompts.append(data['prompt'])
        problems.append(data['problem'])
        processes.append(data['process']) if 'process' in data else processes.append('')
        verifiers.append(data['verifier']) if 'verifier' in data else verifiers.append(None)
        count += 1
        if count >= args.num_samples:
            break
    
    print(f"temperature: {args.temperature}, num_generation: {args.num_generation}")
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=args.temperature,
                                     max_tokens=max_output_tokens,
                                     n = args.num_generation,
                                     )
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=num_gpus,  # number of gpu
                gpu_memory_utilization=0.7,  # prevent OOM
                trust_remote_code=True,
                distributed_executor_backend='mp',
              )

    # # vllm generation
    outputs = llm.chat(prompts,
                           sampling_params,
                           chat_template_kwargs={'enable_thinking': args.enable_thinking})
    acc_scores = []
    format_scores = []
    result_all = []
    total_acc = 0
    total_format = 0

    completions_for_rewards = []
    problems_for_rewards = []
    solutions_for_rewards = []
    processes_for_rewards = []
    verifiers_for_rewards = []
    for output, gold_solution, gold_process, problem, verifier in zip (outputs, solutions, processes, problems, verifiers):

        for output_completion in output.outputs:
            completion = output_completion.text
            completions_for_rewards.append(completion)
            problems_for_rewards.append(problem)
            solutions_for_rewards.append(gold_solution)
            processes_for_rewards.append(gold_process)
            verifiers_for_rewards.append(verifier)

    accuracy_rewards = eval_answer_reward(completions = completions_for_rewards,
                                    problems = problems_for_rewards,
                                    solutions = solutions_for_rewards,
                                    verifiers = verifiers_for_rewards,)
    
    for idx in range(len(completions_for_rewards)):
        completion = completions_for_rewards[idx]
        problem = problems_for_rewards[idx]
        gold_solution = solutions_for_rewards[idx]
        gold_process = processes_for_rewards[idx]
        verifier = verifiers_for_rewards[idx]

        acc_score = accuracy_rewards[idx]
        acc_scores.append(acc_score)
        total_acc += acc_score

        format_score = format_reward(completion)
        format_scores.append(format_score)
        total_format += format_score

        result_all.append({
            'problem': problem,
            'gold_solution': gold_solution,
            'gold_process': gold_process,
            'verifier': verifier,
            'completion': completion,
            'acc_score': acc_score,
            'format_score': format_score,
        })

    print('='*100)
    print('eval num: ', len(acc_scores))
    print('eval acc: ', total_acc / len(acc_scores))
    print('eval format: ',total_format / len(format_scores))

    current_result_file = output_name + '.json'
    with open(current_result_file, 'w', encoding='utf-8') as file:
        json.dump(result_all, file, ensure_ascii=False, indent=4)
        
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',  type=str, default='', required=True,
                        help='model name path')
    parser.add_argument('--output_name', type=str, default='', required=True,
                        help='output path')
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceH4/MATH-500', required=True,
                        help='dataset path')
    parser.add_argument('--max_output_tokens', type=int, default=4096,
                        help='generation tokens')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='generation tokens')
    parser.add_argument('--enable_thinking', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='whether to enable thinking mode for qwen3')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature')
    parser.add_argument('--num_generation', type=int, default=1,
                        help='number of responses generated for each prompt')
    parser.add_argument('--num_samples', type=int, default=99999999,
                        help='number of samples to evaluate')
    args = parser.parse_args()
    print(args)

    vllm_generate(args.model_name,
                  args.output_name,
                  args.dataset_name,
                  args.num_gpus,
                  args.max_output_tokens,)