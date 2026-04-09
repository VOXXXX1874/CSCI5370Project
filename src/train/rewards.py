"""Reward functions for GRPO training."""

import re
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from latex2sympy2_extended import NormalizationConfig
from math_verify.parser import *
from sympy import nan, zoo
from openai import AsyncOpenAI
import asyncio
import os

prompt_general_verification = """We are using Reinforcement Learning to train LLM to solve problems. However, we find that the dataset contains some questions that cannot be verified by a rule-based verifier. Your task is to verify whether the provided solution is correct for the given problem, ground truth, and the model's generated answer. You need to following these instructions:
1. If the model's answer is long but contains the correct final answer, for example "After calculation, the final answer is 42", then consider it as correct. However, never accept long answers that do not contain a clear final answer indicator like "final answer is", "###", "\\boxed{}", etc.
2. Tolerate minor calculation mistakes in the model's answer according to the problem, for example, "81414" and "814 * 10^2" can be considered equivalent if the problem requires such precision.
3. If it is a multiple-choice question but the model directly provides the answer without the letter option, you need to check whether the provided answer matches the ground truth answer. If they match, consider it as correct.
4. NEVER verify the answer based on your own calculation. You should only perform pattern matching.
5. Only output "Verification: Correct" or "Verification: Wrong" without any extra explanation or text.
"""

user_prompt_general_verification = """Problem: {problem}
Ground Truth: {ground_truth}
Model's Answer: {model_answer}
Please verify the model's answer according to the above instructions.
"""


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default

VERIFIER_MAX_CONCURRENCY = _get_env_int("XR1_VERIFIER_MAX_CONCURRENCY", 50)
LOCAL_TEST_MAX_CONCURRENCY = _get_env_int("XR1_LOCAL_TEST_MAX_CONCURRENCY", 40)
API_KEY = os.getenv("OPENAI_API_KEY", "")

def outcome_reward(answer, solution):
    try:
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            raise_on_error=True,
        )
        answer_parsed = parse(
            answer,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
            raise_on_error=True,
        )
        if len(answer_parsed) != 0 and (answer_parsed[0] == nan or answer_parsed[0] == zoo):
            return gold_parsed, 'nan', 0.0

        reward = float(verify(answer_parsed, gold_parsed, raise_on_error=True))

        return gold_parsed, answer_parsed, reward
    except TimeoutException as e:
        # Timeout during mathematical operations
        print(f"Timeout during verification: {e}")
        return None, None, 0.0
    except Exception as e:
        # Other errors
        print(f"Error during verification: {e}")
        return None, None, 0.0

def outcome_rewards_general_code(answers, solutions, problems, verifiers):
    async def _general_verifier_batch(answers, solutions, problems, verifiers):
        client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
        sem = asyncio.Semaphore(VERIFIER_MAX_CONCURRENCY)
        local_test_sem = asyncio.Semaphore(LOCAL_TEST_MAX_CONCURRENCY)

        async def score_one(ans, sol, prob, verifier):
            async with sem:
                try:
                    if "all" in verifier:
                        tmp_user_prompt = user_prompt_general_verification.format(
                            problem=prob,
                            ground_truth=sol,
                            model_answer=ans
                        )
                    else:
                        parse_result = parse(ans, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()], fallback_mode='first_match')
                        if len(parse_result) >= 2:
                            model_answer_extracted = parse_result[1]
                        else:
                            return 0.0
                        tmp_user_prompt = user_prompt_general_verification.format(
                            problem=prob,
                            ground_truth=sol,
                            model_answer=model_answer_extracted
                        )
                    # generate response
                    response = await client.chat.completions.create(
                        model = "deepseek-chat",
                        messages = [
                            {
                                "role": "system",
                                "content": prompt_general_verification
                            },
                            {
                                "role": "user",
                                "content": tmp_user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": "Verification: "
                            }
                        ],
                        stream = False,
                        temperature = 0.0,
                    )
                    # Check the response
                    output = response.choices[0].message.content.strip()
                    if "Correct" in output:
                        return 1.0
                    elif "Wrong" in output:
                        return 0.0
                    else:
                        return 0.0
                except Exception as e:
                    print(f"Error in processing problem: {prob}")
                    print(e)
                    print(f"The solution was: {sol}")
                    return 0.0

        return await asyncio.gather(*(score_one(a, s, p, v) for a, s, p, v in zip(answers, solutions, problems, verifiers)))
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_general_verifier_batch(answers, solutions, problems, verifiers))
    else:
        # already inside an event loop (e.g., notebook); offload to a dedicated thread
        return asyncio.run_coroutine_threadsafe(
            _general_verifier_batch(answers, solutions, problems, verifiers), loop
        ).result()
        
# for training
def accuracy_reward(completions, solution, silence, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    verifier = kwargs.get('verifier', [None]*len(contents))
    problem = kwargs.get('problem', [None]*len(contents))
    # Group the completions and solutions according to three different verifiers
    verifier_contents = {
        'general_code': [],
        'default': []
    }
    verifier_solutions = {
        'general_code': [],
        'default': []
    }
    verifier_problem = {
        'general_code': [],
        'default': []
    }
    verifier_verifier = {
        'general_code': [],
        'default': []
    }
    original_indices = []
    for i, (content, sol) in enumerate(zip(contents, solution)):
        if verifier[i] and ("general" in verifier[i] or "code" in verifier[i]):
            verifier_contents['general_code'].append(content)
            verifier_solutions['general_code'].append(sol)
            verifier_problem['general_code'].append("code" if verifier[i] == "code" else problem[i])
            verifier_verifier['general_code'].append(verifier[i])
            original_indices.append(('general_code', len(verifier_contents['general_code']) - 1))
        else:
            verifier_contents['default'].append(content)
            verifier_solutions['default'].append(sol)
            verifier_problem['default'].append(problem[i])
            verifier_verifier['default'].append(verifier[i])
            original_indices.append(('default', len(verifier_contents['default']) - 1))

    # Process the default verifier first
    default_rewards = []
    for content, sol in zip(verifier_contents['default'], verifier_solutions['default']):
        gold_parsed, answer_parsed, reward = outcome_reward(content, sol)
        if not silence[0]:
            print('-'*100)
            try:
                print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
            except:
                print('\nanswer_parsed:', 'NaN', '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        default_rewards.append(reward)

    # Process the general_code verifier
    if len(verifier_contents['general_code']) > 0:
        general_code_rewards = outcome_rewards_general_code(verifier_contents['general_code'], verifier_solutions['general_code'], verifier_problem['general_code'], verifier_verifier['general_code'])

    # Combine the rewards back to the original order
    general_code_idx = 0
    default_idx = 0
    for v, idx in original_indices:
        if v == 'general_code':
            rewards.append(general_code_rewards[general_code_idx])
            general_code_idx += 1
        else:
            rewards.append(default_rewards[default_idx])
            default_idx += 1

    if not silence[0]:
        print('\naccuracy rewards:', rewards)

    return rewards

def length_reward_threshold(max_length, overlong_punishment_threshold):

    def length_reward(completions, solution, silence, **kwargs):
        """Reward function that gives higher reward for shorter completions."""
        rewards = []
        completion_ids_list = kwargs.get('completion_ids_list', [None]*len(completions))
        cache_length = max_length - max_length * overlong_punishment_threshold
        for completion_ids in completion_ids_list:
            if completion_ids is None:
                rewards.append(0.0)
            elif len(completion_ids) > max_length - cache_length and len(completion_ids) <= max_length:
                reward = (max_length - cache_length - len(completion_ids)) / cache_length
                rewards.append(reward)
            elif len(completion_ids) > max_length:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        if not silence[0]:
            print('\nlength rewards:', rewards)
        return rewards

    return length_reward

# for benchmark.py
# The verifier, silence, and other parameters are passed as one element
def eval_answer_reward(completions, solutions, silence=False, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # Get the "verifier" abd "problems" from kwargs if provided
    verifiers = kwargs.get('verifiers', [None]*len(completions))
    problems = kwargs.get('problems', [None]*len(completions))
    # Convert to the input format of accuracy_reward
    formatted_completions = [[{"content": c}] for c in completions]
    rewards = accuracy_reward(completions=formatted_completions, 
                              solution = solutions, 
                              silence=[silence], 
                              verifier=verifiers, 
                              problem=problems)

    return rewards
