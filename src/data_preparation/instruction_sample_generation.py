import json
# Import OpenAI API
from openai import AsyncOpenAI
import asyncio
import os
import random
import traceback
import transformers
import argparse

# Get environment variable of OPENAI_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")

SCENARIO_0 = """0.
Meta-Scenario: Implementation Synthesis
User Intent: Convert a mathematical description or a logical flow into executable code (Qiskit, Cirq, etc.)."""

SCENARIO_1 = """1.
Meta-Scenario: Formal Derivation
User Intent: Provide a rigorous, step-by-step mathematical proof for a given quantum state or operation."""

SCENARIO_2 = """2.
Meta-Scenario: Conceptual Deconstruction
User Intent: Explain a highly technical concept to a specific audience (e.g., a beginner vs. a PhD candidate)."""

SCENARIO_3 = """3.
Meta-Scenario: Circuit Optimization
User Intent: Rewrite a given circuit to reduce gate depth, width, or to fit a specific hardware topology (e.g., linear array)."""

SCENARIO_4 = """4.
Meta-Scenario: Question Answering
User Intent: Answer a specific question about quantum computing based on the provided text."""

SCENARIO_5 = """5.
Meta-Scenario: Debugging & Troubleshooting
User Intent: Identify why a piece of quantum code is failing or why the output statevector is unexpected."""

SCENARIO_6 = """6.
Meta-Scenario: Classical-to-Quantum Mapping
User Intent: Formulate a classical problem (like Max-Cut or a specific optimization) into a quantum Hamiltonian or circuit."""

SCENARIO_7 = """7.
Meta-Scenario: Research Synthesis
User Intent: Extract key contributions, limitations, and future work from a technical passage."""

SCENARIO_8 = """8.
Meta-Scenario: Predictive Execution
User Intent: Predict the final statevector or the probability distribution of measurement results without running a simulator."""

SCENARIO_9 = """9.
Meta-Scenario: Comparative Analysis
User Intent: Compare two quantum algorithms or hardware approaches (e.g., Transmon vs. Trapped Ion) based on the text."""

SCENARIO_LIST = [SCENARIO_0, SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5, SCENARIO_6, SCENARIO_7, SCENARIO_8, SCENARIO_9]

prompt_instruction_generation = """We are training a small language model (likely Qwen3-4B) for the starter and researcher to solve various problems related to quantum computing, for example, circuit design, mathematical derivation, and explanation.
To fully shape the distribution of instruct model/base model, we first craft an instruction tuning dataset to cover different possible scenarios that can happen in the QC. To ensure the diversity of data, each instruction tuning sample should be from a “seed” and a “scenario”. 
1. The “seed” is the raw material, and the instruction tuning sample should be weakly related to this material. 
2. The “scenario” is the main topic of instruction, including designing circuit, explaining concept, answering question, and so on.
You will be given a seed raw material, which is a document related to quantum computing. You will also be given a list of scenarios. Your task is to select the most suitable scenario for the given seed raw material, and then generate an instruction tuning sample. 
The instruction tuning sample only need to have a weak relation to the seed material (such that we can ensure diversity). The instruction tuning sample should be in the format of a JSON object, which contains the following fields:
- "instruction" (str): The instruction for the model to follow. It should be a clear and concise description of the task that the model needs to perform.
- "input" (str): The input for the model to process. It should be a relevant and informative part of the seed raw material that can help the model to complete the task.
- "output" (str): The expected output from the model. It should be a correct and complete answer to the instruction, based on the input and the seed raw material.
- "scenario" (int): The scenario that you selected for this seed raw material. It should be the index of one of the scenarios in the provided list.
- "difficulty" (int): The difficulty level of the instruction tuning sample. It should be an integer from 1 to 5, where 1 means very easy and 5 means very hard. The difficulty level should be determined based on the complexity of the instruction and the input, as well as the expected output.
The provided scenarios include:
"""

async def generate_questions(seed, prompt, scenario_distribution, tokenizer, result_list, exception_list, sem, client):
    async with sem:
        print(f"Processing question: {len(result_list)}")
        try:
            scenario_probability = [1 / x for x in scenario_distribution]
            # Sample three scenarios based on the distribution
            sampled_scenarios = random.choices(SCENARIO_LIST, weights=scenario_probability, k=3)
            prompt += "\n".join(sampled_scenarios)

            # Measure the token length of the prompt
            seed_text = seed["text"]
            token_length = len(tokenizer.encode(seed_text))
            while token_length > 131072:
                # If the token length is greater than 131072, we need to truncate the seed
                seed_text = seed_text[:len(seed_text) // 2]
                token_length = len(tokenizer.encode(seed_text))

            seed_prompt = f"The provided seed is:\n\"\"\"{seed_text}\"\"\"\nPlease generate the instruction tuning sample with the most suitable scenario."
        
            # generate response
            response = await client.chat.completions.create(
                model = "deepseek-reasoner",
                messages = [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": seed_prompt
                    }
                ],
                stream = False,
                response_format = {
                    'type': 'json_object'
                }
            )
            # Load the response content as json
            instruction_tuning_sample = json.loads(response.choices[0].message.content)
            result_list.append(instruction_tuning_sample)
            chosen_scenario = instruction_tuning_sample["scenario"]
            scenario_distribution[chosen_scenario] += 1
            print(f"Finish processing question: {len(result_list)}")
        except Exception as e:
            print(f"Error in processing question: {len(result_list)}")
            print(traceback.format_exc())
            exception_list.append(seed)
        
# A function to generate the dataset
async def generate_dataset(input_file, output_path, prompt, scenario_distribution, tokenizer):
    # Initialize deepseek client
    client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    # Control the number of tasks running concurrently
    sem = asyncio.Semaphore(20)
    # Create a list to store the results
    result_list = []
    # Create a exception list to store the exception
    exception_list = []

    # Read the seed raw material
    with open(input_file, "r") as f:
        seed_data = json.load(f)

    print("Start generating questions")
    # Create a list of tasks to generate questions for each vid_name
    tasks = [generate_questions(seed, prompt, scenario_distribution, tokenizer, result_list, exception_list, sem, client) for seed in seed_data]

    # Run the tasks
    await asyncio.gather(*tasks)

    # Save it as result.json
    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(result_list, f)

    # Save the exception into a json file
    with open(os.path.join(output_path, "exception.json"), "w") as f:
        json.dump(exception_list, f)

if __name__ == "__main__":
    # args include tokenizer dir, input file, output file
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default="data/InstructionTuning/deepseek_tokenizer", help="The directory of the tokenizer")
    parser.add_argument("--input_file", type=str, default="data/InstructionTuning/raw_materials/QC_corpus_merged_100_3100.jsonl", help="The input file for generating instruction tuning samples")
    parser.add_argument("--output_path", type=str, default="result.json", help="The output path for saving the generated instruction tuning samples")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained( 
            args.tokenizer_dir, trust_remote_code=True
            )

    scenario_distribution = [1] * len(SCENARIO_LIST)

    # Run the async function
    asyncio.run(generate_dataset(args.input_file, args.output_path, prompt_instruction_generation, scenario_distribution, tokenizer))