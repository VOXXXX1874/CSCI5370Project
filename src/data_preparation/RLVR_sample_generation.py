import json
# Import OpenAI API
from openai import AsyncOpenAI
import asyncio
import os
import traceback
import argparse
import transformers

# Get environment variable of OPENAI_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")

SCENARIO_0 = """0.
RLVR-Scenario: Multiple-Choice Question Generation
Requirement: Generate a multiple-choice question based on a given quantum computing seed material, including one or more correct answers and several plausible distractors. It is for questions with non-parsable answer like a sentence or a paragraph."""

SCENARIO_1 = """1.
RLVR-Scenario: Question with Parsable Answer Generation
Requirement: Generate a question based on a given quantum computing seed material, where the expected answer can be easily parsed and evaluated for correctness, such as a specific numerical value or a well-defined math expression. Never treat text answer as parsable. You can make large modification or creation on the seed material to make sure the answer is parsable. For example, if it is about circuit simplification based on original circuit, you can write the circuit in qiskit and ask how many CNOT gates are there after simplification."""

SCENARIO_LIST = [SCENARIO_0, SCENARIO_1]

prompt_instruction_generation = """We are training a small language model (likely Qwen3-4B) for the starter and researcher to solve various problems related to quantum computing, for example, circuit design, mathematical derivation, and explanation.
To fully stimulate the reasoning ability of the model, we are creating a diverse set of questions with easily verifiable answers. To ensure the diversity of data, each instruction tuning sample should be from a “seed” and a “scenario”. 
1. The “seed” is the raw material, and the question and answer should be weakly related to this material. 
2. The “scenario” is the form of the question, which can be multiple-choice question or question with parsable answer.
You will be given a seed raw material, which is a document related to quantum computing. You will also be given a list of scenarios. Your task is to select the most suitable scenario for the given seed raw material, and then generate an question-answer pair. 
The question-answer pair only need to have a weak relation to the seed material (such that we can ensure diversity). The question-answer pair should be in the format of a JSON object, which contains the following fields:
- "question" (str): The question for the model to answer. It should be a clear and concise description of the problem that the model needs to solve.
- "answer" (str): The expected answer from the model. If it is scenario 0, it should be like "A,B,D". If it is scenario 1, it should be a specific numerical value or a well-defined math expression.
- "scenario" (int): The scenario that you selected for this seed raw material. It should be the index of one of the scenarios in the provided list.
- "difficulty" (int): The difficulty level of the question and answer pair. It should be an integer from 1 to 5, where 1 means very easy and 5 means very hard. The difficulty level should be determined based on the complexity of the question, as well as the expected answer.
The provided scenarios include:
"""

async def generate_questions(seed, prompt, scenario_distribution, tokenizer, result_list, exception_list, sem, client):
    async with sem:
        print(f"Processing question: {len(result_list)}")
        try:
            #scenario_probability = [1 / x for x in scenario_distribution]
            ## Sample three scenarios based on the distribution
            #sampled_scenarios = random.choices(SCENARIO_LIST, weights=scenario_probability, k=3)
            #prompt += "\n".join(sampled_scenarios)

            prompt += "\n".join(SCENARIO_LIST)

            # Measure the token length of the prompt
            seed_text = seed["text"]
            token_length = len(tokenizer.encode(seed_text))
            while token_length > 131072:
                # If the token length is greater than 131072, we need to truncate the seed
                seed_text = seed_text[:len(seed_text) // 2]
                token_length = len(tokenizer.encode(seed_text))

            seed_prompt = f"The provided seed is:\n\"\"\"{seed_text}\"\"\"\nPlease generate the question-answer pair with the most suitable scenario."
        
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
            question_answer_pair = json.loads(response.choices[0].message.content)
            result_list.append(question_answer_pair)
            chosen_scenario = question_answer_pair["scenario"]
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
    tasks = [generate_questions(seed, prompt, scenario_distribution, tokenizer, result_list, exception_list, sem, client) for seed in seed_data[:1500]]

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