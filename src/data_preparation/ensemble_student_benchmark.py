import json
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, default="data/InstructionTuning/middleware/benchmark_v1/result.json", help="The input file of the middleware data")
    args.add_argument("--output_path", type=str, default="data/Benchmark/Student", help="The output path of the final dataset")
    args = args.parse_args()
    # Read the middleware
    with open(args.input_file, "r") as f:
        middleware_data = json.load(f)

    # go through each item in middleware and generate the instruction tuning sample
    instruction_tuning_benchmark = []

    scenario_distribution = {}
    difficulty_distribution = {}

    for item in middleware_data:
        instruction = item["instruction"]
        input = item["input"]
        output = item["output"]
        scenario = item["scenario"]
        difficulty = item["difficulty"]

        # Generate the instruction tuning sample
        instruction_tuning_benchmark_sample = {
            "problem": f"{input}\n{instruction}",
            "solution": output,
        }
        instruction_tuning_benchmark.append(instruction_tuning_benchmark_sample)
        # Update the scenario distribution
        if scenario not in scenario_distribution:
            scenario_distribution[scenario] = 0
        scenario_distribution[scenario] += 1
        # Update the difficulty distribution
        if difficulty not in difficulty_distribution:
            difficulty_distribution[difficulty] = 0
        difficulty_distribution[difficulty] += 1

    # Save the instruction tuning benchmark to a json file
    with open(f"{args.output_path}/test.json", "w") as f:
        json.dump(instruction_tuning_benchmark, f)

    print("Scenario distribution:", scenario_distribution)
    print("Difficulty distribution:", difficulty_distribution)