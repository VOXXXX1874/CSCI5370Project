import json
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, default="data/ReinforcementLearning/middleware/distillation3000/result.json", help="The input file of the middleware data")
    args.add_argument("--output_path", type=str, default="data/ReinforcementLearning/final_data/distillation_v1", help="The output path of the final dataset")
    args = args.parse_args()

    # Read the middleware
    with open(args.input_file, "r") as f:
        middleware_data = json.load(f)

    # go through each item in middleware and generate the distillation data
    distillation_samples = []

    scenario_distribution = {}
    difficulty_distribution = {}

    for item in middleware_data:
        problem = item["question"]
        process = item["process"]
        solution = item["answer"]
        scenario = item["scenario"]
        difficulty = item["difficulty"]

        # Generate the distillation sample
        distillation_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful and precise assistant with expertise in Quantum Computing.",
                },
                {
                    "role": "user",
                    "content": problem,
                },
                {
                    "role": "assistant",
                    "content": f"{process}\nThe final solution is: $\\boxed{{{solution}}}$",
                },
            ],
        }
        distillation_samples.append(distillation_sample)
        # Update the scenario distribution
        if scenario not in scenario_distribution:
            scenario_distribution[scenario] = 0
        scenario_distribution[scenario] += 1
        # Update the difficulty distribution
        if difficulty not in difficulty_distribution:
            difficulty_distribution[difficulty] = 0
        difficulty_distribution[difficulty] += 1

    # Partition the distillation samples into train and test sets
    total_samples = len(distillation_samples)
    train_samples = int(0.8 * total_samples)
    train_data = distillation_samples[:train_samples]
    test_data = distillation_samples[train_samples:]

    # Save the distillation samples to a json file
    with open(f"{args.output_path}/train.json", "w") as f:
        json.dump(train_data, f)
    with open(f"{args.output_path}/test.json", "w") as f:
        json.dump(test_data, f)

    print("Scenario distribution:", scenario_distribution)
    print("Difficulty distribution:", difficulty_distribution)