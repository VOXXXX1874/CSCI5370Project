import json
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, default="data/ReinforcementLearning/middleware/sample1500/result.json", help="The input file of the middleware data")
    args.add_argument("--output_path", type=str, default="data/ReinforcementLearning/final_data/rlvr_sample1500", help="The output path of the final dataset")
    args = args.parse_args()

    # Read the middleware
    with open(args.input_file, "r") as f:
        middleware_data = json.load(f)

    # go through each item in middleware and generate the RLVR problem-solution pairs
    problem_solution_samples = []

    scenario_distribution = {}
    difficulty_distribution = {}

    for item in middleware_data:
        problem = item["question"]
        solution = item["answer"]
        scenario = item["scenario"]
        difficulty = item["difficulty"]

        # Generate the problem-solution sample
        problem_solution_sample = {
            "problem": problem,
            "solution": str(solution),
        }
        problem_solution_samples.append(problem_solution_sample)
        # Update the scenario distribution
        if scenario not in scenario_distribution:
            scenario_distribution[scenario] = 0
        scenario_distribution[scenario] += 1
        # Update the difficulty distribution
        if difficulty not in difficulty_distribution:
            difficulty_distribution[difficulty] = 0
        difficulty_distribution[difficulty] += 1

    # Partition the problem-solution samples into train and test sets
    total_samples = len(problem_solution_samples)
    train_samples = int(0.8 * total_samples)
    train_data = problem_solution_samples[:train_samples]
    test_data = problem_solution_samples[train_samples:]

    # Save the problem-solution samples to a json file
    with open(f"{args.output_path}/train.json", "w") as f:
        json.dump(train_data, f)
    with open(f"{args.output_path}/test.json", "w") as f:
        json.dump(test_data, f)

    print("Scenario distribution:", scenario_distribution)
    print("Difficulty distribution:", difficulty_distribution)