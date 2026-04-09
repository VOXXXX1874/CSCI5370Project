import json
import argparse

# A program to merge the chunks into one file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin_index", type=int, default=100)
    parser.add_argument("--end_index", type=int, default=1600)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--input_dir", type=str, default="data/ReinforcementLearning/raw_materials", help="The directory of the input chunks")
    parser.add_argument("--output_dir", type=str, default="data/ReinforcementLearning/raw_materials", help="The directory of the output merged file")
    args = parser.parse_args()

    begin_index = args.begin_index
    end_index = args.end_index
    interval = args.interval
    result_list = []

    for i in range(begin_index, end_index, interval):
        with open(f"{args.input_dir}/QC_corpus_chunk_{i}.jsonl", "r") as f:
            # Each line is a json object, we need to read it and append it to the result list
            for line in f:
                result_list.append(json.loads(line.strip()))

    with open(f"{args.output_dir}/QC_corpus_merged_{begin_index}_{end_index}.jsonl", "w") as f:
        json.dump(result_list, f, indent=4)