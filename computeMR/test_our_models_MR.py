import argparse
import os
import os.path as osp
import json
import time

from tools.cityPerson.eval_demo import validate, validate_with_output

prefix = "./"

predict_file_path = "predict.json"
ground_truth_path = "./computeMR/val_gt.json"


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test detector")
    parser.add_argument(
        "--location", help="location of output result.json file to be evaluated"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # prefix, postfix = args.location.split('.')
    # /*** validate using generated results file from specified location ***/
    with open(prefix + "_output_results.txt", "w") as f1:
        MRs = validate_with_output(ground_truth_path, predict_file_path, f1)
        print(
            "\nSummary: [Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]"
            % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100),
            file=f1,
        )
    f1.close()


if __name__ == "__main__":
    main()
