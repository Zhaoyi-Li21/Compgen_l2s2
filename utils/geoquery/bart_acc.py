import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        required=True,
        type=str,
        help="pred results",
    )
    parser.add_argument(
        "--gold",
        help="ground truth",
    )
    args = parser.parse_args()
    pred = open(args.pred, "r")
    gold = open(args.gold)
    pred_lines = pred.readlines()
    gold_lines = gold.readlines()
    hit = 0
    for i in range(len(pred_lines)):
        pred_line = pred_lines[i].strip()
        gold_line = gold_lines[i].strip()
        if "answer (" + pred_line == gold_line:
            hit += 1
    print("pred_acc:", hit/len(pred_lines))