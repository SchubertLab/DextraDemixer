import argparse
import sys
sys.path.append('../..')

from dextrademixer.utils import aggregate_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_ins", nargs="+", required=True)
    parser.add_argument("--f_out", required=True)
    args = parser.parse_args()
    aggregate_csv(fps=args.f_ins, output_path=args.f_out, rerun=True)
