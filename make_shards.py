import argparse
from utils import split_dataset_shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into shards")
    parser.add_argument(
        "pkl_file", type=str, help="Input pickle file with conformations (FeNNol format)"
    )
    parser.add_argument(
       "n_shards", type=int, help="Number of shards to create"
    )

    args = parser.parse_args()

    split_dataset_shards(args.pkl_file, args.n_shards,verbose=True)