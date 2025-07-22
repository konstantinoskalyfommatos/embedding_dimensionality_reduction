from argparse import ArgumentParser
from utils.embedding_precalculation import precalculate_embeddings


parser = ArgumentParser(description="Precalculate embeddings for a dataset")
parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en")
parser.add_argument("--dataset_name", type=str, default="cl-nagoya/wikisplit-pp")
parser.add_argument("--batch_size", type=int, default=2048)
args = parser.parse_args()


if __name__ == "__main__":
    precalculate_embeddings(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size
    )
