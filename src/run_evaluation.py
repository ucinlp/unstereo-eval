"""Script to evaluate the log-probabilities of the templates."""
import argparse
import traceback, glob
import pandas as pd

from templates import compute_templates_logprobs
from model_utils import load_model


# Mapping from gender to the placeholders infills
# (we replace occurrences of the KEYS {pronoun},
# {pronoun1} with the corresponding value.)
TEMPLATE_INFILS = {
    "male": {
        "{pronoun}": "he",
        "{pronoun1}": "his",
        "{pronoun2}": "him",
    },
    "female": {
        "{pronoun}": "she",
        "{pronoun1}": "her",
        "{pronoun2}": "her",
    },
}


def get_output_name(path: str, suffix: str, sep=".") -> str:
    base_name, _, ext = path.rpartition(sep)
    return f"{base_name}{suffix}{sep}{ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=f"EleutherAI/pythia-70m", type=str)
    parser.add_argument("--model_revision", default=None, type=str)

    parser.add_argument("--input_dir", default="..", type=str)
    parser.add_argument("--filename", default=None, type=str)
    args = parser.parse_args()

    kwargs = (
        {"revision": args.model_revision} if args.model_revision is not None else {}
    )
    print(kwargs, args.model_revision)
    MODEL_FILENAME, MODEL, TOKENIZER, _ = load_model(args.model_name, **kwargs)
    print(MODEL_FILENAME)

    if args.filename is None:
        FILENAMES = glob.glob(f"{args.input_dir}/step3_filter_is_likely__*.csv")
    else:
        FILENAMES = [args.filename]

    # To avoid recomputing in consecutive calls for different models
    FILENAMES = [f for f in FILENAMES if "__scores__" not in f]

    print("Found", FILENAMES)
    for input_path in FILENAMES:
        try:
            templates = pd.read_csv(input_path, index_col=0)
            assert len(templates) == templates.index.nunique()
            print("In:", input_path, len(templates))

            output_filepath = get_output_name(input_path, f"__scores__{MODEL_FILENAME}")
            assert input_path != output_filepath

            kwargs = {"model": MODEL, "tokenizer": TOKENIZER}
            results = templates.copy()

            male_results = compute_templates_logprobs(
                results["template"].values, MODEL, TOKENIZER, TEMPLATE_INFILS["male"]
            )
            results["M_num_tokens"] = male_results["num_tokens"]
            results["M_logprob"] = male_results["logprobs"]
            results["M_template"] = male_results["templates"]

            female_results = compute_templates_logprobs(
                results["template"].values, MODEL, TOKENIZER, TEMPLATE_INFILS["female"]
            )
            results["F_num_tokens"] = female_results["num_tokens"]
            results["F_logprob"] = female_results["logprobs"]
            results["F_template"] = female_results["templates"]

            results["FM_logprob"] = results["F_logprob"] - results["M_logprob"]
            results["model"] = MODEL_FILENAME

            print("Out:", output_filepath)
            results.to_csv(output_filepath, index=None)
        except:
            traceback.print_exc()
            pass
