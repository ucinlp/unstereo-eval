"""This script is used to run the pipeline and create the stereotype free benchmarks."""
from openai_utils import get_completion_block_until_succeed
from templates import fill_template
from typing import List, Tuple, Dict

import argparse, warnings
import json, os, re, tqdm, yaml
import pandas as pd


def print_sep(msg: str, indent: int = 80):
    print("=" * indent)
    print(msg)
    print("=" * indent)


def read_cli_args(base_dir: str = ".."):
    """Read the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--placeholders_config",
        default=f"{base_dir}/configs/placeholders.yml",
        type=str,
        help="Contains the definition of the placeholders to use in the parse of the templates.",
    )
    parser.add_argument(
        "--exp_config",
        default=f"{base_dir}/configs/one_prompt_they.yml",
        type=str,
        help="Config file",
    )
    args = parser.parse_args()
    return args


def readlines(filepath: str) -> List[str]:
    """Read txt file where lines are separated by newlines."""
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def build_template_gen_prompt(
    attr_word: str,
    target_word: str,
    prompt: str,
    num_templates: int = 10,
    num_words: int = 20,
) -> str:
    """Create the prompt for the generation step.

    For the main paper, we set:
    - num_templates = 5
    - num_words = {5, 10, 20}
    - target_word = {"he", "she"}
    and used 493 different attribute (or seed) words.
    """
    prompt = prompt.replace("{num_templates}", str(num_templates))
    prompt = prompt.replace("{num_words}", str(num_words))
    prompt = prompt.replace("{target_word}", str(target_word))
    prompt = prompt.replace("{attr_word}", str(attr_word))
    return prompt.strip()


def parse_gen_response(response: str) -> List[str]:
    """Parse the model-generated response removing prefixes and extra whitespaces."""
    # Separate based on "\n"
    sentences = response.split("\n")
    orig_sentences = [seq for seq in sentences if seq != "" and not seq.isspace()]

    # Remove prefixed numbers (if any)
    sentences = [re.sub(r"^[0-9]{1,2}\. ", "", seq).strip() for seq in orig_sentences]
    sentences = [re.sub(r"^[0-9]{1,2}\) ", "", seq).strip() for seq in sentences]

    # This will remove sentences that did not have any number, as well as duplicates
    sentences = [seq for seq in sentences if seq not in orig_sentences]
    if len(sentences) != len(orig_sentences):
        print(orig_sentences)
    return sentences


def run_generate_step(
    target_word: str,
    input_filepath: str,
    output_dir: str,
    prompt: str,
    num_templates=10,
    num_words=20,
    **gen_kwargs,
) -> Tuple[pd.DataFrame, str]:
    """Generate the number of specified templates using the target word and a
    list of additional words, as specified in the ``input_filepath``.
    """
    print("Reading input filepath at", input_filepath)
    attr_words = readlines(input_filepath)
    attr_words = [w for w in attr_words if w is not None and w.strip() != ""]
    print("Loaded", len(attr_words), "words")

    # ------------------------------------------------------------------------
    # Run the generation step and save information about the configurations
    # the prompts, the orignal responses from the model (before any parsing).
    # ------------------------------------------------------------------------
    gen_prompts = []
    gen_responses = []
    for attr_word in tqdm.tqdm(attr_words):
        prompt_attr = build_template_gen_prompt(
            attr_word, target_word, prompt, num_templates, num_words
        )
        gen_prompts.append(prompt_attr)

        response = get_completion_block_until_succeed(prompt_attr, **gen_kwargs)
        gen_responses.append(response)

    prompt_fp = f"{output_dir}__prompts_{target_word}.csv"
    print("Persisting prompts' information:", prompt_fp)
    prompts_df = pd.DataFrame(
        {
            "word": attr_words,
            "target_word": [target_word] * len(attr_words),
            "prompt": gen_prompts,
            "response": gen_responses,
            "generation_configs": [gen_kwargs] * len(attr_words),
        }
    )
    prompts_df.to_csv(prompt_fp, index=None)

    # ------------------------------------------------------------------------
    # Parse the generated responses and save a CSV file with these
    # sentences and the crucial information.
    # ------------------------------------------------------------------------
    gen_words = []
    gen_responses_parsed = []
    for attr_word, response in zip(attr_words, gen_responses):
        response: List[str] = parse_gen_response(response)

        if len(response) != num_templates:
            warnings.warn(
                f"Insufficient number of sentences for '{attr_word}': {len(response)}"
            )
        gen_responses_parsed.extend(response)
        gen_words.extend([attr_word] * len(response))

    sentences = pd.DataFrame(
        {
            "word": gen_words,
            "target_word": [target_word] * len(gen_words),
            "sentence": gen_responses_parsed,
        }
    )

    sentences_fp = f"{output_dir}__sentences_{target_word}.csv"
    print("Dumping generated responses at", sentences_fp)
    sentences.to_csv(sentences_fp, index=False)
    return sentences, sentences_fp


def build_convert_prompt(
    sentence: str,
    prompt: str,
    from_target: str = "she/her/her",
    to_target: str = "he/his/him",
) -> str:
    """Create the conversion prompt to change the pronouns from one target to another.

    This method is particularly useful when direct mapping cannot be applied. For
    instance, consider the mapping from "her" to either "his" or "him". Doing so,
    requires context information to determine the correct mapping, e.g., "her"
    can be mapped to "his" in "her car" but to "him" in "I saw her"."""
    prompt = prompt.replace("{from_target}", from_target)
    prompt = prompt.replace("{to_target}", to_target)
    prompt = prompt.replace("{sentence}", sentence)
    return prompt.strip()


def parse_replace_placeholders(
    sentences: List[str], placeholder_remap: Dict[str, str]
) -> Tuple[List[bool], List[str]]:
    """Use the placeholder regex map to replace specified expressions (i.e.,
    the key in the dictionary) by the corresponding placholder values (i.e.,
    the value in the dictionary).

    One example of a placeholder remap is ``{"\\b((H|h)e)\\b": "{pronoun}"}``
    which implies that `"He is a doctor"` is transformed into `"{pronoun} is a
    doctor"`.
    """
    modified_sentences = []
    is_modified = []
    for s in sentences:
        modified_s = s
        for regex, replcmnt in placeholder_remap.items():
            modified_s = re.sub(regex, replcmnt, modified_s, count=1_000)
        is_modified.append(modified_s != s)
        modified_sentences.append(modified_s)
    return is_modified, modified_sentences


def detect_modifications(new: str, original: str) -> Dict[str, List[str]]:
    """Detect the modifications between the ``new`` and the ``original`` sentences."""
    if new == original:
        return None

    from nltk import word_tokenize
    from collections import Counter

    ntokens, otokens = word_tokenize(new), word_tokenize(original)
    ncounter, ocounter = Counter(ntokens), Counter(otokens)
    # TODO - Add heuristic to make sure changes are only plural and pronouns
    return (ncounter | ocounter) - (ncounter & ocounter)


def run_convert_step(
    target_word: str,
    output_dir: str,
    input_path: str = None,
    input_df: pd.DataFrame = None,
    placeholder_mapping: Dict[str, str] = None,
    exclude_targets: List[str] = ["he"],
    **convert_kwargs,
) -> Tuple[pd.DataFrame, str]:
    """Converts the target word according to a from/to target mapping encoding.

    It may be necessary to guarantee that some templates are in the same form.
    The ``exclude_targets`` parameter can be used to dictate which target_words do not
    require this conversion step. For instance, it is not necessary to apply this
    procedure to convert from male sentences to female sentences, since we can directly
    map the pronouns without requiring additional information about the context.
    However, we need to use this when mapping female sentences to male sentences,
    because "her" can be mapped to either "him" or "his" depending on the context.

    Note that if, you've run `they` or `she` before, we may skip this step.
    """
    if input_df is None and input_path is not None:
        input_df = pd.read_csv(input_path)
    elif input_df is None:
        raise ValueError("Must specify either 'input_path' or 'input_df")

    if target_word.lower() not in exclude_targets:
        sentences = input_df["sentence"].values.tolist()

        prompt_kwargs = {
            k: v
            for k, v in convert_kwargs.items()
            if k.endswith("_target") or k == "prompt"
        }
        convert_prompts = [build_convert_prompt(r, **prompt_kwargs) for r in sentences]

        prompts_df = input_df.copy()
        prompts_df["prompt"] = convert_prompts
        prompts_df["convert_kwargs"] = [convert_kwargs] * len(convert_prompts)

        if output_dir is not None:
            prompt_fp = f"{output_dir}__prompts_{target_word}.csv"
            print("Dumping prompts' information at:", prompt_fp)
            prompts_df.to_csv(prompt_fp)

        convert_responses = []
        completion_kwargs = {
            k: v for k, v in convert_kwargs.items() if k not in prompt_kwargs
        }
        for prompt in tqdm.tqdm(convert_prompts):
            response = get_completion_block_until_succeed(prompt, **completion_kwargs)
            convert_responses.append(response)

    else:  # coded thinking of "he" only (but we can also have she/THEY)
        sentences = input_df["sentence"].values.tolist()
        convert_responses = input_df["sentence"].values.tolist()

    convert_df = input_df.copy()
    mapping = placeholder_mapping
    convert_df["has_placeholder"], convert_df["template"] = parse_replace_placeholders(
        convert_responses, mapping
    )
    convert_df["modifications"] = [
        detect_modifications(nres, origres)
        for nres, origres in zip(convert_responses, sentences)
    ]

    if output_dir is None:
        return convert_df, None

    # The files with the templates, i.e., the appropriate pronouns in the sentence
    # should have been replaced by the placeholders.
    convert_df.to_csv(f"{output_dir}__templates_{target_word}.csv", index=False)
    return convert_df, f"{output_dir}__templates_{target_word}.csv"


def run_filter_step(
    target_word: str,
    input_path: str,
    output_dir: str,
    prompt: str,
    apply: List[str],
    colname: str,
    placeholder_mapping,
    templates: List[str] = None,
    **gen_kwargs,
) -> Tuple[pd.DataFrame, str]:
    """Filter the templates based on whether they are likely or unlikely.


    Originally, we used this function to filter out sentences that were
    semantically incorrect under one of the target words but not the other.
    After some thought, we observed that this was somewhat redundant with
    the PMI based filtering that we had, since these semantic inconsistencies
    can be implicitly observed through the training data. Upon analysis, we still
    find this filtering to be useful to remove ungrammatical sentences or gender
    correlations between words that do not co-occur significantly with "he" or "she".
    One of such examples was the sentence "Brad ..., {pronoun} is ...", where Brad
    is usually associated with "he" but not with "she".

    Parameters
    ----------
    target_word : str
        The target (or group) word to use in the filtering process.

    input_path: str
        The path to the CSV file containing the templates to filter.

    output_dir: str
        The directory where the filtered templates will be saved.

    prompt: str
        The prompt to use in the filtering process.

    apply: List[str]
        To apply to both female and male templates, set this to ["male", "female"].

    colname: str
        The name of the column to associate with the results of the filtering step
        in the final CSV file.

    placeholder_mapping: Dict[str, str]
        The mapping that dictates how to transform the placeholders in the `template`
        column into the corresponding gendered pronouns. It should be a dictionary
        as follows {
            "male": {"{pronoun}": "he", ...},
            "female": {"{pronoun}": "she", ...}
        }
    """
    if input_path is not None:
        df = pd.read_csv(input_path)
        templates = df["template"].values.tolist()

    results = []
    templates = templates if len(templates) < 5 else tqdm.tqdm(templates, desc="Filter")
    for template in templates:
        if "male" in apply:
            male_sentence = fill_template(template, placeholder_mapping["male"])
            mod_prompt = prompt.replace("{template}", male_sentence)
            male_pred = get_completion_block_until_succeed(mod_prompt, **gen_kwargs)
        else:
            male_pred = None

        if "female" in apply:
            female_sentence = fill_template(template, placeholder_mapping["female"])
            mod_prompt = prompt.replace("{template}", female_sentence)
            female_pred = get_completion_block_until_succeed(mod_prompt, **gen_kwargs)
        else:
            female_pred = None

        results.append({"male": male_pred, "female": female_pred})

    if output_dir is None:
        return results, None

    df[colname] = results
    filter_fp = f"{output_dir}__{target_word}.csv"
    print("Dumping filter responses to", filter_fp)
    df.to_csv(filter_fp, index=False)
    return df, filter_fp


def run_pipeline(plcholder_configs, exp_configs):
    """Run stage 2 of the benchmark pipeline.

    To run stage 1, run the Word Selection notebook and create
    a txt file with the selected words."""
    target_word = exp_configs["target_word"]
    output_dir = exp_configs["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/configs_{target_word}.yml", "wt") as f:
        yaml.dump(exp_configs, f)

    print_sep("GENERATION STEP", 40)
    df_gen, input_path = run_generate_step(
        target_word=target_word,
        output_dir=f"{output_dir}/step1_generate",
        **exp_configs["generation"],
    )

    print_sep("CONVERSION STEP", 40)
    conv_configs = exp_configs["conversion"]
    # it determines how to map the target groups to placeholders
    conv_mapping = conv_configs.get(
        "mapping", plcholder_configs["gender_to_placeholder"]
    )
    df_conv, input_path = run_convert_step(
        target_word=target_word,
        output_dir=f"{output_dir}/step2_convert",
        input_path=input_path,
        placeholder_mapping=conv_mapping,
        **conv_configs,
    )

    print_sep("FILTER STEP", 40)
    for filter_name, filter_kwargs in exp_configs["filters"].items():
        print("Applying filter:", filter_name, "with kwargs:", filter_kwargs)
        filt_mapping = filter_kwargs.get(
            "mapping", plcholder_configs["placeholder_to_gender"]
        )
        df_filt, input_path = run_filter_step(
            target_word=target_word,
            input_path=input_path,
            output_dir=f"{output_dir}/step3_filter_{filter_name}",
            placeholder_mapping=filt_mapping,
            **filter_kwargs,
        )

    print_sep("FINISHED!!!")


if __name__ == "__main__":
    args = read_cli_args()
    print_sep("Loading placeholders")
    with open(args.placeholders_config, "r") as f:
        plc_configs = json.load(f)
    print("Loaded:", plc_configs)

    print_sep("Loading experiment configs")
    with open(args.exp_config, "rt") as f:
        exp_configs = yaml.safe_load(f)
    print("Loaded:", exp_configs)

    run_pipeline(plc_configs, exp_configs)
