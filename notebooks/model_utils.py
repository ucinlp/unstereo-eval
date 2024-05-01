import pandas as pd
import numpy as np
import re

def canonic_model_name(model_name: str) -> str:
    if "EleutherAI__" in model_name:
        model_name = model_name.replace("EleutherAI__", "")
    elif "facebook__" in model_name:
        model_name = model_name.replace("facebook__", "")
    elif "70b-hf__snapshots" in model_name:
        model_name = "llama-2-70b"
    elif "llama" in model_name:
        ix = model_name.index("llama")
        model_name = model_name[ix:].replace("__hf_models__", "-")
        model_name = model_name.replace("B", "b")
    elif "mosaicml__" in model_name:
        model_name = model_name.replace("mosaicml__", "")
    elif "allenai__" in model_name:
        model_name = model_name.replace("allenai__", "")
    elif "mistralai__" in model_name:
        model_name = model_name.replace("mistralai__", "")
    if "deduped" in model_name:
        model_name = model_name.replace("-deduped", " (D)")
    return model_name


def get_model_size(canonic_name: str) -> int:
    val = re.search(r"(\d+(\.\d+)?)(b|B|m|M)", canonic_name)[0]
    const = 1_000 if val[-1] in ("b", "B") else 1        
    return float(val[:-1]) * const
        
    
def get_model_family(model_name: str) -> str:
    """Collects information about the model family"""
    if "pythia" in model_name:
        return "pythia"
    elif "opt" in model_name:
        return "opt"
    elif "mpt" in model_name:
        return "mpt"
    elif "llama" in model_name:
        return "llama2"
    elif "gpt" in model_name:
        return "gpt-j"

    
def is_deduped(model_name: str) -> bool:
    """Collect information about whether the model was trained on deduplicated data."""
    return True if '-deduped' in model_name else False
    

def is_intervention(model_name: str) -> bool:
    """Collect information about whether the model was trained on deduplicated data 
    and with gender bias intervention.
    """
    return True if '-intervention' in model_name else False


def remove_unnatural_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out unnatural examples from the provided dataframe.
    
    Natural test sentence pairs are those for which ChatGPT
    indicates that both sentence variants (regardless of gender)
    are both likely to occur. If one of them is unlikely (as per
    ChatGPT prediction) then we will deem the whole test sentence
    pair unnatural and remove it.
    
    The proposed datasets were generated from scratch and therefore
    will be the only ones with this column. The WinoBias and Winogender
    have no such information, since we know by definition that both
    completions of the sentences are both likely.
    """
    if "is_natural" in df.columns:
        return df[df["is_natural"]].reset_index(drop=True)

    return df

def read_filepath(fp: str, dataset: str, filter_unnatural: bool) -> pd.DataFrame:
    # print(fp)
    df = pd.read_csv(fp)
    # df has "model" information, with the fully qualified name (including company name)
    
    # add dataset name to dataframe
    df["dataset"] = dataset
    # add boolean identifying whether model was trained on deduplicated data
    df["is_deduped"] = df["model"].apply(is_deduped)
    # add boolean indentifying whether model was trained with gender swap
    df["is_intervention"] = df["model"].apply(is_intervention)
    # add canonic name (no company name, with size info)
    df["orig_model_name"] = df["model"]
    df["model"] = df["model"].apply(canonic_model_name)
    # add model size (as a float)
    df["model_size"] = df["model"].apply(get_model_size)
    # add model family
    df["model_family"] = df["model"].apply(get_model_family)

    # add information about whether templates are likely or unlikely
    if filter_unnatural:
        bef = len(df)
        df = remove_unnatural_examples(df)
        print(f"Filtered {len(df) - bef} unnatural, removed from", dataset)
        if "is_natural" in df.columns:
            print("Number of unique 'likely_under' labels (should be 1):", df["likely_under"].nunique())
    df = df.reset_index(names=["orig_index"])
    return df
