# Notebook utilities
import pandas as pd


GROUP_PAIRED_WORDLIST = [
    ("she", "he"),
    ("her", "his"),
    ("her", "him"),
    ("hers", "his"),
    ("grandmother", "grandfather"),
    ("grandma", "grandpa"),
    ("stepmother", "stepfather"),
    ("stepmom", "stepdad"),
    ("mother", "father"),
    ("mom", "dad"),
    ("aunt", "uncle"),
    ("aunts", "uncles"),
    ("mummy", "daddy"),
    ("sister", "brother"),
    ("sisters", "brothers"),
    ("daughter", "son"),
    ("daughters", "sons"),
    ("female", "male"),
    ("females", "males"),
    ("feminine", "masculine"),
    ("woman", "man"),
    ("women", "men"),
    ("madam", "sir"),
    ("matriarchy", "patriarchy"),
    ("girl", "boy"),
    ("lass", "lad"),
    ("girls", "boys"),
    ("girlfriend", "boyfriend"),
    ("girlfriends", "boyfriends"),
    ("wife", "husband"),
    ("wives", "husbands"),
    ("queen", "king"),
    ("queens", "kings"),
    ("princess", "prince"),
    ("princesses", "princes"),
    ("lady", "lord"),
    ("ladies", "lords"),
]
# unpack the previous list into female, male
FEMALE_WORDS, MALE_WORDS = zip(*GROUP_PAIRED_WORDLIST)


def canonic_model_name(model_name: str) -> str:
    if "EleutherAI__" in model_name:
        model_name = model_name.replace("EleutherAI__", "")
    elif "facebook__" in model_name:
        model_name = model_name.replace("facebook__", "")
    elif "llama" in model_name:
        ix = model_name.index("llama")
        model_name = model_name[ix:].replace("__hf_models__", "-")
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
    import re 
    val = re.search(r"(\d+(\.\d+)?)(b|B|m|M)", canonic_name)[0]
    const = 1_000 if val[-1] in ("b", "B") else 1        
    return float(val[:-1]) * const


def get_pmi_diff(df: pd.DataFrame, col1: str, col2: str, clip: int=None, missing_val: float=0, prefix_col: str="pmi__") -> pd.Series:
    """Obtains the PMI difference between columns col1 and col2. 
    
    Parameters
    ----------
    df: pandas.DataFrame
    
    col1: str
        The female word to use for computing the PMI. Should be one of the
        available suffixes in the provided dataframe's columns.
    
    col2: str
        The male word to use for computing the PMI. Should be one of the
        available suffixes in the provided dataframe's columns.
        
    clip: int, optional
        Positive integer, specifies the cap. If not specified, the pmi
        difference is only computed for words that co-occur with both
        (col1, col2). If specified, we will fill the PMI value with 0
        (ideally it would be a very negative number). You can tweak
        this value using 'missing_val'.
        
    prefix_col: str
        The prefix anteceding the col1 and col2 in the provided dataframe.
        In our files, we prefixes all columns with gendered lexicons using
        the "pmi__" prefix.
    
    Note
    ----
    To replicate the values of the paper you should pass female lexicon words
    as col1 and male lexicon words as col2.
    """
    assert f"{prefix_col}{col1}" in df.columns, f"column {col1} is undefined in dataframe"
    assert f"{prefix_col}{col2}" in df.columns, f"column {col2} is undefined in dataframe"
    
    if clip is None:
        result = df[["word", f"{prefix_col}{col1}", f"{prefix_col}{col2}"]].dropna()
    else:
        result = df[["word", f"{prefix_col}{col1}", f"{prefix_col}{col2}"]].fillna(missing_val)
        
    print(f"('{col1}', '{col2}') pmi-defined words: {len(result)}")
    result[f"pmi({col1})-pmi({col2})"] = result[f"{prefix_col}{col1}"] - result[f"{prefix_col}{col2}"]
    
    if clip is not None:
        result[f"pmi({col1})-pmi({col2})"].clip(lower=-clip, upper=clip, inplace=True)
    return result


def get_gender_pairs_matrix(gender_pmi_df: pd.DataFrame, parallel_terms: list, **kwargs):
    # dataframe with all the group pairs PMI (per word)
    # (words for which no PMI diff is define)
    pairs = gender_pmi_df[["word"]].copy().set_index("word")
    num_words = []

    for fword, mword in parallel_terms:
        try:
            # Compute the pmi difference between fword and mword
            d = get_pmi_diff(gender_pmi_df, fword, mword, **kwargs).set_index("word")
            # Rename to be easier to visualize
            d = d.rename({f"pmi({fword})-pmi({mword})": f"{fword}-{mword}"}, axis=1)
            # Number of well-defined words for each of the gender pairs
            num_words.append((f"{fword}-{mword}", len(d)))
            pairs = pairs.join(d[[f"{fword}-{mword}"]])
        except:
            print(f"! Pair ({fword}, {mword}) doesn't exist...")

    return pairs, num_words

