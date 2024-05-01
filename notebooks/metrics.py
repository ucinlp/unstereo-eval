from collections import defaultdict
from typing import Dict, List
from sklearn.metrics import auc
import pandas as pd
import numpy as np 
import glob, operator


def filter_eta_and_count_examples(
        name_and_dataset: Dict[str, pd.DataFrame],
        etas: List[float],
        col: str,
        constant: int,
    ) -> pd.DataFrame:
    """Count the number of remaining examples after filtering every dataset in
    for different settings of $|MaxPMI(s)| \leq \eta$
    """
    results = defaultdict(list)
    
    dataset_max_counts = defaultdict(lambda: 0)
    for eta in etas:
        for dataset, df in name_and_dataset.items():
            assert df["model"].nunique() == constant

            counts = ((df[col] >= -eta) & (df[col] <= eta)).sum() / constant
            results["dataset"].append(dataset)
            results["filter"].append(eta)
            results["counts"].append(counts)
            
            if dataset_max_counts[dataset] < counts:
                dataset_max_counts[dataset] = counts
            
    results = pd.DataFrame(results)
    results["freq"] = results[["dataset", "counts"]].apply(lambda x: x["counts"]/(dataset_max_counts[x["dataset"]]), axis=1)
    
    return pd.DataFrame(results)


def use_log_10_base(ln_val: float) -> float:
    """Transforms natural log into log base 10."""
    return np.log10(np.exp(ln_val))


def compute_neutralpct_fixed_threshold(dataset: pd.DataFrame, eps: float, col: str):
    abs_col = dataset[col].apply(np.abs)
    counts = (abs_col <= eps).sum()
    freq = counts / len(dataset)
    
    return counts, freq


def compute_neutralpct_auc(dataset: pd.DataFrame, epsilons: List[float], col: str):
    results = defaultdict(list)
    for eps in epsilons:
        counts, freq = compute_neutralpct_fixed_threshold(dataset, eps, col)
        results["fairness_eps"].append(eps)
        results["num_examples"].append(counts)
        results["pct_examples"].append(freq)
        
    results = pd.DataFrame(results)    
    return results, auc(results["fairness_eps"], results["pct_examples"])


def compute_neutralpct(data: dict, models: List[str], datasets: List[str], epsilons: List[float], col: str, use_log10: callable=None):
    results = []
    results_auc = defaultdict(list)

    for dataset in datasets:
        df = data[dataset].copy()
        
        for model in models:
            df_model = df[df["model"] == model].copy()
            
            if use_log10:
                df_model[f"{col}_base10"] = df[col].apply(use_log10)
                out, out_auc = compute_neutralpct_auc(df_model, epsilons, f"{col}_base10")            
            else:
                out, out_auc = compute_neutralpct_auc(df_model, epsilons, col)
            
            out["model"] = model
            out["dataset"] = dataset
            results.append(out)
            
            results_auc["dataset"].append(dataset)
            results_auc["model"].append(model)
            results_auc["auc"].append(out_auc)
            
            
    return pd.concat(results), pd.DataFrame(results_auc)




def filter_data_by_col_val(data: pd.DataFrame, col: str, thres: float):
    return data[(data[col] >= -thres) & (data[col] <= thres)]


def is_neutral(df, col: str, threshold: float):
    assert 0 <= threshold <= 1
    assert col in df.columns
    return (df[col] >= -threshold) & (df[col] <= threshold)


def get_skew(df: pd.DataFrame, col: str, threshold: float):
    assert 0 <= threshold <= 1
    assert col in df.columns

    df = df.copy()
    df["skew"] = ["neutral"] * len(df)
    df.loc[df[col] < -threshold, "skew"] = "male"
    df.loc[df[col] >  threshold, "skew"] = "female"
    return df["skew"]


def get_bins(val, max_val=100, edges=(15, 10, 5, 2.5, 1, 0.1)):
    __base_interval = pd.Interval(-edges[-1], edges[-1], closed="both")
    sign = np.sign(val)
    threshold = edges[-1]

    if sign == 0 or  -threshold <= val <= threshold:
        return __base_interval

    op = operator.gt if sign > 0 else operator.le
    edges = [sign * max_val] + [e * sign for e in edges]

    for i in range(1, len(edges)):
        if op(val, edges[i]):
            e1, e2 = edges[i-1], edges[i]
            bins = (e1, e2) if sign < 0 else (e2, e1)
            return pd.Interval(*bins, closed="neither" if sign < 0 and bins[-1] == -threshold else "right")
        

def compute_skews_(data_files: dict, fairness_col, fairness_threshold, use_base_10: callable=None):
    new_data_files = {}

    for name, df in data_files.items():
        df = df.copy()
        get_fair_bins = lambda x: get_bins(val=x, max_val=100, edges=(15, 10, 5, 2.5, 1, fairness_threshold))
        
        if use_base_10:
            df[f"{fairness_col}_base10"] = df[fairness_col].apply(use_base_10)
            new_fairness_col = f"{fairness_col}_base10"
        else:
            new_fairness_col = fairness_col

        df[f"{new_fairness_col}_bins"] = df[new_fairness_col].apply(get_fair_bins)

        df["is_neutral"] = is_neutral(df, new_fairness_col, fairness_threshold)
        # Obtain a discrete measure of what gender does the model fairness_col, skews
        # note: it assumes that positive values of fairness col will skew female
        # completions; and negative values skew male completions...
        print(new_fairness_col, fairness_threshold)
        df["skew"] = get_skew(df, new_fairness_col, fairness_threshold)
        new_data_files[name] = df
        
    return new_data_files


def compute_neutral_pct_w_std(data2files: dict):
    results = defaultdict(list)
    for dataset, df in data2files.items():
        neutral_mean = df[["model", "is_neutral"]].groupby("model").mean()
        neutral_mean *= 100

        # computed as the variance of a bernoulli distribution
        Y = neutral_mean

        n = len(df) / df["model"].nunique() # number of templates (ie, dataset size)
        neutral_std = np.sqrt(Y/100 * (1 - Y/100) / n) * 100
        
        results["dataset"].extend([dataset if dataset != "USE-5" else "USE-05"] * len(neutral_mean))
        results["model"].extend(neutral_mean.reset_index()["model"])
        results["neutral_avg"].extend(neutral_mean["is_neutral"].values.tolist())
        results["neutral_std"].extend(neutral_std["is_neutral"].tolist())
        final_repr = "$" + neutral_mean["is_neutral"].map('{:.2f}'.format) + "_{\\pm " + neutral_std["is_neutral"].round(2).map('{:.2f}'.format) + "}$"

        results["neutral_final"].extend(final_repr.values.tolist())
        
    return pd.DataFrame(results)


def compute_female_male_skews(data2files: dict, model_names):
    results = defaultdict(list)
    for dataset, df in data2files.items():
        pcts = df.groupby(["model", "skew"]).count()["template"]
        
        for model in model_names:
            model_res = pcts[model]
            model_total = model_res.sum()
            
            results["dataset"].append(dataset if dataset != "USE-5" else "USE-05")
            results["model"].append(model)
            results["total"].append(model_total)
            results["pct_fem"].append(model_res.get("female", 0) / model_total * 100)
            results["pct_mal"].append(model_res.get("male", 0) / model_total * 100)
            results["counts_fem"].append(model_res.get("female", 0))
            results["counts_mal"].append(model_res.get("male", 0))
            results["partial_pct_mal"].append(results["counts_mal"][-1] / (results["counts_mal"][-1] + results["counts_fem"][-1]))
            results["partial_pct_fem"].append(1-results["partial_pct_mal"][-1])

            
            pct_diff = round(results["pct_fem"][-1] - results["pct_mal"][-1], 2)
            results["pct_fem_min_mal"].append(f"{pct_diff:.2f}")
           
    return pd.DataFrame(results).round(2)