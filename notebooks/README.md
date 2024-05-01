List of notebooks and their corresponding functionality:

# Data preprocessing 

- `0.Preprocess-baselines`: preprocessing notebook, should be run to transform baseline datasets like Winobias, Winogender and StereoSet by converting it to the format expected by the `src/run_evaluation.py` and `0.Preprocess-datasets.ipynb`. This will include creating the templates using the same placeholder format and preserve some of the original information.

- `0.Preprocess-datasets`: preprocessing notebook, it will enrich the datasets with the corresponding measures of MaxPMI and gender skews. Executing this notebook will generate files in `data/datasets/preprocessed` and we uploaded these files to HuggingFace. Note that these files will have no filter on the MaxPMI value and this should be done by the user before using the dataset for evaluation.

# Evaluation notebooks

- `2.Analysis-collect-benchmarks-statistics`: collects basic statistics for each of the individual benchmarks, including length, number and position of pronouns, average pmi per sentence.

- `2.Analysis-Example-Selection`: analysis notebook that we've used to collect a subset of the datasets for human annotation and also to select the examples in the paper.

- `2.Evaluation-post-process-results`: gathers all the individual score files regarding each individual dataset and compile them into a single one.

- `2.Evaluation-post-process-metrics`: loads the files compiled using the notebook `2.Evaluation-post-process-results` and computes the metrics for different levels of gender correlations $\eta$. It persists the results in `results/processed-results`. This makes it easier to access and report values for tables and plots.

- `2.Evaluation-preference-disparity-tables`: loads the files compiled using the notebook `2.Evaluation-post-process-results` and computes the preference disparity metric. Does not create any file, and instead we've used it to obtain the latex for the reported tables in the paper. It reports this value for different values of $eta$.

`2.Evaluation-unstereo-score-and-aufc-tables`: loads the files compiled using the notebook `2.Evaluation-post-process-results` and computes the unstereo score, fairness gap, and area under the curve metrics. We've used it to obtain the latex for the reported tables in the paper. It reports this value for different values of $eta$. It also provides some in depth analysis.


# Non-Stereotype benchmark construction

- `1.BenchmarkConstruction-WordSelection`: runs the first stage of the pipeline. It will select the words from a predefined list of words for which both `PMI(w, she)` and `PMI(w, he)` are well-defined.


## Plotting 

- `3.Plotting-Video`: uploads the [all_datasets.json](../results/processed-results/all_datasets.json) and plots the preference disparity and the unstereo score plots.



