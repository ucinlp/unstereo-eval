The code is subdivided into three main parts, spanning utilities,
creation of the benchmark, and evaluation script. 

## Utils

- `model_utils.py`: contains model wrappers to load models, create names
and compute perplexity.

- `openai_utils.py`: contains the wrappers around openai API to use
ChatGPT for the generation of the benchmark.

- `templates.py`: template-focused functionality, including automatic
mapping from templates to the gendered variants and the computation of
log probabilities for each template.

## Benchmark creation / processing

- `run_pipeline.py`: stage 2 of the benchmark creation. It assumes there exists a file with the list of seed words to use to bootstrap the generation.
- `run_pipeline_revision.py`: fixes the statements generated in the first stage. Instead of this you can simply run a larger number of statements or generate a larger number of statements, and filter out those that do not contain the exact conjugation/form of the *seed word*.


## Evaluation 

- `run_evaluation.py`: load the files and computes the log probabilities for both versions of the template.
