# Benchmark Creation Configurations

This directory contains the configurations used to run the pipeline to generate the stereotype-free bias benchmarks.
As mentioned in the paper, we generate three variants of the benchmark that differ in the number of words within each sentence. 
There exists a tension between the number of words and the gender correlations present in the sentences. 
By definition, smaller sentences are less likely to contain gender skewed words, for we are restricting the models to generate sentences with K words where 2/K words are defined. 
As a result, many of the smaller sentences end up exhibiting the same structure for the USE-5 (5 words) dataset. At the same time, this imposes restrictions in the diversity of the sentences, since there are only a few ways of constructing small sentences that contain the exact words requested.
