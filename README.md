# UnStereoEval

Repository for the paper [*Are Models Biased on Text without Gender-related Language?*](https://openreview.net/forum?id=w1JanwReU6) accepted at ICLR 2024! 
In this paper, we challenge a common observation in prior work considering the gender bias evaluation of large language models (LMs). 
The observation is that models reinforce stereotypes in the training data by picking up on gendered correlations. 
In this paper, we challenge this assumption and instead address the question: **Do language models still exhibit gender bias in non-stereotypical settings?**

To do so, we introduce UnStereoEval (USE), a novel framework tailored for investigating gender bias in stereotype-free scenarios. USE defines a sentence-level score based on pretraining data statistics to determine if the sentence contain minimal word-gender associations. To systematically assess the fairness of popular language models in stereotype-free scenarios, we utilize USE to automatically generate benchmarks without any gender-related language. By leveraging USE's sentence-level score, we also repurpose prior gender bias benchmarks (Winobias and Winogender) for non-stereotypical evaluation.

## Datasets 

The unconstrained version of the datasets can be found at [🤗 datasets](https://huggingface.co/datasets/ucinlp/unstereo-eval).

## Code 

Coming soon!

## Analysis 

Coming soon!

## Citation 

If you find this work interesting and useful for your research, please consider citing us (: 

```bibtex
@inproceedings{belem2024-unstereoeval,
    title={Are Models Biased on Text without Gender-related Language?},
    author={Catarina G Bel{\'e}m and Preethi Seshadri and Yasaman Razeghi and Sameer Singh},
    month={May},
    year={2024},
    booktitle={The Twelfth International Conference on Learning Representations},
    url={https://openreview.net/forum?id=w1JanwReU6}
}
```
