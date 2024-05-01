"""HuggingFace model utilities for LM setup and computing perplexity scores."""
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    OPTForCausalLM,
)
from typing import Tuple, List

import torch, tqdm
from torch.nn import CrossEntropyLoss
import numpy as np


def init_pad_token(tokenizer: AutoTokenizer):
    """Initialize the pad token for the specified tokenizer.

    If the pad token is not already set, set it to the eos token.
    """
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print(f"SETTING pad token to eos token: '{tokenizer.eos_token}'")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def get_model_filename(*args) -> str:
    """Given a set of strings defining the model, create a filename.

    Notes
    -----
    Imagine we'd like to create a filename for the model
    "EleutherAI/pythia-70m-deduped", revision="step3000", the resulting
    filename would be "EleutherAI__pythia-70m-deduped__step3000".
    """
    args = [a for a in args if a]
    args = [a.replace("/", "__") for a in args]
    args = [a for a in args if a]
    return "__".join(args)


def load_model(
    name: str, revision: str = None, device: str = None
) -> Tuple[str, object, object, str]:
    """Load a HuggingFace model and tokenizer given the model name and revision.

    This method has been tested for a subset of models and may not generalize
    to a broader class of language models. For example, we've used this method
    to load gpt2, gpt-neo, pythia, Llama-2, MPT, OPT, Mistral, and OLMo models.

    Parameters
    ----------
    name: str
        Name of the model as defined in the HuggingFace model hub or
        a local path to a model.

    revision: Optional[str]
        Revision of the model to load. If None, no revision is specified
        when loading the model.

    device: Optional[str]
        Device to load the model on. If None, the model is loaded on the GPU
        if the GPU is available, otherwise on the CPU. For models larger than
        7B, we automatically set the device to be "auto"

    Returns
    -------
    model_filename: str
        Filename of the model.

    model: AutoModelForCausalLM
        The loaded model.

    tokenizer: AutoTokenizer
        The loaded tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = {}
    tokenizer_kwargs = {}
    # Load GPT2 model
    if "gpt2" in name:
        model_class = GPT2LMHeadModel

    elif "llama-2" in name:
        model_class = LlamaForCausalLM
        model_kwargs.update(device_map="auto", torch_dtype=torch.float16)
        device = None

    elif "gpt-neo" in name:
        model_class = GPTNeoForCausalLM

    elif "pythia" in name:
        if revision:
            model_kwargs.update(revision=revision)

        if "12b" in name:
            model_kwargs.update(device_map="auto")
            device = "auto"  # avoid the .to(device) call below

        # GPTNeoXTokenizerFast
        model_class = GPTNeoXForCausalLM
        tokenizer_kwargs.update(padding_side="left")

    elif "opt-" in name:
        model_class = OPTForCausalLM
        # model_kwargs = dict(torch_dtype=torch.float16)

    elif "mpt-" in name:
        config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        # config.attn_config['attn_impl'] = 'triton'
        config.init_device = device  # For fast initialization directly on GPU!
        # The option for torch_dtype is useful at training time, when we have large datasets and need to conduct
        # fast updates.
        # model = AutoModelForCausalLM.from_pretrained(name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, config=config, trust_remote_code=True, device_map="auto"
        )
        # model.tie_weights()
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", padding_side="left"
        )
        init_pad_token(tokenizer)
        return get_model_filename(name), model, tokenizer, device

    elif "open_llama" in name:
        # model = LlamaForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map='auto')
        model = LlamaForCausalLM.from_pretrained(name, device_map="auto")
        # model.tie_weights()

        tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        return get_model_filename(name), model, tokenizer, None
    else:
        print(f"Undefined: {name}")
        tokenizer_kwargs = dict(padding_side="left")
        model_class = AutoModelForCausalLM

    print("Using model_kwargs for model:", model_kwargs)
    model = model_class.from_pretrained(name, **model_kwargs)

    print("Using tokenizer_kwargs for model:", tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)

    model_filename = get_model_filename(name, revision)
    print("Filename:", model_filename)

    if device not in (None, "auto"):
        model.to(device)
    print("Model device:", model.device)

    init_pad_token(tokenizer)
    return model_filename, model, tokenizer, device


def compute_perplexity(
    templates: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):
    """Compute the perplexity of a set of templates using a given model.

    It is a modified version of the code from the HuggingFace evaluation script
    that can be found at https://github.com/huggingface/evaluate/commit/9f0f888eb455bc0952f467b1cab47716e3f04e83.

    Parameters
    ----------
    templates: List[str]
        List of input sentences to compute the perplexity of.

    model: AutoModelForCausalLM
        The model to use for computing the perplexity.

    tokenizer: AutoTokenizer
        The tokenizer to use for tokenizing the input sentences.

    batch_size: int, defaults to 16.
        The number of sentences to process at a time. By default, we
        process 16 sentences at a time. Note that due to matrix
        multiplication algorithms, the perplexity values assigned to
        each sentence may change slightly with different batch sizes.

    add_start_token: bool, defaults to True.
        Whether to add a start token to the input sentences. If True,
        the model is expected to have a bos_token defined.

    device: Optional[str]
        Useless argument. Kept for compatibility of the API. The device
        is automatically set to the model's device.

    max_length: Optional[int]
        Maximum length of the input sequences. If specified, the input
        sequences are truncated to this length.
    """
    device = model.device

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert tokenizer.bos_token is not None, (
            "Input model must already have a BOS token if using",
            " add_start_token=True. Please use a different model,",
            " or set add_start_token=False",
        )
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        templates,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    # Pre compute the encodings and attention masks for the input texts
    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(torch.ge(attn_masks.sum(1), 2)), (
            "When add_start_token=False, each input text must be at least",
            " two tokens long. Run with add_start_token=True if inputting",
            " strings of only one token, and remove all empty input strings.",
        )

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    for start_index in tqdm.tqdm(range(0, len(encoded_texts), batch_size)):
        # Select batch sequences
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch
        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        # There is a mismatch between the input sequence and the corresponding logit
        # because the models are autoregressive, and so the logits are shifted by one.
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # Determine the log-likelihood assigned to each token
        log_likelihood = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        # Sum the log-likelihoods for each token in the sequence that is not padding
        log_likelihood = (log_likelihood * shift_attention_mask_batch).sum(1)
        # Compute the perplexity
        perplexity_batch = torch.exp(log_likelihood / shift_attention_mask_batch.sum(1))
        ppls += perplexity_batch.tolist()
    return {
        "perplexities": ppls,
        "mean_perplexity": np.mean(ppls),
        "std_perplexity": np.std(ppls),
    }
