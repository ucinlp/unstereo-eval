from typing import Dict, List

import torch, tqdm
from torch.nn import CrossEntropyLoss


def fill_template(template: str, template_infils: Dict[str, str]) -> str:
    """Replaces the placeholders in the template with the provided values."""
    for placeholder, placeholder_val in template_infils.items():
        template = template.replace(placeholder, placeholder_val)
        template = template.replace('"', "")
    # Capitalize the template to make sure that if we replace pronouns
    # we keep fluent English sentences.
    return template.strip().capitalize()


def compute_logprob_batched(
    templates: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
    add_start_token: bool = True,
    max_length=None,
) -> Dict[str, List[float]]:
    """Compute the log likelihood of a set of templates using a given model.

    It is a modified version of the perplexity code implemented available at HuggingFace
    https://github.com/huggingface/evaluate/commit/9f0f888eb455bc0952f467b1cab47716e3f04e83.

    Parameters
    ----------
    templates: List[str]
        List of input sentences to compute the log likelihood of.

    model: AutoModelForCausalLM
        The model to use for computing the log likelihood .

    tokenizer: AutoTokenizer
        The tokenizer to use for tokenizing the input sentences.

    batch_size: int, defaults to 16.
        The number of sentences to process at a time. By default, we
        process 16 sentences at a time. Note that due to matrix
        multiplication algorithms, the log likelihood values assigned to
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

    Returns
    -------
    dict[str, List[float]]
        A dictionary containing the log likelihood values for each
        sentence in the input list, as well as the number of tokens.
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
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
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
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    logprobs, num_tokens = [], []
    loss_fct = CrossEntropyLoss(reduction="none")
    for start_index in tqdm.tqdm(range(0, len(encoded_texts), batch_size)):
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
        log_likelihood = -(log_likelihood * shift_attention_mask_batch).sum(1)
        logprobs += log_likelihood.tolist()
        num_tokens += shift_attention_mask_batch.sum(1).tolist()
    return {
        "logprobs": logprobs,
        "num_tokens": num_tokens,
    }


def compute_templates_logprobs(
    templates: List[str],
    model,
    tokenizer,
    template_infils: dict[str, str],
    batch_size=16,
    add_start_token=True,
):
    """Compute the log likelihood of a set of templates using a given model.

    Begin by replacing the placeholders for the ``templates`` with the values
    defined in ``template_infils``. We expect the ``template_infils`` to be
    structured as follows:
    - male: ``{"{pronoun}": "he", "{pronoun1}": "his", "{pronoun2}": "him"}``
    - female: ``{"{pronoun}": "she", "{pronoun1}": "her", "{pronoun2}": "her"}``.
    """
    templates = [fill_template(t, template_infils) for t in templates]
    results = compute_logprob_batched(
        templates,
        model,
        tokenizer,
        batch_size=batch_size,
        add_start_token=add_start_token,
    )
    results["templates"] = templates
    return results
