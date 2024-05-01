"""Utils to interact with OpenAI's API.


This script was created before we knew about the `tenacity` library.
We're keeping it for legacy reasons, but we recommend using `tenacity` instead
of creating your own `get_completion_block_until_succeed` function.
To learn more, checkout the `Example 1: Using the Tenacity library` section
in OpenAI's official documentation: 
- https://platform.openai.com/docs/guides/rate-limits/error-mitigation


Before using this script, you also need to set up the API key.
"""
import openai, time, traceback

openai.api_key = "API_KEY"


def get_completion(
    prompt: str, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.3
) -> str:
    """Get a chat completion from the OpenAI API.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model.

    model : str
        The model to use. Default is "gpt-3.5-turbo-0125".

    temperature : float
        The degree of randomness of the model's output. Default is 0.3.

    Returns
    -------
    str
        The model's response.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


def get_completion_block_until_succeed(
    prompt: str,
    model: str = "gpt-3.5-turbo-0125",
    temperature: float = 0.3,
    sleeptime: int = 3,
):
    """Get a chat completion from the OpenAI API, attempting indefinitely until it succeeds."""
    success = False
    while not success:
        try:
            response = get_completion(prompt, model, temperature=temperature)
            success = True
        except:
            traceback.print_exc()
            time.sleep(sleeptime)

    return response
