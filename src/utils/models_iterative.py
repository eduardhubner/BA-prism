"""
Modified version of models.py that generates samples iteratively (one at a time)
to avoid Gemini API formatting issues.
"""
import os
from utils import config


def get_generated_text_iterative(prompt_base: str, text_generator_name: str, model, tokenizer, n_samples: int = 10) -> str:
    """Generates text iteratively - calls Gemini N times for 1 sample each.

    This avoids the issue where Gemini generates wrong number of samples when asked
    to generate multiple samples in one response.

    Args:
        prompt_base (str): The base prompt (description text)
        text_generator_name (str): Name of the text generator model
        model: Text generator model
        tokenizer: Tokenizer (unused for API calls)
        n_samples (int): Number of samples to generate (default: 10)

    Returns:
        str: Generated samples, one per line
    """
    # Modify the prompt to generate 1 sample instead of N
    single_sample_prompt = (
        f"Generate 1 sentence with a length of {config.MAX_TEXT_LENGTH} words. "
        f"It should be a complete, standalone text sample with no additional "
        f"formatting, introduction, numbering, or explanation. "
        f"The sentence should include:\n{prompt_base}"
    )

    samples = []
    import time

    for i in range(n_samples):
        if text_generator_name.startswith("gemini"):
            from google.generativeai.types import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }

            max_retries = 5
            retry_delay = 2  # seconds
            success = False

            for attempt in range(max_retries):
                try:
                    content = model.generate_content(single_sample_prompt, safety_settings=safety_settings)
                    response = content.text.strip()

                    # Collapse internal newlines / tabs so ExplainDataset sees exactly 1 line per sample
                    response = " ".join(response.split())

                    # Remove any numbering or extra formatting
                    if response.startswith(('1.', '1)', '-', '*')):
                        response = response.split(maxsplit=1)[1].strip()

                    samples.append(response)
                    success = True
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Warning: Failed to generate sample {i+1}/{n_samples} (attempt {attempt+1}/{max_retries}): {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"ERROR: Failed to generate sample {i+1}/{n_samples} after {max_retries} attempts")
                        raise Exception(f"Failed to generate sample after {max_retries} retries")
        else:
            raise NotImplementedError(f"Iterative generation not implemented for {text_generator_name}")

    # Join samples with newlines
    return '\n'.join(samples)
