from opendeepsearch.prompts import REWRITER_SYSTEM_PROMPT
from fireworks.client import Fireworks
import os

def rewrite(query: str, model_id, **kwargs):
    """
    Rewrite the given query with an LLM according to the rewriter system prompt
    """
    client = Fireworks(api_key = os.environ["FIREWORKS_API_KEY"])
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
            "role": "system",
            "content": REWRITER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Input Query: " + query,
        }],
        **kwargs
        )
    return response.choices[0].message.content 