from random import random
from smolagents import ChatMessage
from fireworks.client import Fireworks
import os


SYSTEM_PROMPT = """You are a supervisor model within an LLM Search Engine Agent pipeline. Your role is to evaluate a set of candidate responses produced by multiple smaller to medium sized models in response to a user query. The input you receive includes:

- The original user query (which may also include rephrasings of the query)  
- A list of candidate responses generated by various models (the different responses are separated by newlines and the sequence "* * * * *")

Your task is to analyze all candidate responses and select the one that:
1. Is most consistent with the original user query.
2. Is logically correct.
3. Provides a valid and accurate solution to the user query.

After carefully comparing the candidate responses against these criteria, simply output the index (starting at 0) of the candidate response that best meets the requirements. Do not include any additional text or explanation in your output.
"""


def _make_user_prompt(user_query: str, model_responses: list[str]) -> str:
    return f"""
User Query:
{user_query}

Model Responses:
{"\n* * * * *\n".join(model_responses)}
"""


def run_supervisor(
    user_query: str,
    model_responses: list,
    model_id: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
) -> ChatMessage:
    response_texts = [response.content for response in model_responses]
    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _make_user_prompt(user_query, response_texts),
            },
        ],
    )

    try:
        idx = int(response.choices[0].message.content)
    except:
        idx = random.randint(0, len(model_responses) - 1)
    return model_responses[idx]
