import torch
from torch import nn
from smolagents import LiteLLMModel, Tool
from typing import List, Dict, Optional
import openai
import numpy as np
import os
from .supervisor import run_supervisor


class ModelEnsemble(LiteLLMModel):
    """
    A class that uses multiple LLMs to generate outputs from the same prompt
    """

    def __init__(
        self,
        model_id: list[str],
        embedding_model_id: str = "nomic-ai/nomic-embed-text-v1.5",
        supervisor=False,
        **kwargs,
    ):
        self.model_ids = model_id
        self.model_kwargs = kwargs
        self.embedding_model_id = embedding_model_id
        self.embedding_client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
        self.supervisor = supervisor
        super().__init__(model_id="ensemble")

    def set_user_query(self, user_query):
        self.user_query = user_query

    def create_client(self):
        client = {}
        for name in self.model_ids:
            client[name] = LiteLLMModel(name, **self.model_kwargs)
        return client

    def aggregate_responses(self, messages: list):
        """
        Embed all messages, compute mean and return message with embedding closest to mean.
        """
        if self.supervisor:
            return run_supervisor(self.user_query, messages)

        # embed all messages
        # compute mean
        # take plan that is closest to mean
        if len(messages) == 1:
            return messages[0]
        embeddings = []
        for message in messages:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model_id,
                input="search_document: " + message.content,
            )
            embeddings.append(response.data[0].embedding)

        embeddings = np.stack(embeddings, axis=0)
        embed_mean = np.mean(embeddings, axis=0)
        cos_sim = (
            embeddings
            @ embed_mean
            / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embed_mean))
        )
        return messages[np.argmax(cos_sim)]

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ):
        responses = []
        for model in self.client.values():
            responses.append(
                model(
                    messages,
                    stop_sequences,
                    grammar,
                    tools_to_call_from,
                    **kwargs,
                )
            )

        return self.aggregate_responses(responses)

