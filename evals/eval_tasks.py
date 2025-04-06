import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
from opendeepsearch import (
    OpenDeepSearchTool,
    rewrite,
    build_augmented_prompt,
    ModelEnsemble,
)

from smolagents import (
    AgentError,
    CodeAgent,
    LiteLLMModel,
    HfApiModel,
    PythonInterpreterTool,
    ToolCallingAgent,
)
from smolagents.agents import ActionStep


load_dotenv()

APPEND_ANSWER_LOCK = threading.Lock()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Runs an agent powered by the given model on smolagent benchmark."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="The date for the evaluation.",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        nargs="+",
        default=[
            "./evals/datasets/frames_test_set.csv",
            "./evals/datasets/simple_qa_test_set.csv",
        ],
        help="List of evaluation task paths",
    )
    parser.add_argument(
        "--search-model-id",
        type=str,
        default="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
        help="The model ID to use for the search tool (defaults to same as model-id)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        choices=["LiteLLMModel", "HfApiModel", "ModelEnsemble"],
        help="The model type to use (LiteLLMModel or HfApiModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="fireworks_ai/accounts/fireworks/models/qwq-32b",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--agent-action-type",
        type=str,
        default="codeact",
        choices=["codeact", "tool-calling", "vanilla"],
        help="The agent action type: 'codeact', 'tool-calling', or 'vanilla' to use the vanilla llm",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="The number of processes to run in parallel",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run for each evaluation",
    )
    parser.add_argument(
        "--supervisor",
        type=bool,
        default=False,
        help="Use a supervisor to aggregate the ModelEnsemble results",
    )
    return parser.parse_args()


def load_eval_dataset(eval_tasks: list):
    eval_ds = {}
    for task_path in eval_tasks:
        task_name = task_path.split("/")[-1][:-4]
        df = pd.read_csv(task_path)
        dataset = Dataset.from_pandas(df)
        eval_ds[task_name] = dataset
    return eval_ds


def serialize_agent_error(obj):
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with APPEND_ANSWER_LOCK, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"


def run_with_timeout(func, timeout):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return "Timed Out"


def answer_single_question(
    example, model, answers_file, action_type, search_model_id=None
):
    if action_type == "vanilla":
        agent = model
    elif action_type == "codeact":
        agent = CodeAgent(
            tools=[
                OpenDeepSearchTool(
                    model_name=search_model_id or model.model_id, reranker="jina"
                )
            ],
            model=model,
            additional_authorized_imports=["numpy"],
            max_steps=15,
        )
    elif action_type == "tool-calling":
        agent = ToolCallingAgent(
            tools=[
                OpenDeepSearchTool(
                    model_name=search_model_id or model.model_id, reranker="jina"
                ),
                PythonInterpreterTool(),
            ],
            model=model,
            additional_authorized_imports=["numpy"],
            max_steps=15,
        )

    # enable planning (first turn of agent only)
    # agent.planning_interval = 100

    new_system_prompt_instruction = 'You will be given the user task and at most three additional reformulations equivalent to this task. You should solve the user task as best you can, and use the additional reformulations, as needed. You will receive the user task starting with: "Input Query: ..." and the additional reformulations starts with: "Rephrased Versions: ...", where the additional reformulations are enumerated with a "-".'

    agent.system_prompt = agent.system_prompt.replace(
        "You will be given a task to solve as best you can.",
        new_system_prompt_instruction,
    )

    augmented_question = rewrite(
        example["question"],
        model_id="accounts/fireworks/models/llama-v3p3-70b-instruct",
        temperature=0.7,
    )

    if isinstance(model, ModelEnsemble):
        model.set_user_query(augmented_question)

    start_time = time.time()
    TIMEOUT_SECONDS = 300  # 5 minutes timeout

    try:
        if action_type == "vanilla":

            def get_vanilla_response():
                response = agent([{"role": "user", "content": example["question"]}])
                return response.content, agent.last_output_token_count

            answer, token_count = run_with_timeout(
                get_vanilla_response, TIMEOUT_SECONDS
            )
            intermediate_steps = answer
        else:

            def get_agent_response():
                new_prompt = build_augmented_prompt(
                    example["question"], augmented_question
                )
                response = str(agent.run(new_prompt))
                token_count = agent.monitor.get_total_token_counts()
                # Remove memory from logs to make them more compact.
                for step in agent.memory.steps:
                    if isinstance(step, ActionStep):
                        step.agent_memory = None
                return response, token_count, str(agent.memory.steps)

            answer, token_count, intermediate_steps = run_with_timeout(
                get_agent_response, TIMEOUT_SECONDS
            )

        end_time = time.time()
    except Exception as e:
        print("Error on ", augmented_question, e)
        intermediate_steps = []
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "model_id": model.model_id,
        "agent_action_type": action_type,
        "original_question": example["question"],
        "answer": answer,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps,
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": token_count,
    }
    append_answer(annotated_example, answers_file)


def answer_questions(
    eval_ds,
    model,
    date,
    action_type: str = "codeact",
    output_dir: str = "output",
    parallel_workers: int = 32,
    search_model_id: str = None,
    num_trials: int = 1,
):
    date = date or datetime.date.today().isoformat()
    model_id = model.model_id

    # Create directory structure: output/model_id/action_type/task
    model_dir = model_id.replace("/", "__")

    for task in eval_ds:
        task_dir = os.path.join(output_dir, model_dir, action_type, task)
        os.makedirs(task_dir, exist_ok=True)

        for trial in range(num_trials):
            file_name = f"{task_dir}/{model_id.replace('/', '__')}__{action_type}__{task}__trial{trial}.jsonl"
            print(
                f"Starting processing trial {trial + 1}/{num_trials} and writing output to '{file_name}'"
            )
            answered_questions = []
            if os.path.exists(file_name):
                with open(file_name, "r") as f:
                    for line in f:
                        answered_questions.append(json.loads(line)["original_question"])
            examples_todo = [
                example
                for example in eval_ds[task]
                if example["question"] not in answered_questions
            ]
            print(f"Launching {parallel_workers} parallel workers.")

            with ThreadPoolExecutor(max_workers=parallel_workers) as exe:
                futures = [
                    exe.submit(
                        answer_single_question,
                        example,
                        model,
                        file_name,
                        action_type,
                        search_model_id,
                    )
                    for example in examples_todo
                ]
                for f in tqdm(
                    as_completed(futures),
                    total=len(examples_todo),
                    desc="Processing tasks",
                ):
                    f.result()

            print("All tasks processed.")


if __name__ == "__main__":
    args = parse_arguments()

    eval_ds = load_eval_dataset(args.eval_tasks)

    if args.model_type == "LiteLLMModel":
        model = LiteLLMModel(
            args.model_id,
            max_completion_tokens=8192,
            temperature=0.2,
            # api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    elif args.model_type == "ModelEnsemble":
        model = ModelEnsemble(
            max_completion_tokens=8192,
            temperature=0.2,
            model_id=[
                "fireworks_ai/accounts/fireworks/models/qwq-32b",
                "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
                "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
                "fireworks_ai/accounts/fireworks/models/qwen2p5-vl-32b-instruct",
                "fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct",
            ],
            supervisor=args.supervisor,
        )
    else:
        model = HfApiModel(args.model_id, provider="together", max_tokens=8192)

    answer_questions(
        eval_ds,
        model,
        args.date,
        action_type=args.agent_action_type,
        parallel_workers=args.parallel_workers,
        search_model_id=args.search_model_id,
        num_trials=args.num_trials,
    )
