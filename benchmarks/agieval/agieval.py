import csv
import os
from io import StringIO
from typing import Callable, List

import requests

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice, match
from inspect_ai.solver import TaskState, generate, multiple_choice, prompt_template

# TODO:
# - implement FSL
# https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/agieval


ROOT_URL = "https://raw.githubusercontent.com/ruixiangcui/AGIEval/84ab72d94318290aad2e4ec820d535a95a1f7552/data/v1_1/"

CN_TASK = [
    "gaokao-chinese",
    "gaokao-english",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-physics",
    "gaokao-mathqa",
    "jec-qa-kd",
    "jec-qa-ca",
    "logiqa-zh",
    "gaokao-mathcloze",
]

EN_TASK = [
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-math" "sat-en",
    "sat-en-without-passage",
    "aqua-rat",
    "logiqa-en",
    "math",
]

CLOZE_TASK = [
    "math",
    "gaokao-mathcloze",
]

# setup for problem + instructions for providing answer
COT_STRING_EN = r"""Think step by step before answering."""

# The cloze template is copied from the mathematic benchmark in inspect. I have added the possibility to add chain of though in the prompt.
# https://github.com/UKGovernmentBEIS/inspect_ai/blob/52688ccdc88b1dee6abaaa1144e731257b637f6b/benchmarks/mathematics.py

CLOZE_TEMPLATE_EN = r"""
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem. {cot_string}

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

MULTIPLE_CHOICE_TEMPLATE_EN = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. {cot_string}

{question}

{choices}
""".strip()


def record_to_sample(record):
    # We need this as some inputs can have a "passage" (context for the question) before the question
    input_str = (
        str(record["question"])
        if record["passage"] is None
        else f"{record['passage']} \n {record['question']}"
    )

    # the target is different if the task is a MCQ or a coze test
    target_str = str(record["label"]) if record["label"] else record["answer"]

    # the letter are in the choice string (4 first character) I may need a better check for this
    choices_list = [o[3:] for o in record["options"]] if record["options"] else None

    return Sample(
        input=input_str,
        choices=choices_list,
        target=target_str,
        metadata=record["other"] if record["other"] else {},
    )


def build_plan(
    dataset_name: str,
    cot: bool = False,
    fewshot: int | None = None,
    fewshot_seed: int = 42,
) -> List[Callable]:
    """
    NB: In the original paper and implementation, the CoT reasoning is done in two steps.
    see (Fig2 and section 4,2,2 of in ref and post_process_and_evaluation.py). First, the model is asked to generate
    an explanation of the reasoning, then the model is prompted with the question and the explaination
    generated in step one. For now CoT reasoning is implemented in a single-step prompt, as per the standard in Inspect AI
    """  # noqa: D205
    # Determine the template for the tasks in the proper language and for the correct type (MCQ, Cloze)
    if dataset_name in EN_TASK:
        template_prompt = (
            CLOZE_TEMPLATE_EN
            if dataset_name in CLOZE_TASK
            else MULTIPLE_CHOICE_TEMPLATE_EN
        )
        # Add the Chain of Thought (CoT) string if specified
        cot_string = COT_STRING_EN if cot else ""

    elif dataset_name in CN_TASK:
        raise NotImplementedError("Tests in Chinese are not yet implemented.")
        # template_prompt = CLOZE_TEMPLATE_CN if dataset_name in CLOZE_TASK else MULTIPLE_CHOICE_TEMPLATE_CN
        ## Add the Chain of Thought (CoT) string if specified
        # cot_string = COT_STRING_CN if cot else ''
    else:
        # Raise an error if the task name is not recognized
        raise ValueError(f"Task '{dataset_name}' not recognized.")

    if fewshot:
        pass
        # # but find a way to remove the current sample
        # fewshot_samples = dataset
        # plan.insert(
        #     0,
        #     system_message(
        #         SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
        #             examples="\n\n".join(
        #                 [sample_to_fewshot(sample=sample) for sample in fewshot_samples]
        #             )
        #         )
        #     ),
        # )

    # Define the plan consisting of a prompt and a generation step
    plan = [
        prompt_template(template=template_prompt, params={"cot_string": cot_string}),
        generate(),
    ]

    return plan


def task_template(
    dataset_name, cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
) -> Task:
    dataset = json_dataset(
        json_file=f"{ROOT_URL}{dataset_name}.jsonl",
        name=dataset_name,
        sample_fields=record_to_sample,
    )

    # make a plan according to the type of task and language
    plan = build_plan(
        dataset_name=dataset_name, cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )

    # adapt scorer to the type of task
    # https://github.com/UKGovernmentBEIS/inspect_ai/blob/52688ccdc88b1dee6abaaa1144e731257b637f6b/benchmarks/mathematics.py
    expression_equivalence = lambda x: x

    scorer = (
        expression_equivalence() if dataset_name in CLOZE_TASK else choice()
    )  # expression_equivalence

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=scorer,
        # from source paper 4.2.4: Implementation Details
        config=GenerateConfig(
            temperature=0, max_tokens=2048, frequency_penalty=0, top_p=1
        ),
    )


## ENGLISH Tasks
@task(group="en")
def lsat_ar(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="lsat-ar", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="en")
def lsat_lr(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="lsat-lr", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="en")
def lsat_rc(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="lsat-rc", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="en")
def sat_math(cot: bool = False, fewshot: int | None = None):
    return task_template(dataset_name="sat-math", cot=cot, fewshot=fewshot)


@task(group="en")
def sat_en(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="sat-en", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="en")
def sat_en_without_passage(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="sat-en-without-passage",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="en")
def aqua_rat(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="aqua-rat", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="en")
def logiqa_en(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="logiqa-en", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


## TASKS in CHINESE
@task(group="cn")
def gaokao_chinese(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-chinese",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_english(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-english",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_geography(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-geography",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_history(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-history",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_biology(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-biology",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_chemistry(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-chemistry",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_physics(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-physics",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def gaokao_mathqa(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-mathqa",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )


@task(group="cn")
def jec_qa_kd(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="jec-qa-kd", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="cn")
def jec_qa_ca(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="jec-qa-ca", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="cn")
def logiqa_zh(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="logiqa-zh", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


## Cloze
@task()
def math(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="math", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task()
def gaokao_mathcloze(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-mathcloze",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )
