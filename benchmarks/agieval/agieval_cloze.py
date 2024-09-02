from agieval import task_template

from inspect_ai import task


## Cloze
@task(group="en")
def math(cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42):
    return task_template(
        dataset_name="math", cot=cot, fewshot=fewshot, fewshot_seed=fewshot_seed
    )


@task(group="cn")
def gaokao_mathcloze(
    cot: bool = False, fewshot: int | None = None, fewshot_seed: int = 42
):
    return task_template(
        dataset_name="gaokao-mathcloze",
        cot=cot,
        fewshot=fewshot,
        fewshot_seed=fewshot_seed,
    )
