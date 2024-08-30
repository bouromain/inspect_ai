from agieval import task_template

from inspect_ai import Task, task


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
