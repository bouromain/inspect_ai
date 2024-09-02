# AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models

[AGIEval](https://arxiv.org/pdf/2304.06364) is ...


## Execution
Example from the dataset:
```python
Passage: 

Question: 
```
The model is tasked to answer the question with a Multiple Choice Question.

## Evaluation
This implementation is based on the [original implementation](https://github.com/ruixiangcui/AGIEval/tree/main).

``` bash
# to run only one task (eg: sat_math)
inspect eval agieval_en.py@sat_math 

# to run agieval (english group)
inspect list tasks */agieval -F group=en 

inspect eval agieval_en.py  -T fewshot=5 cot=True
```
