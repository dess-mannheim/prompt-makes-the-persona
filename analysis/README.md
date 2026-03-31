# Analysis

Evaluate LLM simulations along the following dimensions:

- simulation language (`/language_switching`)
- stereotypes (`/marked_personas`)
- alignment with survey responses (`/opinion_alignment`)
- semantic diversity (`/semantic_diversity`)

## 1. Simulation Language

The script `classify_lingua.py` performs language detection on the simulation outputs for all models included in our experiments. To adjust which models are analyzed, modify the global variable `MODELS` directly within the script. 

## 2. Stereotypes

We adapt the code from the original [Marked Personas paper](https://aclanthology.org/2023.acl-long.84.pdf) from: https://github.com/myracheng/markedpersonas. The script  `main.py` computes Marked Words for all combinations of demographic groups and prompt strategies, as well as the accuracy of an SVM classifier in distinguishing simulations from different demographic groups. To specify the models and task, set the global variables `MODELS` and `TASK` in the script. 

## 3. Opinion Alignment

We adapt the code from the original [OpinionQA paper](https://arxiv.org/abs/2303.17548) from: https://github.com/tatsu-lab/opinions_qa
To run this analysis, you have to download the human survey data (and save in `opinions_qa/data/human_resp/`) as well as the script `helpers.py` (and rename to `opinionqa_helpers.py`) from the original code repository. After that, to generate the survey alignment results, run the notebook `opinionqa.ipynb`.

## 4. Semantic Diversity

Use `create_embeddings.py` to create embeddings of the simulations of a specific model. 

Example: Create embeddings for self-descriptions generated with Llama-3.1-8B-Instruct:

```bash
python create_embeddings.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --task SD
```

Next, cluster the embeddings based on demographic group and prompt strategy, and compute the mean pairwise distance within each cluster. This provides an estimate of the semantic diversity of the corresponding simulations:

```bash
python evaluate_clusters.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --task SD
```







