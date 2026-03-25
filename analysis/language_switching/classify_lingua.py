from itertools import product
import pandas as pd
from tqdm.auto import tqdm
from lingua import Language, LanguageDetectorBuilder
import os

RES_PATH = "../../data/results"  # path to results

TASKS = [
    {'name': 'SD', 'column': 'self_description'},
    {'name': 'Bio', 'column': 'bio'}
]

MODELS = [
    "google/gemma-3-27b-it",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-0325-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

detector = LanguageDetectorBuilder.from_all_languages().build()

classified_languages = pd.DataFrame()

task_model = list(product(TASKS, MODELS))

for task, model in tqdm(task_model):
    model_short = model.split('/')[1]

    texts = pd.read_csv(f'{RES_PATH}/{task['name']}/merged_results_{model_short}.csv', index_col=0)[task['column']]

    for index, sentence in tqdm(list(texts.items())):
        try:
            results = detector.compute_language_confidence_values(sentence)
            result_df = pd.DataFrame({result.language.name: result.value for result in results if result.value > 0}, index=[index])
            classified_languages = pd.concat([classified_languages, result_df])
        except:
            pass

    out_dir = f"{RES_PATH}/{task['name']}/language_switching"
    os.makedirs(out_dir, exist_ok=True)

    classified_languages.to_csv(f"{out_dir}/classified_languages_{model_short}.csv")
