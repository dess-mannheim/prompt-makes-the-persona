from vllm import LLM
import pandas as pd
import argparse
import pandas as pd
import json


WS_PATH = "." # CHANGE to your working directory
N_GPUS = 2
CLASSIFICATION_TASK = """You are an expert annotator tasked with classifying whether a {task_full} was written by a language model or a human.
Language models might describe themselves as AI, chat assistant, large language model, etc., while {task_full} human are clearly describing a human being.
If the {task_full} contains a language model refusal, also classify it as "AI".
Only provide your annotation, no additional explanation!
Use the following JSON format for your annotation: {{"author": <one of the following options: "AI" or "human", without any extra commentary whatsoever; just the annotation!>,}}

Now annotate the following {task_full}:

"""

def get_task_details(task_name):
    if task_name == 'SD':
        return {'name': 'SD', 'full': 'self-description', 'column': 'self_description'}
    elif task_name == 'Bio':
         return {'name': 'Bio', 'full': 'social-media biography', 'column': 'bio'}
    

def parse_annotation(text):
    try:
        author = json.loads(text.replace(',}','}'))['author']
        assert author in ['human', 'AI']
        return author == 'AI'
    except:
        print(f'Failed to parse annotation: {text}')
        return None


def run_annotation(model_id, task_name):
    llm = LLM(model='Qwen/Qwen2.5-32B-Instruct', 
                    #enforce_eager=True, 
                    #gpu_memory_utilization=0.8, 
                    max_num_seqs=100,
                    tensor_parallel_size=N_GPUS,
                    enable_prefix_caching=True,
                    max_model_len=5000)

    model_short = model_id.split("/")[-1]
    task = get_task_details(task_name)

    if task is None:
        print(f'Cannot annotate task {task_name}, please select one of "SD" or "Bio"')
        return
    
    res_path = f'{WS_PATH}/data/results/{task['name']}'

    try: response_df = pd.read_csv(f"{res_path}/merged_results_{model_short}.csv", index_col=0)
    except:
        print(f'No results found for {model_short}, {task['full']}')
        return

    response_df = response_df.dropna(subset=task['column'])

    chats = [
        [{
            'role': 'user', 'content': CLASSIFICATION_TASK.format(task_full=task['full']) + str(response),
        }]
        for _, response in response_df[task['column']].items()
    ]

    sampling_params = llm.get_default_sampling_params()
    sampling_params.n = 1
    sampling_params.max_tokens = 20

    print(f'Now annotating {model_short}, {task['full']}')
    responses = llm.chat(chats, sampling_params)

    response_df['LLM_AI_annotation'] = [parse_annotation(r.outputs[0].text) for r in responses]
    response_df.to_csv(f"{res_path}/merged_results_{model_short}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="huggingface checkpoint id of the model to run", type=str)
    parser.add_argument("--task", help="which open task to run ('SD' or'Bio')", type=str)
    args = parser.parse_args()

    run_annotation(args.model_id, args.task)