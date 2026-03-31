import argparse
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

RES_PATH = "../../data/results"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

def create_embeddings(model_name, task):

    results_df = pd.read_csv(f'{RES_PATH}/{task}/merged_results_anonymized_{model_name}.csv')

    if task == 'Bio':
        parsed_docs = results_df['bio_anonymized']
    elif task == 'SD':
        parsed_docs = results_df['self_description_anonymized']

    print(f'---\n Embedding text for {model_name}, {task}')

    print(f'{parsed_docs.count()} out of {len(parsed_docs)} could be parsed')
    parsed_docs = parsed_docs.dropna() # do not encode unparsable

    # Prepare embeddings
    sentence_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL)
    embeddings = sentence_model.encode(list(parsed_docs), show_progress_bar=True)

    embedding_df = pd.DataFrame(embeddings, index=parsed_docs.index)

    out_dir = f"{task}" 
    os.makedirs(out_dir, exist_ok=True)

    embedding_df.to_csv(f'{out_dir}/embeddings_anonymized_{model_name}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, help='huggingface checkpoint id of the model')
    parser.add_argument('task', type=str, help="which open task to run ('SD' or 'Bio')")
    args = parser.parse_args()

    model_name = args.model_id.split('/')[1]
    create_embeddings(model_name, args.task)