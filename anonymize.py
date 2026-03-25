import pandas as pd
import regex as re
import argparse

WS_PATH = "." # CHANGE to your working directory

def anonymize(text, remove_names=True, remove_demographic_markers=True, replacement="[identity]"):
    text = str(text)
    
    if remove_names:
        text = re.sub(
            r"\b(?:Olson|Snyder|Wagner|Meyer|Schmidt|Ryan|Hansen|Hoffman|Johnston|Larson|Smalls|Jeanbaptiste|Diallo|Kamara|Pierrelouis|Gadson|Jeanlouis|Bah|Desir|Mensah|Nguyen|Kim|Patel|Tran|Chen|Li|Le|Wang|Yang|Pham|Garcia|Rodriguez|Martinez|Hernandez|Lopez|Gonzalez|Perez|Sanchez|Ramirez|Torres|Khan|Ali|Ahmed|Hassan|Yılmaz|Kaya|Demir|Mohammadi|Hosseini|Ahmadi)\b",
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    if remove_demographic_markers:
        text = re.sub(
            r"\b(?:White|Black|Asian|Middle-Eastern|Hispanic|man|woman|nonbinary person|male|female|nonbinary)\b|(?:Mr\.|Ms\.|Mx\.)", # treat Mr.|Ms.|Mx. separately, because . is not a word boundary
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Replace pronouns
    text = re.sub(
        r"\b(?:he|she|they|himself|herself|themselves|his|her|their|him|them)\b",
        replacement,
        text,
        flags=re.IGNORECASE,
    )

    # Collapse multiple [identity] tokens into one
    text = re.sub(r'(\[identity\]\s*){2,}', replacement + ' ', text)

    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def anonymize_result(model_name, task):
    if task == 'Bio': 
        target = 'bio'
    elif task == 'SD': 
        target = 'self_description'
    else:
        raise ValueError("task must be 'Bio' or 'SD'")
    
    df = pd.read_csv(f'{WS_PATH}/data/results/{task}/merged_results_{model_name}.csv')
    df[target + '_anonymized'] = df.apply(
        lambda row: anonymize(
            row[target],
            remove_names=(row['persona_type'] == 'name'),
            remove_demographic_markers=True
        ),
        axis=1
    )
    df.to_csv(f'{WS_PATH}/data/results/{task}/merged_results_anonymized_{model_name}.csv')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="huggingface checkpoint id of the model", type=str)
    parser.add_argument("--task", help="which open task to run ('SD' or'Bio')", type=str)
    args = parser.parse_args()

    model_name = args.model_id.split('/')[-1]
    anonymize_result(model_name, args.task)