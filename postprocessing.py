import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from ast import literal_eval
import argparse
import re

WS_PATH = "." # CHANGE to your working directory

def postprocess_results(model_name, task):
    # read results
    results_path = f"{WS_PATH}/data/results/{task}"
    res_df = pd.read_csv(f"{results_path}/results_{model_name}.csv", index_col=0)

    if task != "Q&A":
        # extract confidence score
        pattern_conf = r"\b(?:0(?:\.\d+)?|1(?:\.0)?)\b" # match a single floating-point number strictly between 0 and 1
        res_df['confidence'] = res_df.conf_out.str.extract(f"({pattern_conf})").astype(float)
    
        # check for missing values
        print(f"Missing confidence values: {len(res_df[res_df.confidence.isna()])}")
        print(f"Missing generated text: {len(res_df[res_df.generated_text.isna()])}")
        # set confidence for missing text to nan as well
        res_df.loc[res_df['generated_text'].isna(), 'confidence'] = np.nan
        res_df.loc[res_df['generated_text'].isna(), 'out_perplexity'] = np.nan

    if task == "Q&A_full":
        res_df = postprocess_QA_full(res_df)
    elif task == "Bio":
        res_df = postprocess_Bio(res_df)
    elif task == "SD":
        res_df = postprocess_SD(res_df)

    # prompts
    prompt_df = pd.read_csv(f"{WS_PATH}/data/prompts_{task}.csv", index_col=0)
    prompt_df.index.name = 'prompt_id'
    
    # merge prompts with prompt perplexity
    perpl_df = pd.read_csv(f"{results_path}/prompt_perplexities_{model_name}.csv", index_col=0)
    prompt_perpl_df = pd.merge(prompt_df, perpl_df, on='prompt_id', how='left')
    
    # convert prompt perplexity to entropy
    prompt_perpl_df["prompt_entropy"] = prompt_perpl_df["prompt_perplexity"].apply(lambda x: np.log(x))
    
    # merge prompts and results
    merged_df = pd.merge(res_df, prompt_perpl_df, on='prompt_id', how='left')

    if task == "Q&A": 
         # process only after merging
        merged_df = postprocess_QA(merged_df) 
    elif task == "Q&A_full":
        merged_df['qa_options'] = merged_df['qa_options'].apply(literal_eval)
        merged_df = get_answer_dist(merged_df)
        merged_df = get_confidence_dist(merged_df)
    else:
        # convert output perplexity to entropy (we don't collect outputs for OpinionQA)
        merged_df["out_entropy"] = merged_df["out_perplexity"].apply(lambda x: np.log(x))
 
    # save
    merged_df.to_csv(f"{results_path}/merged_results_{model_name}.csv")  
    return merged_df, prompt_perpl_df 


# extract self-description
def postprocess_SD(df):

    def extract_fields(text):
        if not isinstance(text, str): # check for unexptected output
            return pd.Series([None, None])
        
        # Replace typographical quotes with standard quotes
        text = re.sub(r"[‘’‚‛‹›]", "'", text) # typographical single quotes
        text = re.sub(r'[“”„‟]', '"', text) # typographical double quotes

        sd_match = re.search(r'''["']?\bself[ _\(]?description\b["']?\s*:\s*(["'])(.*?(?:(?<!\\)(?:\\\\)*)(?=\1))\1|["']?\bself[ _\(]?description\b["']?\s*:\s*([^,}\n]+)''', text)
        if sd_match:
            sd = sd_match.group(2).lstrip('\'"').rstrip('\'"').strip() if sd_match.group(2) is not None else sd_match.group(3).lstrip('\'"').rstrip('\'"').strip()
            sd = sd.replace(r'\"', '"').replace(r'\'', "'") # handle unnecessary escaping
        else:
            sd = None
        return pd.Series([sd])

    df[['self_description']] = df['generated_text'].apply(extract_fields)
    
    # Note: not needed, as we use an LLM to do an AI check later
    df = ai_check(df, 'self_description')
    return df


# extract username and Bio
def postprocess_Bio(df):
    
    def extract_fields(text):
        if not isinstance(text, str): # check for unexptected output
            return pd.Series([None, None])
    
        # Replace typographical quotes with standard quotes
        text = re.sub(r"[‘’‚‛‹›]", "'", text) # typographical single quotes
        text = re.sub(r'[“”„‟]', '"', text) # typographical double quotes
    
        # extract username
        username_match = re.search(r'["\']?username["\']?\s*:\s*["\']?([^"\',\n\r{}]+)["\']?', text)
        username = username_match.group(1).lstrip('\'"').rstrip('\'"').strip() if username_match else None

        # First try quoted bio (handles "I'm", etc.)
        bio_match = re.search(r'["\']?bio["\']?\s*:\s*"([^"]+)"', text)

        if not bio_match:
            # Fallback: unquoted bio, stop at }, newline, etc.
            bio_match = re.search(r'["\']?bio["\']?\s*:\s*([^}\n\r]+)', text)
        
        bio = bio_match.group(1).lstrip('\'"').rstrip('\'"').strip() if bio_match else None
    
        return pd.Series([username, bio])
    
    # extract username and bio from text
    df[['username', 'bio']] = df['generated_text'].apply(extract_fields)

    # Note: not needed, as we use an LLM to do an AI check later
    df = ai_check(df, "bio")
    return df


# extract anwer option and confidence distribution of OpinionQA
def postprocess_QA(df):
    
    # discard all options that are not valid
    df['answer_dist'] = df['answer_dist'].apply(lambda x: literal_eval(x))
    df['answer_dist_clean'] = df.apply(lambda row: {k: row['answer_dist'][k] for k in row['qa_options'] if k in row['answer_dist']}, axis=1)
    
    # extract confidence score
    df['conf_out_dist'] = df['conf_out_dist'].apply(lambda x: literal_eval(x))

    pattern_conf = r"\b(?:0(?:\.\d+)?|1(?:\.0)?)\b"

    df['confidence_dist'] = df['conf_out_dist'].apply(
        lambda d: {
            k: float(re.search(pattern_conf, v).group(0)) if re.search(pattern_conf, v) else None 
            for k, v in d.items()
        }
    )
        
    return df

def get_confidence_dist(df):
    # average confidence per prompt id and answer option
    avg_conf = df.groupby(['prompt_id', 'answer_option'])['confidence'].mean().reset_index()
    avg_conf_wide = avg_conf.pivot(index='prompt_id', columns='answer_option', values='confidence')
    
    # get all possible answer options
    qa_options_map = df[['prompt_id', 'qa_options']].drop_duplicates()
    avg_conf_wide = avg_conf_wide.merge(qa_options_map, on='prompt_id')
    
    # disribution with only valid answer options, fill missing with 0
    def build_conf_dict(row):
        options = row['qa_options']
        return {
            opt: float(row[opt]) if opt in row and pd.notna(row[opt]) else 0.0
            for opt in options
        }
    
    avg_conf_wide['confidence_dist'] = avg_conf_wide.apply(build_conf_dict, axis=1)
    
    # merge back to original df
    df = df.merge(avg_conf_wide[['prompt_id', 'confidence_dist']], on='prompt_id')
    return df

def get_answer_dist(df):
    #count answer options per prompt id
    count_df = df.groupby(['prompt_id', 'answer_option']).size().reset_index(name='count')
    count_wide = count_df.pivot(index='prompt_id', columns='answer_option', values='count')
    
    # get answer options per prompt id
    qa_options_map = df[['prompt_id', 'qa_options']].drop_duplicates()
    count_wide = count_wide.merge(qa_options_map, on='prompt_id')

    # include only options from qa_options, fill missing with 0
    def build_count_dict(row):
        options = row['qa_options']
        return {
            opt: int(row[opt]) if opt in row and pd.notna(row[opt]) else 0
            for opt in options
        }
    
    count_wide['answer_dist_counts'] = count_wide.apply(build_count_dict, axis=1)
    df = df.merge(count_wide[['prompt_id', 'answer_dist_counts']], on='prompt_id')
    return df


# extract answer option of OpinionQA
def postprocess_QA_full(df):

    def extract_fields(text):
        if not isinstance(text, str): # check for unexptected output
            return pd.Series([None])
            
        # extract the text from "answer_option"
        pattern = r'["\']?\banswer[ _\(]?option\b["\']?\s*:\s*(?:"([^"]*?)"|([^,}\n]+))'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = (match.group(1) or match.group(2) or "").strip()
    
        # match last uppercase A–J (with optional period)
        letter_match = re.search(r'\b([A-J])\.?["\']?\s*$', answer)
        
        if letter_match:
            letter = letter_match.group(1)
    
            # only accept A or I if they are the *entire* value (or with a dot)
            if letter in {"A", "I"} and answer not in {letter, f"{letter}."}:
                return pd.Series([None])
            return pd.Series([letter])
    
        return pd.Series([None])
        
    df[['answer_option']] = df['generated_text'].apply(extract_fields)  

    # Note: not needed as we use an LLM to do an AI check later
    df = ai_check(df, 'answer_option')
    return df


# check how often the model answered as an AI using a regex
def ai_check(df, content_col): 
    ai_words = ["an AI",  "an artificial intelligence", "an ai assistant", "an AI assistant" "a virtual assistant", "a language model",  "a large language model", "a Language Model", 
                "a LM", "a LLM", "a Chatbot", "a computer program", "a conversational AI", "a crafted AI", "a multilingual AI", "a digital assistant"] 
    pattern_ai = r'(?i)' + '|'.join([r'\b' + re.escape(w) + r'\b' for w in ai_words])

    def ai_labels(row, content_col, pattern_ai):
        text = row['generated_text']
        content = row[content_col] 
    
        if pd.isna(text):
            return None
    
        if not re.search(pattern_ai, str(text)):
            return "no"
    
        if pd.isna(content):
            return "yes"
    
        if re.search(pattern_ai, str(content)):
            return "yes"
    
        return "partial"

    ai_classifier = partial(ai_labels, content_col=content_col, pattern_ai=pattern_ai)
    df["regex_AI"] = df.apply(ai_classifier, axis=1)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="huggingface checkpoint id of the model", type=str)
    parser.add_argument("--task", help="which task to run ('SD', 'Bio', 'Q&A' and Q&A_full", type=str)
    args = parser.parse_args()

    model_name = args.model_id.split('/')[-1]
    postprocess_results(model_name, args.task)


    
    