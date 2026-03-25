from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from persona_prompts import create_first_round, create_second_round, create_second_round_all_answers
from enum import Enum
from collections import OrderedDict
from pydantic import create_model
import numpy as np
import pandas as pd
import random
import json
import argparse
import datetime

WS_PATH = "." # CHANGE to your working directory
ANSWER_CONFIDENCE_TWO_STEP = """How confident are you in the answer that you just provided?
Reply only with a confidence score between 0.0 and 1.0 with no other words or explanation. Use the following JSON format:
{"confidence": <the confidence in your answer between 0.0 and 1.0, without any extra commentary whatsoever; just the confidence score!>,}
"""
SEED = 33 # seed for reproducibility
QU_NUM = 100 # number of questions to use from Opinion QA

def get_output_logprobs(output):
    res = []
    for req in output:
        logprobs = [list(x.values())[0].logprob for x in req.outputs[0].logprobs if x is not None]
        res.append(logprobs)
    return res

def get_prompt_logprobs(output):
    res = []
    for req in output:
        logprobs = [list(x.values())[0].logprob for x in req.prompt_logprobs if x is not None]
        res.append(logprobs)
    return res

def perplexity_from_logprobs(logprobs):
    return np.exp(-1*np.mean(logprobs))

def get_perplexity(output, text="generated"):
    if text == "generated":
        logprobs = get_output_logprobs(output)
    elif text == "prompt":
        logprobs = get_prompt_logprobs(output)
    else:
        print("Specify valid text type")
        return
    perplexities = [perplexity_from_logprobs(req) for req in logprobs]  
    return perplexities

def get_qa_logprobs(output):
    res = []
    for req in output:
        answer_dict = {x.decoded_token: x.logprob for x in req.outputs[0].logprobs[5].values()} # dict with answer options and corresponding logprobs (logprobs in structured output at index 5)
        res.append(answer_dict)
    return res

def get_confidence_dist(output, second_round, prompt_ids):
    conf_out_list = [req.outputs[0].text for req in output] # extract text from the output
    conf_out_dict = {k:v for (k,v) in zip(list(second_round.keys()), conf_out_list)} # map the output back to prompt id and answer option
    conf_out_dict_nested = {prompt_id: {option: conf_out_dict[(prompt_id, option)] for (i, option) in conf_out_dict if i == prompt_id} for prompt_id, _ in conf_out_dict} # restructure to get the confidence distribution for each prompt id
 
    return [conf_out_dict_nested[p_id] for p_id in prompt_ids] # ensure the correct ordering of the outputs
    
    
def run_model(args):
    # task specific settings
    if "Q&A" in args.task:
        max_mlen = 500
    else:
        max_mlen = 1500

    # model configuration  
    llm = LLM(model=args.model_id, 
              enforce_eager=True, 
              max_model_len=max_mlen,
              gpu_memory_utilization=0.8, 
              #guided_decoding_backend="xgrammar",
              generation_config='auto',
              tensor_parallel_size=args.tensor_parallel_size)

    # set sampling parameters
    sampling_params = llm.get_default_sampling_params()
    sampling_params.n = 1
    sampling_params.logprobs=0
    sampling_params.max_tokens=1000
    print(sampling_params)

    # get prompts 
    print("generate chats...")
    if args.task == "Q&A":
        first_round, first_round_name, answer_options = create_first_round(task=args.task, qu_num=QU_NUM, qa_path=args.qa_path) # get the answer options as well
        first_round.update(first_round_name) # we run Q&A only once, regardless if names are included or not
    else:
        first_round, first_round_name = create_first_round(task=args.task, qu_num=QU_NUM, qa_path=args.qa_path)
    print("...done")

    # collect the prompt ids
    prompt_ids = list(first_round.keys())
    if args.k_name > 0:
        prompt_ids_name = list(first_round_name.keys())
    
    # set seed for reproducibility
    random.seed(SEED)
    seeds = random.sample(range(100000), k=args.k)

    all_gens = []
    for i,seed in enumerate(seeds):
        # full first round of conversations with fixed seed
        sampling_params.seed = seed
        print(f"{datetime.datetime.now()} - {i}: {seed}")
        
        if i == 0:
            sampling_params.prompt_logprobs=0 # we only gather prompt logprobs for the first seed
        else:
            sampling_params.prompt_logprobs=None

        # Q&A (use logprobs instead of multiple runs per prompt)
        if args.task == "Q&A":
            second_round = create_second_round_all_answers(first_round, answer_options, ANSWER_CONFIDENCE_TWO_STEP)
            max_options = len(set(o for _,o in second_round.keys())) # maximum number of answer options for the selected questions
            first_round_options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:max_options]
            
            sampling_params.logprobs = len(first_round_options)
            sampling_params.prompt_logprobs=0

            # create a dynamic Enum with the enum functional API
            AnswerOptionEnum = Enum(
                "AnswerOptionEnum",
                {option: option for option in first_round_options}
                )
            
            # create a (dynamic) pydantic model from the answer options
            outputs = {"answer_option": (AnswerOptionEnum, ...)}
            ResultModel = create_model("ResultModel", **outputs)
            json_schema = ResultModel.model_json_schema()
            decoding_params = GuidedDecodingParams(json=json_schema)      
            sampling_params.guided_decoding=decoding_params

            # first round with structured outputs: gather model answers to OpinionQA questions
            out = llm.chat(list(first_round.values()), sampling_params=sampling_params, use_tqdm=False,) 
            prompt_perplexity = get_perplexity(out, "prompt")
            answer_dist = get_qa_logprobs(out)  # answer option distributions for each prompt
            print(f"{datetime.datetime.now()} - finished first round")

            pd.DataFrame({"prompt_id":prompt_ids, "prompt_perplexity":prompt_perplexity}).to_csv(f"{args.out_path}/prompt_perplexities_{args.model_id.split('/')[-1]}.csv") 
            print("saved perplexity")


            # second round: how confident is the model in its answer?
            sampling_params.temperature = 0 # greedy decoding
            sampling_params.logprobs = 0
            sampling_params.prompt_logprobs=None
            sampling_params.guided_decoding=None

            conf_out = llm.chat(list(second_round.values()), sampling_params, use_tqdm=False,) # each request is an answer option in combination with a prompt
            confidence_dist = get_confidence_dist(conf_out, second_round, prompt_ids)  # get the confidence for each possible answer option
            print(f"{datetime.datetime.now()} - finished second round")                   

            res = pd.DataFrame({"prompt_id":prompt_ids, "answer_dist": answer_dist, "conf_out_dist": confidence_dist})
            res["seed"] = seed

        # Bio & SD & Q&A_full (multiple runs per prompt)
        else:    
            # first round (no names)
            out = llm.chat(list(first_round.values()), sampling_params, use_tqdm=False,)
    
            if i < args.k_name: # first round with names
                out_name = llm.chat(list(first_round_name.values()), sampling_params, use_tqdm=False,)

            print(f"{datetime.datetime.now()} - first round")
    
            # for the first run, compute the prompt perplexity
            if i == 0: 
                if i < args.k_name:
                    prompt_perplexity = get_perplexity(out, "prompt") + get_perplexity(out_name, "prompt") # output is a list
                else:
                    prompt_perplexity = get_perplexity(out, "prompt")
                    prompt_ids_name=[]
                    
                pd.DataFrame({"prompt_id":prompt_ids+prompt_ids_name, "prompt_perplexity":prompt_perplexity}).to_csv(f"{args.out_path}/prompt_perplexities_{args.model_id.split('/')[-1]}.csv") 
                print("saved perplexity")                
    
            # collect outputs and output perplexities
            if i < args.k_name:
                out_perplexity = get_perplexity(out) + get_perplexity(out_name)
                gen_df = pd.DataFrame({"prompt_id":prompt_ids+prompt_ids_name, 
                                       "out_perplexity":out_perplexity, 
                                       "generated_text":[req.outputs[0].text for req in out]+[req.outputs[0].text for req in out_name]})
            else:
                out_perplexity = get_perplexity(out)
                gen_df = pd.DataFrame({"prompt_id":prompt_ids, "out_perplexity":out_perplexity, "generated_text":[req.outputs[0].text for req in out]})
            gen_df["seed"] = seed
    
            # second round: how confident is the model in its answer?
            sampling_params.prompt_logprobs=None
            second_round = create_second_round(first_round, out, ANSWER_CONFIDENCE_TWO_STEP)
            conf_out = llm.chat(list(second_round.values()), sampling_params, use_tqdm=False,)
            conf_df = pd.DataFrame({"prompt_id":list(second_round.keys()), "conf_out":[req.outputs[0].text for req in conf_out]})
    
            if i < args.k_name:
                second_round_name = create_second_round(first_round_name, out_name, ANSWER_CONFIDENCE_TWO_STEP)
                conf_out_name = llm.chat(list(second_round_name.values()), sampling_params, use_tqdm=False,)
                conf_df = pd.DataFrame({"prompt_id":list(second_round.keys())+list(second_round_name.keys()), 
                                        "conf_out":[req.outputs[0].text for req in conf_out]+[req.outputs[0].text for req in conf_out_name]})

            print(f"{datetime.datetime.now()} - second round")

            # collect intermediate results
            all_gens.append(gen_df.merge(conf_df, on='prompt_id', how='inner'))
            res_interm = pd.concat(all_gens, ignore_index=True)
            res_interm.to_csv(f"{args.out_path}/results_intermediate_{args.model_id.split('/')[-1]}.csv")

    if args.task != "Q&A":
        res = pd.concat(all_gens, ignore_index=True)

    res.to_csv(f"{args.out_path}/results_{args.model_id.split('/')[-1]}.csv")
    del llm
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="huggingface checkpoint id of the model to run", type=str)
    parser.add_argument("--task", help="which task to run (currently 'SD', 'Bio', 'Q&A' and Q&A_full", type=str)
    parser.add_argument("--k", default=100, help="number of runs with different random seeds", type=int)
    parser.add_argument("--k_name", default=10, help="number of runs with different random seeds for the name prompts", type=int)
    parser.add_argument("--tensor_parallel_size", default=1, help="number of GPUs to use for distributed setup (tensor parallelism)", type=int)
    parser.add_argument("--qa_path", default=None, help="path to the location of the opinionQA csv file", type=str)
    args = parser.parse_args()
    
    # set output path
    args.out_path = f"{WS_PATH}/data/results/{args.task}"  
     
    # only one run for Q&A
    if args.task == "Q&A":
        args.k = 1
        args.k_name = 0  # no separate run for names needed
        print("set --k and --k_name to 1")
    
    # run experiments with specified model
    run_model(args)


