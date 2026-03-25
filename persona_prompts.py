import pandas as pd
from collections import OrderedDict
import os
from ast import literal_eval
import datetime

# SD/Bio: (2*3*2*15) + (2*3*15*10) = 1080 prompts
# Q&A: ((2*3*2*8) + (2*3*8*10))*100 = 57600 prompts

WS_PATH = "." # CHANGE to your working directory

def intersectional_persona_templates(task): # "SD", "Bio", "Q&A"
    templates, ps_list, pt_list, pv_list = [], [], [], []
    sd_json = ' Use the following JSON format: {{\"self_description\": <a detailed self-description>}}'
    bio_json = ' Use the following JSON format: {{\"username\": <social media username>, \"bio\": <social media bio, without any extra commentary whatsoever>}}'
    qa_json = ' Use the following JSON format: {{\"answer_option\": <one of the listed answer options, without any extra commentary whatsoever>}}'

    prompt_version = ["v1", "v2"]
    persona_string = ["2nd", "3rd", "interview"]
    persona_type = ["name", "dem_descr", "dem_cat+descr"]
    
    for ps in persona_string:
        for pt in persona_type:
            if pt == "name":
                    pt_t = "a person named {P4} {NM}"
            elif pt == "dem_descr":
                pt_t = "a {R} {G1}"
            elif pt == "dem_cat+descr":
                pt_t = "a person of race/ethnicity '{R}' and gender '{G2}'"
    
            for pv in prompt_version:
                
                # 2nd person
                if ps == "2nd":
                    if task == "SD":
                        task_t = "How would you describe yourself? Only respond with the self-description."+sd_json
                    elif "Q&A" in task:
                        if not " who lives in the U.S" in pt_t:
                            pt_t = pt_t + " who lives in the U.S"
                        task_t = "How would you answer the following question:\n{question}\nOnly respond with one of the listed answer options."+qa_json 
                    elif task == "Bio":
                        task_t = "What is your social media username and bio? Only respond with the username and bio."+bio_json # JSON
                    
                    if pv == "v1":
                        persona_t = f"You are {pt_t}. {task_t}"  # US
                    elif pv == "v2":
                        persona_t = f"Act as {pt_t}. {task_t}"
                        
                # 3rd person        
                elif ps == "3rd":
                    if task == "SD":
                        if pv == "v1":
                            persona_t = f"Think of {pt_t}. How would {{P1}} describe {{P2}}? Only respond with the self-description."+sd_json
                        elif pv == "v2":
                            persona_t = f"How would {pt_t} describe {{P2}}? Only respond with the self-description."+sd_json
                    elif "Q&A" in task:
                        if pv == "v1":
                            persona_t = f"Think of {pt_t} who lives in the U.S. How would {{P1}} answer the following question:\n{{question}}\nOnly respond with one of the listed answer options."+qa_json 
                        elif pv == "v2":
                            persona_t = f"How would {pt_t} who lives in the U.S answer the following question:\n{{question}}\nOnly respond with one of the listed answer options."+qa_json 
                    elif task == "Bio":
                        if pv == "v1":
                            persona_t = f"Think of {pt_t}. What would be {{P3}} social media username and bio? Only respond with the username and bio."+bio_json # JSON
                        if pv == "v2":
                            persona_t = f"What would be the social media username and bio of {pt_t}? Only respond with the username and bio."+bio_json # JSON
                            
                # interview
                elif ps == "interview":
                    if task == "SD":
                        task_t = "Interviewer: How would you describe yourself? Only respond with the self-description."+sd_json
                    elif "Q&A" in task:
                        task_t = """Interviewer: Where do you live?
                        Interviewee: I live in the U.S.
                        Interviewer: {question}
                        Only respond with one of the listed answer options."""+qa_json
                    elif task == "Bio":
                        task_t = "Interviewer: What is your social media username and bio? Only respond with the username and bio."+bio_json # JSON

                    if pv == "v1":
                        persona_t = "You are an interviewee. Based on your previous answers, provide an answer to the last question.\n"
                        if pt == "name":
                            persona_t += f"""Interviewer: Please tell me your name.
                            Interviewee: My name is {{P4}} {{NM}}.
                            {task_t}
                            Interviewee:..."""
                        elif pt == "dem_descr":
                            persona_t += f"""Interviewer: Do you identify as {{G_list1}}?
                            Interviewee: I identify as a {{G1}}.
                            Interviewer: Do you consider yourself to be {{R_list1}}?
                            Interviewee: I consider myself to be {{R}}.
                            {task_t}
                            Interviewee:..."""  
                        elif pt == "dem_cat+descr":
                            persona_t += f"""Interviewer: What gender do you identify as? Do you identify as {{G_list2}}?
                            Interviewee: I identify as '{{G2}}'.
                            Interviewer: What race/ethnicity do you consider yourself to be? Do you consider yourself to be {{R_list2}}?
                            Interviewee: I consider myself to be '{{R}}'.
                            {task_t}
                            Interviewee:...""" 
                        persona_t = "\n".join(line.strip() for line in persona_t.splitlines()) # remove whitespace

                    elif pv == "v2":
                        persona_t = "You are Speaker 2. Based on your previous answers, provide an answer to the last question.\n"
                        if pt == "name":
                            persona_t += f"""Speaker 1: What is your name?
                            Speaker 2: My name is {{P4}} {{NM}}.
                            {task_t}
                            Speaker 1:..."""
                        elif pt == "dem_descr":
                            persona_t += f"""Speaker 1: Are you {{G_list1}}?
                            Speaker 2: I am a {{G1}}.
                            Speaker 1: Are you {{R_list1}}?
                            Speaker 2: I am {{R}}.
                            {task_t}
                            Speaker 2:..."""  
                        elif pt == "dem_cat+descr":
                            persona_t += f"""Speaker 1: What is your gender? Are you {{G_list2}}?
                            Speaker 2: My gender is '{{G2}}'.
                            Speaker 1: What is your race/ethnicity? Are you {{R_list2}}?
                            Speaker 2: My race/ethnicity is '{{R}}'.
                            {task_t}
                            Speaker 2:..."""
                        persona_t = persona_t.replace("Interviewer", "Speaker 1").replace("Interviewee", "Speaker 2")
                        persona_t = "\n".join(line.strip() for line in persona_t.splitlines()) # remove whitespace
                        
                templates.append(persona_t)
                ps_list.append(ps)
                pt_list.append(pt)
                pv_list.append(pv)
                
    df = pd.DataFrame({"task": task,
                       "template": templates,
                       "persona_string": ps_list,
                       "persona_type": pt_list,
                       "prompt_version": pv_list,
                      })
    return df


# racially coded surnames
def get_surnames(race):
    if race == "White":
        names = ["Olson", "Snyder", "Wagner", "Meyer", "Schmidt", "Ryan", "Hansen", "Hoffman", "Johnston", "Larson"]
    elif race == "Black":
        names = ["Smalls", "Jeanbaptiste", "Diallo", "Kamara", "Pierrelouis", "Gadson", "Jeanlouis", "Bah", "Desir", "Mensah"]
    elif race == "Asian":
        names = ["Nguyen", "Kim", "Patel", "Tran", "Chen", "Li", "Le", "Wang", "Yang", "Pham"]
    elif race == "Hispanic":
        names = ["Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez", "Sanchez", "Ramirez", "Torres"]
    elif race == "Middle-Eastern":
        names = ["Khan", "Ali", "Ahmed", "Hassan", "Yılmaz", "Kaya", "Demir", "Mohammadi", "Hosseini", "Ahmadi"]
    else:
        print("no names available")
    return names

# demographic descriptors used for each task
def get_demographic_descriptors(task):
    if task == "SD" or task == "Bio":
        R = ['White', 'Black', 'Asian', 'Middle-Eastern','Hispanic']
        G = ['F', 'M', 'N']
        R_list1 = "White, Black, Asian, Middle-Eastern or Hispanic"
        R_list2 = "'White', 'Black', 'Asian', 'Middle-Eastern' or 'Hispanic'"
        G_list1 = "a man, a woman or a nonbinary person"
        G_list2 = "'male', 'female' or 'nonbinary'"
        G1 =  {"M": "man", "F": "woman", "N": "nonbinary person"}
        G2 =  {"M": "male", "F": "female", "N": "nonbinary"}
        P1 = {"M": "he", "F": "she", "N": "they"}
        P2 = {"M": "himself", "F": "herself", "N": "themselves"}
        P3 = {"M": "his", "F": "her", "N": "their"}
        P4 = {"M": "Mr.", "F": "Ms.", "N": "Mx."}
        NM = {race: get_surnames(race) for race in R}
        
    elif "Q&A" in task:
        R = ['White', 'Black', 'Asian', 'Hispanic']
        G = ['F', 'M']
        R_list1 = "White, Black, Asian or Hispanic"
        R_list2 = "'White', 'Black', 'Asian' or 'Hispanic'"
        G_list1 = "a man or a woman"
        G_list2 = "'male' or 'female'"
        G1 =  {"M": "man", "F": "woman"}
        G2 =  {"M": "male", "F": "female"}
        P1 = {"M": "he", "F": "she"}
        P2 = {"M": "himself", "F": "herself"}
        P3 = {"M": "his", "F": "her"}
        P4 = {"M": "Mr.", "F": "Ms."}
        NM = {race: get_surnames(race) for race in R}
        
    else:
        print("Task not available")
        return
    return R, G, R_list1, R_list2, G_list1, G_list2, G1, G2, P1, P2, P3, P4, NM

# prepare OpinionQA questions
def get_qa(qu_num, qa_path):
    if qa_path is None:
        qa_path = f"{WS_PATH}/data/OpinionQA.csv"  
    qa_df = pd.read_csv(qa_path, converters={"option_mapping": literal_eval, "references": literal_eval, "option_ordinal": literal_eval})
    if qu_num is not None:  # select only a subset of questions
        qa_df = qa_df[-qu_num:]
    return format_qa(qa_df)  # dict with formatted questions and mapping keys

def format_qa(df, question_col="question", option_col="references", key_col="key"):
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    questions = {"question":{}, "options":{}}
    for _,row in df.iterrows():
        options =''.join(f"\n{x}. {option}" for x, option in zip(letters[:len(row[option_col])], row[option_col]))  # format answer options
        questions["question"][row[key_col]] = row[question_col] + options
        questions["options"][row[key_col]] = tuple(letters[:len(row[option_col])])
    return questions


# enrich prompt templates with demographic descriptors
def fill_prompts(template_df, task, qu_num=100, qa_path=None):
    # collect task-specific demographics
    R, G, R_list1, R_list2, G_list1, G_list2, G1, G2, P1, P2, P3, P4, NM = get_demographic_descriptors(task)
    if "Q&A" in task:
        qa = get_qa(qu_num, qa_path)
        questions = qa["question"]  # collect the questions
        options = qa["options"]
        
    templates = template_df["template"].to_list()
    dfs = []

    for template in templates: 
        prompts, race, gender = [], [], []
        if "Q&A" in task:
            qa_key = []
            qa_options = [] 
        for g in G:
            for r in R:
                if " name" in template:
                    for name in NM[r]:
                        if "Q&A" in task:
                            for key, q in questions.items():  # loop through all questions
                                prompts.append(template.format(R=r, G=g, R_list1=R_list1, R_list2=R_list2, G_list1=G_list1, G_list2=G_list2, \
                                                               G1=G1[g], G2=G2[g], P1=P1[g], P2=P2[g], P3=P3[g], P4=P4[g], NM=name, question=q))
                                race.append(r)
                                gender.append(g)   
                                qa_key.append(key)
                                qa_options.append(options[key]) # answer options with corresponding key
                        else:
                            prompts.append(template.format(R=r, G=g, R_list1=R_list1, R_list2=R_list2, G_list1=G_list1, G_list2=G_list2, \
                                                           G1=G1[g], G2=G2[g], P1=P1[g], P2=P2[g], P3=P3[g], P4=P4[g], NM=name))
                            race.append(r)
                            gender.append(g)
                else:
                    if "Q&A" in task:
                        for key, q in questions.items():  # loop through all questions
                            prompts.append(template.format(R=r, G=g, R_list1=R_list1, R_list2=R_list2, G_list1=G_list1, G_list2=G_list2, \
                                            G1=G1[g], G2=G2[g], P1=P1[g], P2=P2[g], P3=P3[g], question=q))
                            race.append(r)
                            gender.append(g)   
                            qa_key.append(key)
                            qa_options.append(options[key]) # answer options with corresponding key
                    else:
                        prompts.append(template.format(R=r, G=g, R_list1=R_list1, R_list2=R_list2, G_list1=G_list1, G_list2=G_list2, \
                                        G1=G1[g], G2=G2[g], P1=P1[g], P2=P2[g], P3=P3[g]))
                        race.append(r)
                        gender.append(g) 
        prompts = [p.replace("a Asian", "an Asian") for p in prompts] 
        
        if "Q&A" in task:
            dfs.append(pd.DataFrame({"prompt": prompts, "template": template, "race": race, "gender": gender, "qa_key": qa_key, "qa_options": qa_options}))                
        else:
            dfs.append(pd.DataFrame({"prompt": prompts, "template": template, "race": race, "gender": gender})) 

    df = pd.concat(dfs, ignore_index=True) 
    all_prompts = pd.merge(df, template_df, on="template", how ="left").reset_index(drop=True)
    all_prompts.index.name = 'prompt_id'
    return all_prompts


def create_first_round(task, qu_num=100, qa_path=None): # specify task
    
    template_df = intersectional_persona_templates(task)
 
    # fill templates with demographic details        
    prompt_df = fill_prompts(template_df, task=task, qu_num=qu_num, qa_path=qa_path)
    # save the prompts
    dir_path = f'{WS_PATH}/data'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    prompt_df.to_csv(f'{dir_path}/prompts_{task}.csv')

    prompts = prompt_df[~(prompt_df["persona_type"] == "name")]
    prompts_name = prompt_df[prompt_df["persona_type"] == "name"] # prompts with names are separate 

    # i is the prompt id
    chats = OrderedDict({
        i: [{"role": "user", "content": row["prompt"]}]
        for i, row in prompts.iterrows()
    }) 
    chats_name = OrderedDict({
        i: [{"role": "user", "content": row["prompt"]}]
        for i, row in prompts_name.iterrows()
    }) 

    if task == "Q&A":
        answer_options = OrderedDict({
            i: row["qa_options"]
            for i, row in prompt_df.iterrows() # we don't care about names here
        }) 
        return chats, chats_name, answer_options
    return chats, chats_name 


def create_second_round(first_round, output, confidence_prompt):
    # concatenate original prompt with first-round output & confidence prompt
    chats = OrderedDict({
        prompt_id: prompt + [
                {'role': 'assistant', 'content': req.outputs[0].text},
                {'role': 'user', 'content': confidence_prompt}
        ]
        for (prompt_id, prompt), req in list(zip(first_round.items(), output))
    })
    return chats

def create_second_round_all_answers(first_round, answer_options, confidence_prompt):    
    # concatenate original prompt with all possible answer options & confidence prompt for Q&A setting
    chats = OrderedDict({
        (prompt_id, option): prompt + [
                {'role': 'assistant', 'content': f'{{"answer_option": {option}}}'},
                {'role': 'user', 'content': confidence_prompt}
        ]
        for prompt_id, prompt in first_round.items()
        for option in answer_options[prompt_id]
    })
    return chats

            
if __name__ == "__main__":
    # save prompts for all tasks
    for task in ["SD", "Bio", "Q&A"]:
        create_first_round(task=task)
