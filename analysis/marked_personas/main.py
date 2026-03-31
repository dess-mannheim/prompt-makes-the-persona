
from collections import Counter
from marked_words import marked_words
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import sklearn.metrics

MODELS = ['Llama-3.1-8B-Instruct', 'gemma-3-27b-it', 'Llama-3.3-70B-Instruct', 'OLMo-2-0325-32B-Instruct', 'OLMo-2-1124-7B-Instruct']
TASK = "SD" # or "Bio"

def pprint(dic): 
    full_list = []
    for word in sorted(dic, key=lambda x: x[1], reverse=True):
        full_list.append(word[0])
    return full_list

marked_words_df = pd.DataFrame(columns=['race', 'gender', 'persona_string', 'persona_type', 'model', 'group', 'marked_words', 'classifier_words', 'amount_marked_words', 'amount_classifier_words'])
accuracies_df = pd.DataFrame(columns=['race', 'gender', 'persona_string', 'persona_type', 'model', 'demographic(s)', 'accuracy'])


persona_strings = ['2nd', '3rd', 'interview']
persona_types = ['name', 'dem_descr', 'dem_cat+descr']

for model in MODELS:
    for persona_string in persona_strings:
        for persona_type in persona_types:
            # path to results
            path_to_data = f'../..data/results/{TASK}/merged_results_anonymized_user_{model}.csv'

            df = pd.read_csv(path_to_data)
            print('Data loaded successfully')

            df = df.loc[(df['persona_string'] == persona_string) & (df['persona_type'] == persona_type)]
            df = df.loc[df['LLM_AI_annotation'].isin([False, "False"])]
            df['racegender'] = df['race'] + df['gender']

            if TASK == "SD":
                df = df[df['self_description_anonymized'].apply(lambda x: isinstance(x, str))]
            else: # Bio
                df = df[df['bio_anonymized'].apply(lambda x: isinstance(x, str))]

            dv_mw = {}
            for racegender in df['racegender'].unique():
                outs = pprint(marked_words(df, [racegender], ['racegender'], ['WhiteM']))
                dv_mw[racegender] = outs
            temps = []
            for racegender in df['racegender'].unique():
                temp = pprint(marked_words(df, ['WhiteM'], ['racegender'], [racegender]))
                temps.extend(temp)
            seen = Counter(temps).most_common()
            dv_mw['WhiteM'] = [w for w, c in seen if c == 14]
            
            alldata = df.copy()
            if TASK == "SD":
                data = alldata['self_description_anonymized'].str.lower().replace('[^\w\s]', '', regex=True)   
            else: # Bio
                data = alldata['bio_anonymized'].str.lower().replace('[^\w\s]', '', regex=True)   

            dv_svm = {}
            st = 'racegender'  
            concept_data = [d for d in data]
            labels = alldata[st]
############################
            try:
                bios_data_train, bios_data_test, Y_train, Y_test = train_test_split(
                    concept_data, labels, test_size=0.2, random_state=42, stratify=labels
                )

                vectorizer = CountVectorizer(analyzer='word', min_df=0.001, binary=False)
                X_train = vectorizer.fit_transform(bios_data_train)
                X_test = vectorizer.transform(bios_data_test)
                accs = []
                feature_names = vectorizer.get_feature_names_out()
                for r in alldata[st].unique():
                    try:
                        svm = LinearSVC()
                        Y_train_bin = Y_train == r
                        svm.fit(X_train, Y_train_bin)
                        acc = sklearn.metrics.accuracy_score(Y_test == r, svm.predict(X_test))
                        accs.append(acc)
                        coef = svm.coef_[0]
                        _, names = zip(*sorted(zip(coef, feature_names)))
                        dv_svm[r] = names[-10:][::-1]

                        accuracies_df.loc[len(accuracies_df)] = {
                            'race': r[:-1], 
                            'gender': r[-1],  
                            'persona_string': persona_string,
                            'persona_type': persona_type,
                            'model': model,
                            'demographic(s)': r,
                            'accuracy': acc
                        }
                    except ValueError:
                        print(f"Skipping group {r} for model: {model}, persona_string: {persona_string}, persona_type: {persona_type} due to insufficient samples.")
                        accuracies_df.loc[len(accuracies_df)] = {
                            'race': r[:-1], 
                            'gender': r[-1],  
                            'persona_string': persona_string,
                            'persona_type': persona_type,
                            'model': model,
                            'demographic(s)': r,
                            'accuracy': np.nan
                        }
            except ValueError:
                print(f"Skipping train-test split for model: {model}, persona_string: {persona_string}, persona_type: {persona_type} due to insufficient samples.")
                for r in alldata[st].unique():
                    accuracies_df.loc[len(accuracies_df)] = {
                        'race': r[:-1], 
                        'gender': r[-1],  
                        'persona_string': persona_string,
                        'persona_type': persona_type,
                        'model': model,
                        'demographic(s)': r,
                        'accuracy': np.nan
                    }

            for group in dv_mw:
                race = group[:-1]  
                gender = group[-1]  
                top_words = [word for word in dv_mw[group] if word !='']
                #extra_words = [word for word in dv_svm.get(group, []) if word not in dv_mw[group]]
                classifier_words = [word for word in dv_svm[group] if word != '']
                marked_words_df.loc[len(marked_words_df)] = {
                    'race': race,
                    'gender': gender,
                    'persona_string': persona_string,
                    'persona_type': persona_type,
                    'model': model,
                    'group': group,
                    'marked_words': top_words,
                    'classifier_words': classifier_words, 
                    'amount_marked_words': len(top_words),
                    'amount_classifier_words': len(classifier_words)
                }
        print(f'{model}, {persona_string}, {persona_type} successful!')

marked_words_df.to_csv(f'marked_words_results_{TASK}_racegender.csv', index=False)
accuracies_df.to_csv(f'accuracies_results_{TASK}_racegender.csv', index=False)

print("Marked words and accuracies for racegender saved successfully.")