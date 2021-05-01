import pandas as pd 
import numpy as np 
import random 
import pickle 
import pdb 
import tqdm 
from argparse import ArgumentParser


def get_skill_list(idx2skill, skill2idx, skill_list_path):
    skill_list = []
    idx = 0 
    f = open(skill_list_path, 'r')
    for line in f.readlines():
        skill = line.replace('\n','')
        skill2idx[skill] = idx
        idx2skill[idx] = skill
        skill_list.append(skill)
        idx += 1
    f.close()
    return skill_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dataset", default="../dataset/job_dataset.train.pkl", type=str)
    parser.add_argument("--skill_list_path", default="../dataset/processed/skill_list.txt", type=str)
    parser.add_argument("--final_dataset", default="../dataset/processed_job_dataset.train.pkl", type=str)

    args = parser.parse_args()

    # train_dataset
    df_train = pd.read_pickle(args.train_dataset)  

    skill_list = []
    idx2skill = {}
    skill2idx = {}
    skill_list = get_skill_list(idx2skill,skill2idx,args.skill_list_path)

    # skill list
    idx = 0
    df_skills = []
    print("Processing skill list")
    for skill in tqdm.tqdm(skill_list):
        binary_vector = np.zeros(len(skill_list))
        binary_vector[idx] = 1
        df_skills.append((skill,binary_vector))
        idx += 1
    
    # train_set_based_skill_correlation
    df_correlation = []

    print("Processing training samples to create correlation based samples")
    for sample in tqdm.tqdm(df_train):
        curr_skill_list = []
        for i in range(len(skill_list)):
            if(sample[1][i]==1):
                curr_skill_list.append(idx2skill[i])   
        description = ' '.join(curr_skill_list)
        df_correlation.append((description,sample[1]))

    df_train_final = df_train + df_skills + df_correlation
    random.shuffle(df_train_final)
    with open(args.final_dataset, 'wb') as f:
        pickle.dump(df_train_final, f)
    

