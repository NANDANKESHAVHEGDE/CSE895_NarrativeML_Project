import os
import re
import json
import pandas as pd
from tqdm import tqdm
from typing import Literal
from evaluate import load

def create_cuasal_vidqa_ground_truth_file(ground_truth_dir:str, output_dir:str=None):
    #Extract video ids list
    video_ids = [vid for vid in sorted(os.listdir(ground_truth_dir)) if not vid.endswith('.ipynb_checkpoints')]
    # Creates a new list without the first element

    
    #Instantinate ground truth DataFrame
    ground_truth_df = pd.DataFrame(columns=["video_id"])
    ground_truth_df["video_id"] = video_ids
    ground_truth_df["descriptive"] = ""
    ground_truth_df["explanatory"] = ""
    ground_truth_df["predictive_answer"] = ""
    ground_truth_df["predictive_reason"] = ""
    ground_truth_df["predictive"] = ""
    ground_truth_df["counterfactual_answer"] = ""
    ground_truth_df["counterfactual_reason"] = ""
    ground_truth_df["counterfactual"] = ""

    #Run through video ids to get ground truth answer for each video id
    for video_id in tqdm(video_ids):  
        answer_file = os.path.join(ground_truth_dir, video_id, "answer.json")
        with open(answer_file, "r") as afile:
            answers = json.load(afile)
        
        #Gathering the answer
        for answer_type in answers.keys():
            if answer_type == "descriptive" or answer_type == "explanatory":
                answer = answers[answer_type]["answer"]
                ground_truth_df.loc[ground_truth_df['video_id'] == video_id, answer_type] = f"{answer}"
            else:
                answer = answers[answer_type]["answer"]
                reason = answers[answer_type]["reason"]
                ground_truth_df.loc[ground_truth_df['video_id'] == video_id, f"{answer_type}_answer"] = f"{answer}"
                ground_truth_df.loc[ground_truth_df['video_id'] == video_id, f"{answer_type}_reason"] = f"{reason}"
                ground_truth_df.loc[ground_truth_df['video_id'] == video_id, answer_type] = f"{answer}_{reason}"
    
    #Saving the file
    ground_truth_df.to_csv(f"{output_dir}/causal_vidqa_test_ground_truth.csv", index=False)
    return 1

def fix_json_errors(json_text):
    # Fix common JSON errors
    json_text = json_text.strip()  # Remove leading/trailing spaces
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)  # Remove trailing commas
    json_text = json_text.replace("'", '"')  # Convert single quotes to double quotes
    json_text = re.sub(r'(?<!\\)"([^"]*?)"(?=\s*:)', r'"\1"', json_text)  # Ensure keys are properly quoted
    json_text = re.sub(r'(?<!\\)"([^"]*?)"(?=\s*[:,})])', r'"\1"', json_text)  # Ensure values are properly quoted
    json_text = re.sub(r'\s+', ' ', json_text)  # Remove excessive whitespace
    json_text = json_text.replace("\n", " ")  # Remove newlines
    
    # Check if there are missing closing braces/brackets
    open_braces = json_text.count("{")
    close_braces = json_text.count("}")
    
    if open_braces > close_braces:
        json_text += "}" * (open_braces - close_braces)  # Add missing closing braces

    open_brackets = json_text.count("[")
    close_brackets = json_text.count("]")
    
    if open_brackets > close_brackets:
        json_text += "]" * (open_brackets - close_brackets)  # Add missing closing brackets

    try:
        json_data = json.loads(json_text)
        return json_text
    except:
        open_braces = json_text.count("{")
        close_braces = json_text.count("}")
        if open_braces == close_braces == 5:
            match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if match:
                return match.group(0).strip()
        elif open_braces == close_braces == 4:
            json_text = "{\n" + json_text + "\n}"

    return json_text

def create_causal_vidqa_prediction_file(prediction_dir:str, output_dir:str=None,
                                        data_mode:Literal["narrative", "narrativeml", "both"]=None, suffix:str=None,
                                        des_nar_narml_csv:str=None):
    #Extract video ids list
    if des_nar_narml_csv == None:
        video_ids = os.listdir(prediction_dir)
        video_ids = sorted(video_ids)
        video_ids = video_ids[1:]
    else:
        des_nar_narml_df = pd.read_csv(des_nar_narml_csv)
        video_ids = des_nar_narml_df["video_id"].tolist()

    #Instantinate ground truth DataFrame
    prediction_df = pd.DataFrame(columns=["video_id"])
    prediction_df["video_id"] = video_ids
    prediction_df["descriptive"] = ""
    prediction_df["explanatory"] = ""
    prediction_df["predictive_answer"] = ""
    prediction_df["predictive_reason"] = ""
    prediction_df["predictive"] = ""
    prediction_df["counterfactual_answer"] = ""
    prediction_df["counterfactual_reason"] = ""
    prediction_df["counterfactual"] = ""

    #Run through video ids to get ground truth answer for each video id
    for video_id in tqdm(video_ids):
        print(video_id)
        answer_file = os.path.join(prediction_dir, video_id, f"prediction_{data_mode}_{suffix}.json")
        try:
            with open(answer_file, "r") as afile:
                answers = json.load(afile)
        except:
            with open(answer_file, 'r', encoding='utf-8') as f:
                json_text = f.read()
            json_text = fix_json_errors(json_text)
            parsed_json = json.loads(json_text)
            with open(answer_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            with open(answer_file, "r") as afile:
                answers = json.load(afile)
        
        #Gathering the answer
        for answer_type in answers.keys():
            if answer_type == "descriptive" or answer_type == "explanatory":
                try:
                    answer = int(answers[answer_type]["answer"])
                    if answer not in {0,1,2,3,4}:
                        answer = 5
                except:
                    answer = 5

                prediction_df.loc[prediction_df['video_id'] == video_id, answer_type] = f"{answer}"
            else:
                try:
                    answer = int(answers[answer_type]["answer"])
                    if answer not in {0,1,2,3,4}:
                        answer = 5
                except:
                    answer = 5

                if "reason" in answers[answer_type].keys():
                    try:
                        reason = int(answers[answer_type]["reason"])
                        if reason not in {0,1,2,3,4}:
                            reason = 5
                    except:
                        reason = 5
                else:
                  reason = 5
                  
                prediction_df.loc[prediction_df['video_id'] == video_id, f"{answer_type}_answer"] = f"{answer}"
                prediction_df.loc[prediction_df['video_id'] == video_id, f"{answer_type}_reason"] = f"{reason}"
                prediction_df.loc[prediction_df['video_id'] == video_id, answer_type] = f"{answer}_{reason}"
    
    #Saving the file
    prediction_df.to_csv(f"{output_dir}/causal_vidqa_test_{data_mode}_{suffix}_prediction.csv", index=False)
    return 1

def evaluate_csv(ground_truth_csv, prediction_csv, mode:Literal["detail", "full", "sample", "all"]="all", des_nar_narml_csv:str=None):
    #Loading the metric
    accuracy_metric = load("accuracy")

    #Reading the csv file
    prediction_df = pd.read_csv(prediction_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    #Extracting ground truth
    video_ids = prediction_df["video_id"].tolist()
    ground_truth_df = ground_truth_df[ground_truth_df["video_id"].isin(video_ids)]
    ground_truth_df = ground_truth_df.sort_values(by="video_id")

    if mode == "detail" or mode == "all":
        print()
        print("Evaluation for each question type!")
        print("Accuracy score for:")
        print()

        #Descriptive evaluation
        pred_des = prediction_df["descriptive"].tolist()
        gt_des = ground_truth_df["descriptive"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_des, references=gt_des)
        score = score_dict["accuracy"]
        print(f"\t_Descriptive: {score}")

        print()

        #Explanatory evaluation
        pred_exp = prediction_df["explanatory"].tolist()
        gt_exp = ground_truth_df["explanatory"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_exp, references=gt_exp)
        score = score_dict["accuracy"]
        print(f"\t_Explanatory: {score}")

        print()

        #Predictive evaluation
        print("\t_Predictive:")

        pred_pred_ans = prediction_df["predictive_answer"].tolist()
        gt_pred_ans = ground_truth_df["predictive_answer"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_pred_ans, references=gt_pred_ans)
        score = score_dict["accuracy"]
        print(f"\t\t+ Answer: {score}")

        pred_pred_rea = prediction_df["predictive_reason"].tolist()
        gt_pred_rea = ground_truth_df["predictive_reason"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_pred_rea, references=gt_pred_rea)
        score = score_dict["accuracy"]
        print(f"\t\t+ Reason: {score}")

        pred_pred_ans_rea = prediction_df["predictive"].tolist()
        gt_pred_ans_rea = ground_truth_df["predictive"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_pred_ans_rea, references=gt_pred_ans_rea)
        score = score_dict["accuracy"]
        print(f"\t\t+ Answer - Reason: {score}")

        print()

        #Counterfactual evaluation
        print("\t_ Counterfatual:")
        pred_coun_ans = prediction_df["counterfactual_answer"].tolist()
        gt_coun_ans = ground_truth_df["counterfactual_answer"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_coun_ans, references=gt_coun_ans)
        score = score_dict["accuracy"]
        print(f"\t\t+ Answer: {score}")

        pred_coun_rea = prediction_df["counterfactual_reason"].tolist()
        gt_coun_rea = ground_truth_df["counterfactual_reason"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_coun_rea, references=gt_coun_rea)
        score = score_dict["accuracy"]
        print(f"\t\t+ Reason: {score}")

        pred_coun_ans_rea = prediction_df["counterfactual"].tolist()
        gt_coun_ans_rea = ground_truth_df["counterfactual"].tolist()
        score_dict = accuracy_metric.compute(predictions=pred_coun_ans_rea, references=gt_coun_ans_rea)
        score = score_dict["accuracy"]
        print(f"\t\t+ Answer - Reason: {score}")
    
    if mode == "full" or mode == "all":
        print()
        predictions = prediction_df["descriptive"].tolist() + prediction_df["explanatory"].tolist() + prediction_df["predictive"].tolist() + prediction_df["counterfactual"].tolist()
        references = ground_truth_df["descriptive"].tolist() + ground_truth_df["explanatory"].tolist() + ground_truth_df["predictive"].tolist() + ground_truth_df["counterfactual"].tolist()
        score_dict = accuracy_metric.compute(predictions=predictions, references=references)
        score = score_dict["accuracy"]
        print(f"Accuracy score on the whole dataset: {score}")
    
    if mode == "sample" or mode == "all":
        print()
        print("Evaluate for each video is include with the csv file!")
        des_nar_narml_df = pd.read_csv(des_nar_narml_csv)
        if "narrativeml" in prediction_csv:
            des_nar_narml_df["score_narml"] = 0
        elif "both" in prediction_csv:
            des_nar_narml_df["score_both"] = 0
        else:
            des_nar_narml_df["score"] = 0
        video_ids = des_nar_narml_df['video_id'].tolist()
        for video_id in tqdm(video_ids):
            #Extract predictions
            pred_des = prediction_df.loc[prediction_df["video_id"] == video_id, "descriptive"].values[0]
            pred_exp = prediction_df.loc[prediction_df["video_id"] == video_id, "explanatory"].values[0]
            pred_pre = prediction_df.loc[prediction_df["video_id"] == video_id, "predictive"].values[0]
            pred_cou = prediction_df.loc[prediction_df["video_id"] == video_id, "counterfactual"].values[0]
            predictions = [pred_des, pred_exp, pred_pre, pred_cou]

            #Extract ground truth
            gt_des = ground_truth_df.loc[ground_truth_df["video_id"] == video_id, "descriptive"].values[0]
            gt_exp = ground_truth_df.loc[ground_truth_df["video_id"] == video_id, "explanatory"].values[0]
            gt_pre = ground_truth_df.loc[ground_truth_df["video_id"] == video_id, "predictive"].values[0]
            gt_cou = ground_truth_df.loc[ground_truth_df["video_id"] == video_id, "counterfactual"].values[0]
            references = [gt_des, gt_exp, gt_pre, gt_cou]

            #Assign score
            score_dict = accuracy_metric.compute(predictions=predictions, references=references)
            score = score_dict["accuracy"]
            if "narrativeml" in prediction_csv:
                des_nar_narml_df.loc[des_nar_narml_df["video_id"] == video_id, "score_narml"] = score
            elif "both" in prediction_csv:
                des_nar_narml_df.loc[des_nar_narml_df["video_id"] == video_id, "score_both"] = score
            else:
                des_nar_narml_df.loc[des_nar_narml_df["video_id"] == video_id, "score"] = score
            
        #Save df
        des_nar_narml_df.to_csv(des_nar_narml_csv, index=False)
    return 1
