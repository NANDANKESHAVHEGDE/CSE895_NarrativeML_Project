import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

class Captioner(object):
    def __init__(self, model_id:str='meta-llama/Llama-3.2-11B-Vision-Instruct'):#, num_frames:int=None):
        #Instantiate captioner
        self.model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

        #System prompt
        #self.num_frames = num_frames
        #self.image_prompt = "<|image|>" * num_frames

    """def init_description_generator(self):
        #Dedfine the image prompt
        self.image_prompt = "<|image|>" * self.num_frames"""

    def generate_description(self, images, max_words:int=200):
        #Define the user prompt
        image_prompt = "<|image|>" * len(images)


        user_prompt = image_prompt + f"""\nPlease based on this {len(images)} consecutive images/frames of the current segment of a video provided by the user, write a description for the received images/frames!
        
        Requirements:
        \t+ Use the provided tags (e.g. [person_1], [book_1], [person_2], [car_1], etc.) to call the entities enclosed by the object detection box instead of common nouns like man, woman, car, book, etc.!
        \t+ Use common nouns to call entities not enclosed by the object detection box!
        \t+ Look at all images/frames carefully before you begin writing descriptions. Do not write descriptions for each individual image/frame. Instead, after you have viewed all the images/frames, write a general description that fully describes the content of all images/frames.
        \t+ Maintain the chronological order of events as they appear in the frames!
        \t+ Focus on cause and effect relationships to be able to make predictions, judgments about goals or intentions from the actions of characters, entities!
        \t+ Pay attention to spatial arrangements and object interactions!
        \t+ Describe entities accurately based on their actual appearance instead of using altered colors caused by the object detection box!
        \t+ Write a detailed yet concise description (max {max_words} words) integrating all key details while maintaining coherence!"""
        
        chat_template = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
        ]
        #Process input and generate description
        with torch.no_grad():
            text = self.processor.apply_chat_template(chat_template, add_generation_prompt=True)
            inputs = self.processor(text=text, images=images, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=max_words)
            output_text = self.processor.decode(output[0])
        output_text = output_text.split(f"\t+ Write a detailed yet concise description (max {max_words} words) integrating all key details while maintaining coherence!<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        output_text = output_text[-1].strip()
        return output_text

def init_vision_llm(model_id:str):
    return Captioner(model_id = model_id)

def generate_descriptions(frames_folder_dir, captioner, narrative_csv:str=None, start_pos:int=0, end_pos:int=None,
                          dataset_name:str="Causal-VidQA", num_frames:int=16, num_segments:int=8, checkpoint_steps:int=40,
                          max_words:int=200):
    if narrative_csv == None:
        video_ids = os.listdir(frames_folder_dir)
        video_ids = sorted(video_ids)
        parent_path = os.path.dirname(frames_folder_dir)
        model_name = os.path.basename(captioner_id)
        description_narrative_file = os.path.join(parent_path, dataset_name + "_" + model_name + "_description_narrative.csv")
        description_narrative_df = pd.DataFrame(columns=["video_id", "description"])
        description_narrative_df['video_id'] = video_ids
        description_narrative_df['description'] = ""
        description_narrative_df.to_csv(description_narrative_file, index=False)
        final_file = description_narrative_file
    elif os.path.isfile(narrative_csv) == False:
        video_ids = os.listdir(frames_folder_dir)
        video_ids = sorted(video_ids)
        description_narrative_df = pd.DataFrame(columns=["video_id", "description"])
        description_narrative_df['video_id'] = video_ids
        description_narrative_df['description'] = ""
        description_narrative_df.to_csv(narrative_csv, index=False)
        final_file = narrative_csv
    else:
        description_narrative_df = pd.read_csv(narrative_csv)
        final_file = narrative_csv
        video_ids = description_narrative_df['video_id'].tolist()
    
    #Get start and end pos (for running a small part of video ids only)
    if end_pos == None:
        video_ids = video_ids[start_pos:]
    else:
        video_ids = video_ids[start_pos:end_pos]


    #captioner.init_description_generator()
    count = 0
    for video_id in tqdm(video_ids):
        count += 1
        frames_dir = os.path.join(frames_folder_dir, video_id)
        segment_dir_list = os.listdir(frames_dir)

        #Get real number of segments
        if len(segment_dir_list) < num_segments:
            num_segments_ = len(segment_dir_list)
        else:
            num_segments_ = num_segments

        description_list = []
        for i in tqdm(range(1, num_segments_ + 1)):
            segment_path = os.path.join(frames_dir, f"{video_id}_segment_{i}")
            image_files = os.listdir(segment_path)
            
            #Get real number of frames
            if len(image_files) < num_frames:
                num_frames_ = len(image_files)
            else:
                num_frames_ = num_frames

            image_files = sorted(image_files, key=lambda s: int(re.search(r'\d+', s).group()))
            image_files = image_files[:num_frames]
            images = []
            for frame_name in image_files:
                image = Image.open(os.path.join(segment_path, frame_name))
                images.append(image)
    
            output_description = captioner.generate_description(images, max_words)
            description_list.append(output_description)
        description = "\n".join(description_list)
        description_narrative_df.loc[description_narrative_df['video_id'] == video_id, 'description'] = description
        
        #Save at every checkpoint steps
        if count % checkpoint_steps == 0:
            print(f"Saved at {count} videos")
            print("Description:\n", description)
            description_narrative_df.to_csv(final_file, index=False)

    #The last save
    description_narrative_df.to_csv(final_file, index=False)
    print("Finish generating descriptions")
    return 1