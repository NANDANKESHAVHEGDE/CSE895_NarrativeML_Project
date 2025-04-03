import os
import re
import json
import logging
import pandas as pd
from tqdm import tqdm
from typing import Literal
from string import Template
from vllm import LLM, SamplingParams

class Narrator(object):
    def __init__(self, model_id:str="meta-llama/Llama-3.1-8B-Instruct"):
        #Create the model
        self.llm = LLM(model=model_id, dtype='float16')

    def init_narrative_generator(self, narrative_config:dict=None):
        #Initialize SamplingParams for generating narrative
        if narrative_config == None:
            self.params = SamplingParams(temperature=0, max_tokens=1600)
        else:
            self.params = SamplingParams(
                temperature = narrative_config['temperature'],
                max_tokens = narrative_config['max_tokens']
            )

        #Initialize prompt for narrative generating
        self.general_prompt = """You are a knowledgeable and advanced artificial intelligence assistant designed to act as a narrative writer. You will receive from the user descriptions of scenes cut from a particular video, based on these descriptions, you will write a coherent narrative with accurate descriptions of entities, labels, and spatial relationships between them at different times.
        A more specific description of your task is as follows:
        \tFirst, you will receive the command "Please write a narrative based on the descriptions below about scenes from a video!" from a user.
        \tThen, you will receive a list of descriptions of scenes from a video right below the phrase "Video descriptions:".
        \tNext, you will rely on the descriptions received above to write a coherent narrative with accurate descriptions of entities, labels, and spatial relationships between them at different times. You will then output this narrative below the phrase "Your narrative output:\n".
        \tFinally, you will output the "<<<Finished!>>>" symbol to signal to the user that you have finished writing the narrative.
        Note:
        \t+ There will be some sentences close together in the description that are similar in meaning due to being created from video frames that are too close together, you are allowed to combine or remove some of these sentences to create a more coherent and meaningful narrative.
        \t+ You will have to write this narrative yourself based on descriptions of scenes cut from the video provided by the user. The user is only responsible for providing you with descriptions of scenes cut from the video!
        \t+ Aim to create a narrative with as much accurate information as possible from the description rather than using literary language!
        \t+ Keep the object name tags (if any)!
        ***********************************************************************************
        Here are three examples to help you visualize your task better!
        Example 1:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        The image depicts a serene scene of a person, likely a woman, wearing a blue tank top and sporting short hair, gazing out at a body of water. She is positioned in the foreground, facing away from the camera, with her back to the viewer. Her attire consists of a blue tank top featuring gray trim and a red stripe on the back, complemented by short hair.

        In the background, several individuals are present, including a man standing on a dock, another person partially obscured by the woman, and a boat with people on board. The scene is set against a backdrop of a bridge and a lush green area, evoking a sense of a leisurely day spent outdoors, possibly at a lake or river. The overall atmosphere is one of relaxation and tranquility, with the woman's calm demeanor and the peaceful surroundings contributing to the serene ambiance.<|eot_id|>
        The image shows a person standing on a dock, looking out at a body of water. The person is wearing a blue tank top and has short hair. They are standing on a wooden dock, with their back to the camera. In the background, there is a large metal structure that appears to be a bridge or pier. The sky is blue with some clouds.

        The person is standing on a dock, looking out at the water. They are wearing a blue tank top and have short hair. The dock is made of wood and has a railing around it. In the background, there is a large metal structure that appears to be a bridge or pier. The sky is blue with some clouds.

        The person is standing on a dock, looking out at the water. They are wearing a blue tank top and have short hair. The dock is made of wood and has a railing around it. In the background, there is a large metal structure that appears to be a bridge or pier. The sky is
        The image shows a man with short brown hair wearing a blue tank top with gray trim, standing on a dock facing away from the camera. In the background, a man in a red shirt and white shorts is standing on the dock, and another man in a red hat is sitting on a boat in the water. The man in the blue tank top appears to be looking at the man in the red shirt, who is leaning over the edge of the dock.

        In the background, there is a bridge spanning over the water, with trees and a cloudy sky visible. The overall atmosphere suggests a daytime setting, possibly during the summer or spring season, given the warm clothing and the presence of boats on the water. The man in the blue tank top seems to be engaged in a conversation or interaction with the man in the red shirt, while the man in the red hat appears to be observing the scene from a distance.<|eot_id|>
        The image depicts a scene of a man standing on a dock, with a bridge in the background and a body of water in the foreground. The man is wearing a red shirt, white shorts, and sneakers, and is holding a fishing pole in his left hand. He appears to be engaged in fishing, as evidenced by the presence of a fishing pole and the fact that he is standing near the water's edge.

        In the background, there are several other people standing on the dock, some of whom are also engaged in fishing. The bridge spans the width of the image, connecting two landmasses on either side of the water. The sky above is blue with white clouds, and the overall atmosphere suggests a sunny day.

        The image is likely taken from a boat, as the man is standing on a dock and there is a boat in the foreground. The image captures a moment of leisure and relaxation, with the man enjoying the outdoors and the activity of fishing.<|eot_id|>
        The image depicts a serene scene of a man in a red shirt and white shorts standing on a dock, grasping a long pole, with a bridge in the background and a group of people on boats in the water. The man, labeled as "person 5," is the central figure, wearing a red shirt and white shorts. He is positioned on the right side of the image, facing away from the camera, with his left arm extended, holding a long pole.

        To the left of the man, a person labeled as "person 2" is standing on the dock, wearing a light green shirt and dark pants. Behind person 2, another individual, labeled as "person 1," is also standing on the dock, wearing a light green shirt and dark pants. A fourth person, labeled as "person 3," is standing on a boat in the water, wearing a blue shirt and dark shorts. A fifth person, labeled as "person 4," is also on a
        The image shows a group of people on a boat docked at a bridge, with a man in a red shirt and white shorts standing on the dock, holding a long stick. The man is wearing a red polo shirt and white shorts, and he is looking to the left. Behind him, a man in a blue shirt and shorts is standing on the dock, holding a long stick. A man in a green shirt and shorts is standing on the dock, holding a long stick. A man in a blue shirt and shorts is standing on the dock, holding a long stick. A man in a green shirt and shorts is standing on the dock, holding a long stick. A man in a red shirt and white shorts is standing on the dock, holding a long stick.

        The background of the image shows a bridge with a metal structure and a blue sky with white clouds. The overall atmosphere of the image suggests that the people are enjoying a leisurely day on the water, possibly fishing or boating
        The image shows a scene of people on a dock, with a bridge and boats in the background. The scene appears to be a leisurely activity, with people enjoying the outdoors.

        In the foreground, a man wearing a red shirt and white shorts is standing on the dock, looking out at the water. He is holding a fishing rod in his right hand. To his left, a woman wearing a light-colored shirt and shorts is standing on the dock, also looking out at the water. She has her back to the camera. Another man, wearing a dark-colored shirt and shorts, is standing behind her, facing away from the camera.

        In the background, there is a large bridge with arches spanning across the water. Several boats are visible in the water, and trees and a blue sky with clouds can be seen in the distance. The overall atmosphere suggests a warm and sunny day, with people enjoying the outdoors and engaging in leisure activities.<|eot_id|>
        The image depicts a man standing on a dock, with a bridge in the background. The man is wearing a red shirt and white shorts, and he appears to be holding onto a rope or cable. He is standing on a wooden dock, with a body of water in the background. In the distance, there is a bridge that spans the water. The sky is blue with some clouds.

        There are also several other people in the image, some of whom are standing on the dock with the man, while others are in boats in the water. The overall atmosphere suggests a sunny day with a few clouds in the sky. The man's posture and the presence of the bridge suggest that he may be waiting for something or someone, possibly a boat or a person arriving from the other side of the bridge.<|eot_id|>

        Your narrative output:
        The scene takes place at a waterfront location where a dock extends over a body of water, with a large bridge spanning the background. The sky is blue with scattered clouds, and the atmosphere suggests a warm, sunny day.

        In the foreground, a person labeled "person_1", with short hair and wearing a blue tank top with gray trim and a red stripe on the back, stands at the edge of the dock. They are facing away from the camera, gazing out at the water. Behind them, several individuals are visible. One man, labeled "person_5", dressed in a red shirt and white shorts, is standing on the dock, seemingly engaged in fishing. He holds a fishing pole in his left hand and is positioned near the edge of the dock. To his left, another man labeled "person_2", wearing a light green shirt and dark pants, stands holding a long stick. Behind "person_2", an additional individual labeled "person_1", wearing a light green shirt and dark pants, is also standing on the dock. Further in the background, "person_3", dressed in a blue shirt and dark shorts, is positioned on a boat in the water, while "person_4" is also visible on a separate boat.

        The dock, made of wooden planks, has a railing along its sides. Several boats are moored nearby, with people onboard appearing to enjoy the outdoor setting. One person, labeled "person_4", wearing a red hat, is seated on a boat while observing the activities on the dock.

        As the scene progresses, "person_5" adjusts his grip on the fishing rod, his gaze directed toward the water. A few feet away, another person, labeled "person_6", wearing a light-colored shirt and shorts, stands near the railing, looking out at the expanse of the water. Further in the background, trees line the far shore, adding to the tranquil and scenic setting.

        The bridge in the background, a large metal structure with support beams and arches, serves as a prominent landmark in the composition. It spans across the water, connecting two landmasses. Boats continue to drift nearby, their occupants participating in leisurely activities.

        The overall scene captures a moment of relaxation and outdoor enjoyment, with individuals engaging in fishing, boating, and quiet observation. The combination of water, dock, and bridge creates a picturesque and serene setting, embodying the essence of a peaceful day by the water.

        <<<Finished!>>>

        Example 2:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        The image shows two people standing on a sidewalk, facing each other, with a car parked in the background. The person on the left is wearing a white t-shirt and blue pants, and the person on the right is wearing a white t-shirt and white pants. The car is parked on the sidewalk, partially obscured by the person on the left.

        The person on the left is marked as "person_1" and is standing on the sidewalk, facing the person on the right. The person on the right is marked as "person_2" and is standing on the sidewalk, facing the person on the left. The car is marked as "car_1" and is parked on the sidewalk, partially obscured by the person on the left.

        The scene appears to be a casual encounter between the two individuals, with the car parked in the background. The image does not provide any clear indication of their intentions or goals, but it seems to be a peaceful and ordinary moment. The image does not
        The image shows a scene with two people standing on a sidewalk, facing each other. The person on the left is labeled as "person_1" and is wearing a white t-shirt and dark pants. They have their hands behind their back and are standing with their feet shoulder-width apart. The person on the right is labeled as "person_2" and is wearing a white t-shirt and dark pants. They are also standing with their hands behind their back and their feet shoulder-width apart.

        In the background, there is a tree and a car parked on the street. The overall atmosphere suggests that the two individuals are engaged in a conversation or interaction. The image appears to be a still from a video, with the labels and text overlaid on top of the image. The image does not show any significant changes or comparisons between the frames.<|eot_id|>
        The image shows a man standing on a sidewalk, facing another man who is also standing on the sidewalk. The man on the left is wearing a white shirt and black pants, while the man on the right is wearing a white shirt and light-colored pants. The man on the left is looking at the man on the right.

        The man on the left is enclosed in a blue rectangle with the tag "person_1" and is highlighted in blue. The man on the right is enclosed in a gray rectangle with the tag "person_2" and is highlighted in gray. There is also a car in the background, which is enclosed in a blue rectangle with the tag "car_1" and is highlighted in blue.

        The image appears to be a surveillance video of two men standing on a sidewalk, with the man on the left looking at the man on the right. The image does not provide any additional context or information about the men's identities, their relationship, or the purpose of the surveillance
        The image depicts a man in a white t-shirt and jeans standing on a sidewalk, with a tree and a building in the background. The man is wearing a white t-shirt, dark-colored pants, and sneakers. He has short hair and is facing to the right, with his left arm down and his right arm bent at the elbow. The man is standing on a sidewalk, with a tree to his left and a building behind him.

        In the background, there are several houses and trees. The sky is bright and sunny, suggesting a daytime setting. The man appears to be walking or standing on the sidewalk, possibly waiting for someone or something. There is a car parked on the street behind him.

        The image is likely a still from a video, given the multiple frames provided. The man's casual attire and relaxed posture suggest that he may be taking a break or enjoying the outdoors. Overall, the image captures a moment of everyday life, with the man going about his daily activities in a peaceful
        The image shows a collage of three photos of a young boy and a mannequin dressed in white clothing. The boy is standing on a sidewalk, wearing a light blue t-shirt, jeans, and sneakers. He has dark skin and short black hair. The mannequin is also wearing a white t-shirt, pants, and sneakers. It has no facial features.

        In the first photo, the boy is standing in front of a car, with a tree and a building in the background. In the second photo, the boy is standing in front of the mannequin, which is positioned to the right of him. The mannequin is wearing a white t-shirt and pants, and has no facial features. In the third photo, the boy is standing in front of the mannequin, which is positioned to the left of him. The mannequin is wearing a white t-shirt and pants, and has no facial features.

        The background of the image is a sidewalk with trees and
        The image depicts a man and a child standing on a sidewalk. The man is positioned on the right side of the image, facing left, while the child is on the left side, facing right. The man is wearing a white t-shirt and appears to be bald, with a dark complexion. The child, who is also wearing a white t-shirt, has dark skin and is holding the hand of a person who is partially visible at the bottom of the image.

        The man is standing in front of a tree, and the child is standing in front of a building. In the background, there are houses and a car. The man and the child are both looking at each other. The image suggests that the man and the child are interacting with each other, possibly in a friendly or familial manner. The image does not provide any clear indication of the context or purpose of their interaction, but it appears to be a casual and informal setting.<|eot_id|>
        The image depicts a man standing in a driveway, wearing a white t-shirt and blue jeans, with his back to the camera. He is positioned to the right of the image, facing towards the left. In the background, a child in a white shirt is visible, and a tree, brick wall, and other buildings are also present. The man appears to be looking at a person standing in the driveway, who is also wearing a white t-shirt and blue jeans. The person is facing away from the camera, and their face is not visible. The image suggests that the man may be interacting with the person in the driveway, possibly in a social or professional context. The presence of the child and the buildings in the background adds context to the scene, indicating a residential or urban setting. Overall, the image conveys a sense of everyday life or a casual encounter between two individuals.<|eot_id|>
        The provided images are a sequence of 16 frames from a video, depicting a man walking down a sidewalk with a child following closely behind him. The man is wearing a white t-shirt and dark pants, and he is facing the camera. The child is wearing a white tank top and is walking behind the man, slightly to his right.

        In the background, there is a tree on the left side of the image, and a house can be seen on the right side. The sky is visible in the background, suggesting that the scene is taking place during the daytime. The man and child appear to be walking together, possibly on a stroll or going to a destination.

        The images show the man and child walking in a linear progression, with the man leading the way and the child following closely behind. The scene is peaceful and serene, with no apparent signs of conflict or tension. Overall, the images suggest a warm and intimate moment between the man and child, possibly a father and son or a caregiver

        Your narrative output:
        The scene takes place on a sidewalk in what appears to be a residential or urban area, with trees, houses, and parked cars visible in the background. The sky is bright, indicating a daytime setting. The video captures a sequence of interactions between individuals labeled "person_1", "person_2", and other entities.

        At the start, "person_1", a man wearing a white t-shirt and dark pants, stands on the sidewalk facing "person_2", who is also wearing a white t-shirt but with light-colored pants. The two individuals appear to be engaged in conversation, standing with their feet shoulder-width apart and hands behind their backs. Behind them, "car_1" is parked on the sidewalk, partially obscured by "person_1".

        In another moment, the scene expands to show additional background details. A tree is positioned near the sidewalk, while houses and a building provide depth to the setting. The atmosphere remains casual and peaceful, with no indications of conflict or urgency.

        The video continues, capturing different angles and interactions. "person_1" is seen in a blue rectangle, highlighted with the tag "person_1", while "person_2" is enclosed in a gray rectangle. The two maintain their positions, with "person_1" looking at "person_2", possibly engaged in conversation or awaiting a response.

        As the frames progress, another sequence features a child, dressed in a light blue t-shirt and jeans, interacting with a mannequin clad in white clothing. The mannequin, which lacks facial features, appears in multiple shots, positioned alternately to the left and right of the child. The presence of both the mannequin and the child introduces a different visual contrast, possibly suggesting an observational or playful moment.

        Later, "person_1" is shown standing in a driveway, with a child visible in the background. The man, still wearing a white t-shirt and jeans, is facing another individual dressed in similar attire. The setting includes a tree, a brick wall, and additional buildings, reinforcing the residential context.

        The final sequence captures a man walking down a sidewalk, with a child following closely behind. The man is still wearing a white t-shirt and dark pants, while the child is in a white tank top, positioned slightly to the right. The two appear to be moving in a linear progression, suggesting a stroll or journey to a destination. The peaceful atmosphere, coupled with their close positioning, implies a familial or caregiving relationship.

        Throughout the video, the interactions between "person_1", "person_2", the child, and other elements remain consistent with everyday life. The presence of the parked "car_1", trees, and residential buildings contributes to the sense of normalcy, while the varied perspectives provide a detailed view of their movements and interactions.

        <<<Finished!>>>

        Example 3:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        The image depicts a chain link fence surrounding a circular area, with a person standing in the center. The person is wearing a green shirt and pants, and has their hands raised to their face. They appear to be looking down at something on the ground.

        The fence is made of metal and has a gate that is slightly ajar. In the background, there are trees and other objects that are not clearly visible. The overall atmosphere suggests a daytime setting, possibly in a park or other outdoor recreational area.

        The person's actions and the presence of the fence suggest that they may be trying to access or interact with something inside the fenced area. The image does not provide enough information to determine what the person is looking at or trying to do, but it appears to be some kind of object or activity that is not visible from this angle.<|eot_id|>
        The image depicts a person standing in a fenced-in area, likely a batting cage or practice facility, with a chair and a fence visible in the background. The person is wearing a black shirt and pants, with their head covered by the object detection box labeled "person_1". They are standing in the center of the image, facing away from the camera.

        To the left of the person, there is a purple object detection box labeled "chair_1", which appears to be a chair. The chair is positioned behind the person and to their left. In the background, there are trees and a fence surrounding the area, suggesting that the scene is set in a park or outdoor recreational facility.

        The overall atmosphere of the image suggests that the person is engaged in some form of physical activity, possibly practicing their batting skills. The presence of the chair and the fenced-in area implies that the person is in a controlled environment, possibly for safety reasons. Overall, the image conveys a sense of focus and
        The image depicts a baseball player standing in front of a fence, with a chair situated behind him. The player is wearing a green shirt and pants, and is likely preparing to bat or practice his swing. The chair appears to be a standard folding chair, with a metal frame and a flat seat and backrest.

        In the background, there are several trees and a fence, suggesting that the scene is set in a park or outdoor recreational area. The overall atmosphere of the image is one of casual, relaxed activity, with the player focused on his practice and the chair providing a convenient place to sit or lean on.<|eot_id|>
        The image depicts a man standing in a circular enclosure, with a chain-link fence surrounding it. The man is wearing a dark-colored shirt and pants, and he appears to be standing on a concrete surface. The fence is made of metal and has a circular shape, with a gate in the background.

        In the background, there are trees and a playground, suggesting that the scene is set in a park or outdoor recreational area. The overall atmosphere of the image is one of a sunny day, with the man standing alone in the enclosure.

        The man is standing in the center of the enclosure, with his arms at his sides. He appears to be looking down at the ground, possibly examining something or lost in thought. The chain-link fence is made up of interlocking metal links, with a circular shape that forms a complete circle. The gate in the background is made of the same metal and has a similar design to the fence.

        Overall, the image suggests that the man is in a secure or restricted
        The image depicts a man standing in a circular pit, surrounded by a chain-link fence, with a chair situated behind him. The man is wearing a light-colored shirt and pants, and his head is not visible. He appears to be standing in a batting cage, with a yellow ball lying on the ground in front of him.

        The chair, positioned behind the man, has a metal frame and a backrest. The man is facing away from the camera, and his body is slightly turned to the right. The background of the image shows a park-like setting with trees and grass, suggesting that the scene is taking place outdoors.

        The overall atmosphere of the image is one of a casual, relaxed setting, possibly a recreational area or a sports field. The presence of the batting cage and the man's attire suggest that the image may be related to baseball or another sport.<|eot_id|>
        The image depicts a chain-link fence surrounding a playground area, with a person sitting on a chair. The person is wearing dark pants and a dark shirt, and is sitting in the center of the image. The chair is to the right of the person, and it appears to be a standard metal chair.

        The person is sitting on the chair, facing the camera. The person's face is not visible, but their body language suggests that they are relaxed and comfortable. The chair is positioned in front of a chain-link fence, which surrounds the playground area. In the background, there are trees and other playground equipment, such as a slide and a swing set.

        The overall atmosphere of the image is one of a sunny day at a playground, with the person enjoying their time sitting on the chair. The image suggests a carefree and peaceful scene, with the person taking a break from playing on the playground equipment to sit and relax.<|eot_id|>
        The image depicts a person standing in a circular, fenced-in area, likely a sports field, with a chain-link fence surrounding it. The person, marked by a purple rectangle, is dressed in a light-colored shirt and pants and is holding a long object, possibly a bat or a stick, in their hands, highlighted by a green rectangle. The person appears to be in motion, as indicated by the blurred image and the motion lines.

        The background of the image shows a park with trees and grass, suggesting that the scene is set in a public outdoor setting. The overall atmosphere of the image is one of activity and movement, with the person's dynamic pose and the blurred background creating a sense of energy and action.<|eot_id|>
        The image shows a person in a cage, possibly a batting cage, with a chair behind them. The person is wearing a dark-colored shirt and pants, and their face is not visible. They are standing in the center of the cage, facing towards the right side of the image. The chair is positioned behind the person, also facing towards the right side of the image. The cage is made of metal and has a chain-link fence surrounding it. The background of the image is blurry, but it appears to be a park or outdoor area with trees and grass. The overall atmosphere suggests that the person is engaged in some sort of physical activity or training, possibly baseball or softball.<|eot_id|>

        Your narrative output:
        The scene takes place in an outdoor recreational area, enclosed by a circular chain-link fence. A person_1, wearing a green shirt and pants, stands at the center of the fenced-in area with their hands raised to their face, looking down at something on the ground. The metal fence has a gate that is slightly ajar, suggesting possible access to or interaction with something inside the enclosed space. Trees and other objects are visible in the background, reinforcing the setting as a park or sports field.

        As the scene progresses, person_1 is now wearing a black shirt and pants, standing in what appears to be a batting cage or practice facility. Their head is covered by an object detection box labeled "person_1". To their left, there is a purple object detection box labeled "chair_1", identifying a chair positioned behind them. The structured environment and the presence of a chair suggest that the area is intended for training or sports practice, likely baseball.

        Further in the sequence, person_1 is depicted again, standing in front of the fence, with chair_1 still situated behind them. They appear focused, possibly preparing to bat or practice their swing. The chair, identified as a standard folding chair with a metal frame and backrest, is placed for convenience, allowing for moments of rest between activities.

        Another angle captures person_1 in the circular enclosure, still surrounded by the chain-link fence. This time, they are seen on a concrete surface, reinforcing the idea that this is a controlled sports facility. A yellow ball is lying on the ground in front of them, adding to the implication of a batting practice setup. The background reveals a playground with trees and grass, emphasizing the park-like setting.

        At one point, person_1 is seen sitting on chair_1, now positioned in front of the chain-link fence surrounding a playground area. Wearing dark pants and a dark shirt, they sit facing the camera, their face not visible. The relaxed posture suggests they are taking a break. Behind them, additional playground equipment, such as a slide and a swing set, can be seen, reinforcing a casual and recreational environment.

        In a later moment, person_1 is standing in the circular fenced-in area, holding a long object, possibly a bat or a stick, which is highlighted by a green rectangle. Their blurred movement and motion lines indicate activity and energy, likely preparing to swing or engage in a physical activity.

        Towards the end of the sequence, person_1 is seen again in the batting cage, standing with chair_1 positioned behind them. They remain facing towards the right side of the image, seemingly engaged in a training exercise. The chain-link fence continues to enclose the space, maintaining a controlled setting for sports practice. The background, though blurry, still shows elements of a park with trees and grass, reinforcing the outdoor and recreational nature of the scene.

        Throughout the video, person_1 transitions between moments of focus, action, and rest, all within a structured sports practice environment. The chain-link fence, batting cage, and presence of a chair and ball highlight an emphasis on training or physical activity, while the surrounding park and playground equipment suggest a public recreational area.

        <<<Finished!>>>
        ***********************************************************************************
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        $video_descriptions

        Your narrative output:
        """

    
    def generate(self, description):
        prompt = Template(self.general_prompt)
        prompt = prompt.substitute(video_descriptions=description)
        outputs = self.llm.generate(prompt, self.params)
        output = outputs[0].outputs[0].text
        if "<<<Finished!>>>" in output:
            finish_pos = output.find("<<<Finished!>>>")
            output = output[:finish_pos]
        narrative = output.strip().strip("\n")
        return narrative
    
    def _read_file_(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
    
    def _narml_filter_(self, examples_text, spatial_flag:bool=None, temporal_flag:bool=None):
        if spatial_flag == None and temporal_flag == None:
            logging.error(f"Error in filter narrativeML")
        
        if spatial_flag == False and temporal_flag == False:
            return examples_text
        
        patterns_to_keep = []
        if spatial_flag == True:
            patterns_to_keep = patterns_to_keep + [
                r"<\?xml.*?>",  # XML header
                r"<NarrativeML.*?>", r"</NarrativeML>",  # NarrativeML opening and closing
                r"<NARRATIVE.*?>", r"</NARRATIVE>",  # NARRATIVE opening and closing
                r"<CHARACTER.*?>",  # CHARACTER tags
                r"<CONDITION.*?>",  # MENTION tags
                r"<PLACE.*?/PLACE>",  # PLACE self-closing tag
                r"<EVENT.*?/EVENT>",  # EVENT self-closing tag
                r"<SPATIALREL.*?>",  # SPATIALREL tags
            ]
        
        if temporal_flag == True:
            patterns_to_keep = patterns_to_keep + [
                r"<\?xml.*?>",  # XML header
                r"<NarrativeML.*?>", r"</NarrativeML>",  # NarrativeML opening and closing
                r"<NARRATIVE.*?>", r"</NARRATIVE>",  # NARRATIVE opening and closing
                r"<CHARACTER.*?>",  # CHARACTER tags
                r"<CONDITION.*?>",  # MENTION tags
                r"<TIME.*?/TIME>",  # TIME self-closing tag
                r"<EVENT.*?/EVENT>",  # EVENT self-closing tag
                r"<TLINK.*?>",  # TLINK tags
                r"<NEC.*?>", #NEC tags
            ]

        #Remove duplicate
        patterns_to_keep = list(set(patterns_to_keep))

        # Compile the pattern
        pattern = re.compile("|".join(patterns_to_keep))

        #Convert str to list of lines
        content = examples_text.splitlines()

        #Filter processing
        filtered_lines = []

        for line in content:
            stripped_line = line.strip()
            
            # Keep XML lines that match the patterns
            if pattern.match(stripped_line):
                filtered_lines.append(line)
            # Keep normal text (not inside XML tags)
            elif not stripped_line.startswith("<") and not stripped_line.endswith(">"):
                filtered_lines.append(line)

        # Return the filtered content as a string
        return "\n".join(filtered_lines)

    def init_narrativeml_generator(self, narrativeml_config:dict=None, dtd_file:str=None, examples_input_file:str=None,
                                   spatial_flag:bool=None, temporal_flag:bool=None):
        if narrativeml_config is None:
            self.narrativeml_params = SamplingParams(temperature = 0.7, max_tokens = 2500)
        else:
            self.narrativeml_params = SamplingParams(
                temperature = narrativeml_config['temperature'],
                max_tokens = narrativeml_config['max_tokens']
            )
        
        # Read static files
        logging.info("Reading static input files..")
        dtd = self._read_file_(dtd_file)
        ex_input = self._read_file_(examples_input_file)

        #Construct prompt
        self.examples_prompt = f"""
        Here is the DTD definition for NarrativeML:
        {dtd}
        """

        if spatial_flag == True:
            #THIS HAS TO BE USED ONLY WHEN SPATIAL_FILTER_FLAG is True (BOOLEAN FLAG)
            dcc_text = r"""
            The 15 base relations of the Double Cross calculus:
            Starting from camera position \(a\) and looking to location \(b\), one can describe qualitatively the position of a location \(c\). This  location is to the left of the oriented line given by \((a, b)\) and on a line that is perpendicular to \((a, b)\) going through \(b\). Such a configuration is described using the relation \(lp\) (left-perpendicular). Similarly, we use:
            - \(lf\) (left-forward) to describe configurations where point \(c\) is left of \((a, b)\) and “in front of” the perpendicular line going through \(b\),
            - \(lc\) (left-center) to describe configurations where \(c\) is left of \((a, b)\) and between the two perpendicular lines going through \(a\) and \(b\),
            - \(ll\) (left-line) to describe configurations where \(c\) is left of \((a, b)\) and on the perpendicular line going through \(a\), and
            - \(lb\) (left-back) to describe configurations where \(c\) is left of \((a, b)\) and “in the back” of the perpendicular line going through \(a\).
            Configurations where point \(c\) is on the oriented line given by \((a, b)\) are described using the relations \(sf\) (straight-front), \(sp\) (straight-second-point), \(sc\) (straight-center), \(sl\) (straight-same-location), and \(sb\) (straight-back). Furthermore, the relations for configurations where \(c\) is right of \((a, b)\) are named in a similar manner as the relations describing the situations when \(c\) is on the left side. Finally, since we want to describe all configurations with three points involved, we will also consider the pathological situation when \(a = b\), which gives us two additional relations: \(eq\) (when \(a = b = c\)) and \(ex\) (when \(a = b \neq c\)). The resulting set of 17 ternary relations will be denoted by \(\mathcal{D}\) in the sequel.
            """

            self.examples_prompt = self.examples_prompt + f"""
            Here is the definition of the Double Cross Calculus:
            {dcc_text}
            """

        #Filter the example text
        ex_input = self._narml_filter_(ex_input, spatial_flag=spatial_flag, temporal_flag=temporal_flag)

        examples = f"""
        ********************************************************************************
        Here are three examples of Narrative and its NarrativeML to help you visualize your task better:
        {ex_input}
        """

        self.examples_prompt = self.examples_prompt + examples
        
        self.narml_test_prompt = """
        Now, here is a new Narrative:
        Narrative:
        $new_input

        Based on DTD, example Narratives and NarrativeML outputs, generate a new NarrativeML XML output for the new Narrative. Fill out ONLY the CHARACTERS with their attributes such as appearance, clothing, etc., as well as EVENTS and their pre- and post-CONDITIONs.
        """

        if spatial_flag == True:
            self.narml_test_prompt = self.narml_test_prompt + """
            Also fill out PLACEs and SPATIALRELs (with RCC-8 as in the examples and Double Cross Calculus - DCC). No explanations needed.
            """
        
        if temporal_flag == True:
            self.narml_test_prompt = self.narml_test_prompt + """
            Also fill out TIMEs and TLINKs. No explanations needed.
            """
        
        self.narml_test_prompt = self.narml_test_prompt + """
        Please do not generate mention (<Mention> tags) or character offsets (e.g., startOffset, endOffset) and also do not include inline annotations within the output NarrativeML!
        Your output NarrativeML:
        """
        
    def _ask_LLM_(self, test_prompt, examples_prompt, params):
        
        messages=[
            {
                "role": "user",
                "content": examples_prompt
                },
            {
                "role": "user",
                "content": test_prompt
                },
                ]
        outputs = self.llm.chat(messages, sampling_params=params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        return generated_text
    
    def generate_narrativeml(self, narrative):
        test_prompt = Template(self.narml_test_prompt)
        test_prompt = test_prompt.substitute(new_input=narrative)
        narrativeml = self._ask_LLM_(test_prompt=test_prompt, examples_prompt=self.examples_prompt, params=self.narrativeml_params)
        return narrativeml
    
    def init_tagless_narrativeml_generator(self, tagless_narrativeml_config:dict=None, dtd_file:str=None, example_file:str=None,
                                           spatial_flag:bool=None, temporal_flag:bool=None):
        #Define the tagless narrativeml generator params
        if tagless_narrativeml_config is None:
            self.tagless_narrativeml_params = SamplingParams(temperature = 0.7, max_tokens = 2500)
        else:
            self.tagless_narrativeml_params = SamplingParams(
                temperature = tagless_narrativeml_config['temperature'],
                max_tokens = tagless_narrativeml_config['max_tokens']
            )
        
        
        #Read necessary file
        ##DTD file
        dtd = self._read_file_(dtd_file)
        examples = self._read_file_(example_file)

        #Filter example
        examples = self._narml_filter_(examples_text=examples, spatial_flag=spatial_flag, temporal_flag=temporal_flag)
        
        
        #Define the system prompt:
        self.tagless_narml_system_prompt = f"""You are an extremely intelligent and advanced artificial intelligence assistant, designed to act as a Narrative Spatial Reasoner.
        More specifically, you will be given a NarrativeML, along with a Document Type Definition for NarrativeML. Your task will be to summarize the NarrativeML into a coherent paragraph by answering the following questions: What are the CHARACTERs with their appearance, what are the EVENTs (with pre- and post CONDITIONs) they are involved in, what are the PLACEs if any, and what is the spatial relationship (using SPATIALREL) within those events?

        A few additional notes and instructions:
        \t+ Use NarrativeML to provide your answers.
        \t+ The  NarrativeML describes a sequence of images from a movie.
        \t+ Make sure that the paragraph narrates events in chronological order (using TLINKS).
        \t+ The paragraph should be a narrative in temporal order based on the NarrativeML.
        \t+ Output the "<<<Finished!>>>" symbol to signal to the user that you have finished generating the paragraph!
        
        Here is the Document Type Definition for NarrativeML:
        {dtd}
        ******************************************************"""

        self.tagless_narml_system_prompt = self.tagless_narml_system_prompt + f"""Here are three examples of NarrtiveML input and output paragraphs to help you better visualize your task:
        {examples}
        """

        self.tagless_narml_test_prompt = """Now, based on the Document Type Definition for NarrativeML provided above along with the NarrativeML below, summarize the NarrativeML into a coherent paragraph!
        NarrativeML:
        $narrativeml
        
        Your output paragraph:
        """

    def generate_tagless_narrativeml(self, narrativeml):
        test_prompt = Template(self.tagless_narml_test_prompt)
        test_prompt = test_prompt.substitute(narrativeml=narrativeml)
        output = self._ask_LLM_(test_prompt=test_prompt, examples_prompt=self.tagless_narml_system_prompt,
                                     params= self.tagless_narrativeml_params)
        if "<<<Finished!>>>" in output:
            finish_pos = output.find("<<<Finished!>>>")
            output = output[:finish_pos]
        tagless_narrativeml = output.strip().strip("\n")
        return tagless_narrativeml
    
    def init_qa_generator(self, qa_config:dict=None, dtd_file:str=None, examples_input_file:str=None,
                          input_mode:Literal["narrative", "narrativeml", "both"]="both", narml_type:str=None, use_subtypes:bool=None,
                          subtypes_examples_dir:str=None):
        #Initialize Question Answer Sampling Params for the generator
        if qa_config == None:
            self.qa_params = SamplingParams(temperature = 0.7, max_tokens = 2500)
        else:
            self.qa_params = SamplingParams(
                temperature = qa_config['temperature'],
                max_tokens = qa_config['max_tokens']
            )

        #Define if using subtypes classified
        if use_subtypes == True:
            subtypes_examples = self._read_file_(subtypes_examples_dir)
            #Prompt to define LLM task
            subtypes_sys_prompt = """You are an extremely intelligent and advanced artificial intelligence assistant designed to help users classify subtypes for four types of questions: descriptive, explanatory, predictive and counterfactual.
            You will receive a list of questions belonging to the four main types: descriptive, explanatory, predictive and counterfactual in JSON format along with a list of subtypes for each main type with a short definition for each subtype. Your task is to use this list to classify subtypes for each question belonging to each main type by adding a new key "subtype" to the JSON for each question type and recording the exact name of the subtype that you think is the best and most suitable for that question!

            Note:
            \t- Your output must be in JSON format only, capturing a complete JSON list of questions from the user with "subtypes" of each main question type!No need to explain your choice further or generating anything outside of the JSON!
            \t- Each question belonging to each main type is only classified into one subtype corresponding to each main type!
            ********************************************************************************
            """

            #Prompt to define subtypes for each main type of question
            subtypes_definition = """Here is a list of corresponding subtypes for each main type along with a short definition for each subtype:
            **Descriptive questions subtypes**
            \t+ Location-Based Questions: These questions ask about the location of people, objects, or places within the video scene.
            \t+ Object and Appearance Description: These involve the appearance, identity, or description of an object, person, or scene attribute (e.g., color, shape, size, texture).
            \t+ Action-Based Question: These focus on describing what a person or object is doing—capturing motion, gesture, or interaction.
            \t+ Counting Question: These ask about the quantity of specific items or people visible in the scene.
            \t+ Yes/No Binary Questions: These questions expect a binary answer (yes or no) based on a clearly visible scene attribute or condition.
            \t+ Temporal Relation Questions: These involve sequencing of actions or events in time (before, after, during).
            \t+ Tool/Object Usage Questions: These questions inquire about the use or purpose of objects/tools in the scene.
            \t+ Scene or Environment Description: These relate to the overall setting, environment, weather, brightness, or ambiance of the video.

            **Explanatory questions subtypes**
            \t+ Intent-Based "Why" Questions: These questions ask about the motivation or intention behind a person's action.
            \t+ Procedure-Oriented "How" Questions: These focus on methods or mechanisms used by individuals to complete an action or achieve an effect.
            \t+ Cause-Effect Observational Questions: These ask for the visible reasons behind observed behavior or state, often referring to consequences or visible cues.
            \t+ Object Interaction & Environment Causality: These explore the reason or method for object manipulation or environmental interactions that are evident in the video.
            \t Social and Emotional Reactions: These capture emotional responses or social behaviors and seek explanations for such responses based on visible events.

            **Predictive questions subtypes**
            \t+ Immediate Next Action Prediction: These questions ask what the person will do immediately after the current visual context, with minimal or no delay.
            \t+ Future Action after a Specific Event: These involve a known prior event and require the model to reason what will happen next based on that event.
            \t+ Intent or Tendency Prediction: These questions ask what the subject intends, tends, or wants to do, requiring commonsense inference about goals or motivations.
            \t+ Effect Prediction on Others or Environment: These questions shift focus from the subject's actions to the effects or responses resulting from those actions.
            \t+ Collaborative or Group Behavior Prediction: These involve multiple agents and require understanding social dynamics or teamwork.
            \t+ Unusual or Hypothetical Behavior Prediction: These pose counterfactual or hypothetical situations, requiring deeper reasoning or moral/social understanding.

            **Counterfactual question subtypes**
            \t+ Physical Object Alteration: These questions explore scenarios involving changes or damage to inanimate objects in the scene, emphasizing physical consequences.
            \t+ Human Physical Action or Accident: These questions involve human movement or unexpected incidents such as falling, tripping, or being injured.
            \t+ Human Decision or Intention Change: These questions ask what would happen if a person chooses not to act, refuses to help, or deviates from a planned course of action.
            \t+ Tool or Instrument Malfunction: These questions focus on failures of tools, machines, or instruments due to malfunction, depletion, or loss of power.
            \t+ Weather or Environment Change: These questions simulate natural environmental changes and their effects on the scene or characters involved.
            \t+ Emotional or Mental State: These questions inquire into how emotional or physical internal states (like hunger, tiredness, or pain) affect behavior or outcomes.
            \t+ Social Interaction or Presence: These questions deal with changes in social configurations—like a person leaving, arriving, or interacting with others.
            \t+ Failure to Complete Task or Goal: These questions involve failure in executing a task or achieving a goal, often with focus on skill, performance, or timing.
            \t+ Other / Unclassified: Questions that are either too general, abstract, or ambiguous for clear categorization.
            ********************************************************************************
            """
            #Subtypes example prompt
            self.subex_prompt = subtypes_sys_prompt + subtypes_definition + subtypes_examples

            #Define a test prompt
            self.stqa_prompt = """Please base on a list of corresponding subtypes for each main type given above classify the following questions into subtypes!
            Questions in JSON format:
            $stq
            
            Your output subtypes classified questions in JSON format:
            """
        
        #Main input prompt
        if input_mode == "narrative":
            system_prompt = system_prompt = """You are an extremely intelligent, advanced, and knowledgeable artificial intelligence assistant, specifically optimized to accurately answer multiple-choice questions based on provided narrative text.
            The user will provide a detailed narrative and multiple-choice questions, including answer options, in JSON format. Carefully read the provided narrative to accurately identify key characters, events, actions, and their temporal and causal relationships. Based on this understanding, select only the single most correct answer from the provided options for each question and clearly record the corresponding option number (0, 1, 2, 3, or 4) in JSON format.
            If a question also requires selecting the most appropriate reason to support your chosen answer, clearly record the corresponding reason option number alongside your chosen answer, in JSON format.
            Do not include any additional explanations, analyses, or reasoning processes in your responses.
            """

            user_prompt = """
            After carefully reading and thoroughly analyzing the provided narrative, utilize the following analytical skills to accurately answer all multiple-choice questions:            
            \t- Scene Understanding & Event Sequencing: Clearly recognize the locations, entities, and events in the narrative and establish their correct chronological sequence.
            \t- Causal Reasoning: Accurately determine cause-and-effect relationships explicitly or implicitly indicated within the narrative.
            \t- Predictive Reasoning: Infer plausible next actions or events based on character intentions, narrative flow, and logical consistency.
            \t- Logical Consistency & Real-World Context: Evaluate alternative scenarios realistically and logically, consistent with human behavior and real-world physics.
            
            Note:
            \t+ Provide your responses exclusively as a JSON object.
            \t+ Do not include your analytical process or any further explanations.
            ********************************************************************************
            """

            examples = self._read_file_("./Related_files/narrative_only_examples.txt")

            self.examples_prompt = system_prompt + user_prompt + examples

            self.qa_prompt = """
            Now, here is a new narrative and its questions in json format!
            Narrative:
            $narrative

            Questions in json format:
            $questions

            Your answers to the questions in json format:
            """
        elif input_mode == "narrativeml":
            #Define system prompt which give the model define of narrative ml and some example
            #dtd = self._read_file_(dtd_file)
            #ex_input = self._read_file_(examples_input_file)

            #Filter narrativeml
            if narml_type == "spatial":
                spatial_flag = True
                temporal_flag = False
            elif narml_type == "temporal":
                spatial_flag = False
                temporal_flag = True
            elif narml_type == "both":
                spatial_flag = True
                temporal_flag = True
            else:
                spatial_flag = False
                temporal_flag = False

            #ex_input = self._narml_examples_filter_(ex_input)

            system_prompt = """
            You are an extremely intelligent, advanced and knowledgeable artificial intelligence assistant, specially designed to act as a support for users to answer multiple choice questions.
            More specifically, the user will provide you with a narrativeML and questions along with the answer options for each question in json format. You will rely on the narrativeML and answer the questions by selecting only one correct answer from the list of answers for each question and recording the answer number (the answer number is the number in front of each answer option, these numbers can be 0, 1, 2, 3 and 4) in json format. One thing to note, some questions also ask you to choose the most correct reason option to best match the most correct answer option you have chosen for that question, so you will also need to record the number of the most correct reason option if the question requires it. Again you just need to write down the answer number or reason number you choose in json format, absolutely no need for any further explanation.
            """
            
            user_prompt = """
            After reading and thoroughly analyzing the structured NarrativeML data provided by the user, use these skills to completely answer all the multiple-choice questions:

            - Structured Event Interpretation: Extract key entities, actions, and locations from the structured data format. Understand relationships between labeled elements and use them to interpret the scene accurately.
            - Causal & Procedural Analysis: Identify causal dependencies between events, actions, and entities by leveraging structured event links. Use logical reasoning to explain why certain actions occur.
            - Predictive Modeling & Temporal Dependencies: Analyze sequential patterns within the structured format to infer what is likely to happen next. Use learned temporal correlations from TLINKs and NECs to support the answer choice.
            - Counterfactual Evaluation & Logical Consistency: Modify structured elements in a realistic manner to assess the outcome under alternative conditions while ensuring consistency with factual world mechanics.
            
            Note:
            \t+ You just need to output the answers to the questions as a JSON file.
            \t+ There is no need to output your analysis process or thinking, and there is no need to explain anything more for your answers!
            ********************************************************************************
            """

            examples = self._read_file_("./Related_files/narrativeml_only_examples.txt")

            examples = self._narml_filter_(examples, spatial_flag=spatial_flag, temporal_flag=temporal_flag)

            self.examples_prompt = system_prompt + user_prompt + examples

            self.qa_prompt = """
            Now, here is a new narrativeML and questions in json format!
            NarrativeML:
            $narrativeml
    
            Questions in json format:
            $questions

            Your answers to the questions in json format:
            """
        
        elif input_mode == "tagless_narrativeml":
            #Define system prompt which give the model define of narrative ml and some example
            examples = self._read_file_(examples_input_file)

            system_prompt = """You are an extremely intelligent, advanced, and knowledgeable artificial intelligence assistant, specially designed to support users in answering multiple-choice questions based on tagless NarrativeML.
            The user will provide you with a detailed, structured narrative in tagless NarrativeML format and associated multiple-choice questions, including answer options, in JSON format. Carefully read the provided narrative to accurately comprehend events, entities, causal relationships, temporal sequences, and spatial configurations. Using this understanding, select only the single most correct answer from the provided options for each question and output only the number corresponding to that option (0, 1, 2, 3, or 4) in JSON format.
            Some questions will also require you to select the most correct reason option corresponding to your chosen answer. For these questions, provide both the chosen answer number and the reason number, clearly marked and recorded in JSON format.
            Do not provide any additional explanations, analysis, or reasoning processes in your responses.
            """
            
            user_prompt = """
            After carefully reading and thoroughly analyzing the provided tagless NarrativeML, apply the following analytical skills to answer the multiple-choice questions accurately:
            \t- Event and Entity Extraction: Identify and clearly understand key characters, events, actions, and their roles based solely on the text provided.
            \t- Causal Reasoning: Infer causal relationships and dependencies explicitly or implicitly described in the narrative.
            \t- Temporal Analysis: Determine the chronological sequence of events and actions from textual context and temporal indicators.
            \t- Spatial Reasoning: Interpret spatial relationships and the relative positions of characters and objects described.
            \t- Predictive Reasoning: Anticipate plausible future events or actions based on narrative context and logical progression.
            \t- Logical Consistency and Counterfactual Analysis: Evaluate alternative scenarios logically implied by the narrative, maintaining coherence and consistency with given facts.
            
            Note:
            \t+ Provide your responses exclusively as a JSON object.
            \t+ Do not include your analytical process or any further explanations.
            ********************************************************************************
            """

            self.examples_prompt = system_prompt + user_prompt + examples

            self.qa_prompt = """
            Now, here is a new tagless narrativeML and questions in json format!
            Tagless NarrativeML:
            $tagless_narrativeml
    
            Questions in json format:
            $questions

            Your answers to the questions in json format:
            """

        elif input_mode == "both":

            system_prompt = """You are an extremely intelligent, advanced and knowledgeable artificial intelligence assistant, specially designed to act as a support for users to answer multiple choice questions.
            More specifically, the user will provide you with a Narrative, a Tagless NarrativeML, and questions along with the answer options for each question in json format. You will rely on the Narrative and the Tagless NarrativeML and then answer the questions by selecting only one correct answer from the list of answers for each question and recording the answer number (the answer number is the number in front of each answer option, these numbers can be 0, 1, 2, 3 and 4) in json format. One thing to note, some questions also ask you to choose the most correct reason option to best match the most correct answer option you have chosen for that question, so you will also need to record the number of the most correct reason option if the question requires it. Again you just need to write down the answer number or reason number you choose in json format, absolutely no need for any further explanation.
            """

            user_prompt = """
            After reading and thoroughly analyzing both the Narrative and the Tagless NarrativeML data provided by the user, use these skills to completely answer all the multiple-choice questions:
            
            \t- Contextual Understanding Across Formats: Cross-reference details from both unstructured narrative text and structured yet tagless NarrativeML-style descriptions to build a complete understanding of the scene. Ensure consistency between different representations.  
            \t- Causal & Procedural Reasoning: Identify motivations, causes, and procedural steps for observed actions using both textual descriptions and structured temporal-spatial logic embedded in the Tagless NarrativeML.  
            \t- Predictive & Temporal Analysis: Integrate sequential dependencies from the narrative flow and structured event descriptions to predict the most likely next event based on temporal relations and character behaviors.  
            \t- Counterfactual Reasoning & Logical Coherence: Modify conditions within both formats to assess realistic alternative outcomes while ensuring logical consistency across representations.  

            Note:     
            \t+ You just need to output the answers to the questions as a JSON file.  
            \t+ There is no need to output your analysis process or thinking, and there is no need to explain anything more for your answers!
            ********************************************************************************
            """
            
            examples = self._read_file_("./Related_files/both_examples.txt")

            self.examples_prompt = system_prompt + user_prompt + examples

            self.qa_prompt = """
            Now, here is a new Narrative and NarrativeML along with questions!
            Narrative:
            $narrative

            Tagless NarrativeML:
            $tagless_narrativeml
    
            Questions in json format:
            $questions

            Your answers to the questions in json format:
            """
        else:
            logging.error(f"Error in input mode switch!")

        #Save the input mode
        self.qa_input_mode = input_mode

    #Function for classifying questions into subtypes
    def classify_subtypes(self, stq):
        stq_str = str(stq)
        qa_prompt = Template(self.stqa_prompt)
        qa_prompt = qa_prompt.substitute(stq=stq_str)
        stcq = self._ask_LLM_(test_prompt=qa_prompt, examples_prompt=self.subex_prompt, params=self.qa_params)
        #Try to convert stcq from str to json
        try:
            stcq_dict = json.loads(stcq)
            return stcq_dict
        except:
            logging.error("There is a problem in classifying questions into subtypes!")

    #Function for run question answering for one sample
    def generate_answer(self, narrative:str=None, narrativeml:str=None, questions:str=None, spatial_flag:bool=None,
                        temporal_flag:bool=None):
        #Process input for each type by case
        qa_prompt = Template(self.qa_prompt)
        if self.qa_input_mode == "narrative":
            if narrative == None:
                logging.error(f"Narrative cannot be None when choosing {self.qa_input_mode} input mode!")
            qa_prompt = qa_prompt.substitute(narrative=narrative, questions=questions)
        elif self.qa_input_mode == "narrativeml":
            if narrativeml == None:
                logging.error(f"NarrativeML cannot be None when choosing {self.qa_input_mode} input mode!")
            narrativeml = self._narml_filter_(narrativeml, spatial_flag=spatial_flag, temporal_flag=temporal_flag)
            qa_prompt = qa_prompt.substitute(narrativeml=narrativeml, questions=questions)
        elif self.qa_input_mode == "tagless_narrativeml":
            if narrative == None:
                logging.error(f"Narrative cannot be None when choosing {self.qa_input_mode} input mode!")
            qa_prompt = qa_prompt.substitute(tagless_narrativeml=narrative, questions=questions)
        elif self.qa_input_mode == "both":
            if narrative == None or narrativeml == None:
                logging.error(f"Narrative or NarrativeML cannot be None when choosing {self.qa_input_mode} input mode!")
            qa_prompt = qa_prompt.substitute(narrative=narrative, tagless_narrativeml=narrativeml, questions=questions)
        
        #Answering questions
        json_answer = self._ask_LLM_(qa_prompt, self.examples_prompt, self.qa_params)
        return json_answer
    
def init_text_llm(model_id):
    text_llm = Narrator(model_id=model_id)
    return text_llm

def generate_narratives(generator, generate_config:dict=None, des_nar_csv_dir:str=None,
                        checkpoint_steps:int=None):
    generator.init_narrative_generator(narrative_config=generate_config)
    des_nar_df = pd.read_csv(des_nar_csv_dir)

    if 'narrative' in des_nar_df.columns:
        video_ids = des_nar_df[des_nar_df['narrative'].isna()]['video_id'].tolist()
    else:
        des_nar_df['narrative'] = ""
        video_ids = des_nar_df["video_id"].tolist()

    if checkpoint_steps == None:
        checkpoint_steps = 400

    step = 0
    for video_id in tqdm(video_ids):
        step += 1
        description = des_nar_df[des_nar_df['video_id'] == video_id]['description'].values[0]
        narrative = generator.generate(description = description)
        des_nar_df.loc[des_nar_df['video_id'] == video_id, 'narrative'] = narrative
        if step % checkpoint_steps == 0:
            des_nar_df.to_csv(des_nar_csv_dir, index=False)

    #The last save
    des_nar_df.to_csv(des_nar_csv_dir, index=False)
    print("Finish generating narratives for all video descriptions!")
    return 1

def generate_narrativeml_files(generator, csv_file, dtd_file, examples_input_file, narrativeml_config:dict=None,
                               checkpoint_steps:int=None, spatial_flag:bool=None, temporal_flag:bool=None, column_name:str=None):
    generator.init_narrativeml_generator(narrativeml_config=narrativeml_config, dtd_file=dtd_file,
                                         examples_input_file=examples_input_file, spatial_flag=spatial_flag, temporal_flag=temporal_flag)
    #Read csv file
    des_nar_narml_df = pd.read_csv(csv_file)

    #Create or get narrativeml column
    """if 'narrativeml' in des_nar_narml_df.columns:
        video_ids = des_nar_narml_df[des_nar_narml_df['narrativeml'].isna()]['video_id'].tolist()
    else:
        des_nar_narml_df['narrativeml'] = ""
        video_ids = des_nar_narml_df["video_id"].tolist()"""

    if column_name in des_nar_narml_df.columns:
        video_ids = des_nar_narml_df[des_nar_narml_df[column_name].isna()]['video_id'].tolist()
    else:
        des_nar_narml_df[column_name] = ""
        video_ids = des_nar_narml_df["video_id"].tolist()
    
    #Initialize check point steps
    if checkpoint_steps == None:
        checkpoint_steps = 400

    step = 0
    for video_id in tqdm(video_ids):
        step += 1

        #Get the narrative based on the video_id
        narrative = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id]['narrative'].values[0]
        
        
        #Generate NarrativeML from the narrative
        try:
            narrative_ml = generator.generate_narrativeml(narrative=narrative)
            narrative_ml = generator._narml_filter_(examples_text=narrative_ml, spatial_flag=spatial_flag, temporal_flag=temporal_flag)
            des_nar_narml_df.loc[des_nar_narml_df['video_id'] == video_id, column_name] = narrative_ml
        except Exception as e:
            logging.error(f"Error in ask_LLM during Text2NML")
        
        #Save checkpoint
        if step % checkpoint_steps == 0:
            des_nar_narml_df.to_csv(csv_file, index=False)
        
    #Final save
    des_nar_narml_df.to_csv(csv_file, index=False)

    #Finish!
    print(f"Finished generating {column_name} for all narrative of all videos from the dataset!")
    return 1

def generate_tagless_narrativeml_files(generator, csv_file, dtd_file, example_file, tagless_narrativeml_config:dict=None, checkpoint_steps:int=None,
                                       narml_type:Literal["full", "spatial", "temporal", "both"]=None, narml_column_name:str=None,
                                       tagless_narml_column_name:str=None, spatial_flag:bool=None, temporal_flag:bool=None):
    #Initialize the genrator for generating tagless narrativeml
    generator.init_tagless_narrativeml_generator(tagless_narrativeml_config=tagless_narrativeml_config, dtd_file=dtd_file,
                                                 example_file=example_file, spatial_flag=spatial_flag, temporal_flag=temporal_flag)

    #Read csv file - this csv file contains narrative and narrativeml necessary for generating tagless narrativeml
    des_nar_narml_df = pd.read_csv(csv_file)

    #Define the narml type that the user wants to use
    ##Check if the user has provide the narml column name
    if narml_column_name is not None:
        narml_column = narml_column_name
    elif narml_type is not None:
        if narml_type == "full":
            narml_column = "narrativeml"
        elif narml_type == "spatial":
            narml_column = "narml_spatial"
        elif narml_type == "temporal":
            narml_column = "narml_temporal"
        else:
            narml_column = "narml_spatial_temporal"
    else:
        raise ValueError("Either the narml_type or narml_column_name hyperparameter must be other than None!")
    
    #Initialize check point steps
    if checkpoint_steps == None:
        checkpoint_steps = 400

    #Extract video id list
    if tagless_narml_column_name == None:
        tagless_narml_column_name = "tagless_narrativeml"

    if tagless_narml_column_name in des_nar_narml_df.columns:
        video_ids = des_nar_narml_df[des_nar_narml_df[tagless_narml_column_name].isna()]['video_id'].tolist()
    else:
        des_nar_narml_df[tagless_narml_column_name] = ""
        video_ids = des_nar_narml_df["video_id"].tolist()

    step = 0
    for video_id in tqdm(video_ids):
        step += 1

        #Get the narrativeml based on the video_id
        narrativeml = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id][narml_column].values[0]        
        
        #Generate tagless NarrativeML from the NarrativeML
        try:
            tagless_narrativeml = generator.generate_tagless_narrativeml(narrativeml=narrativeml)
            des_nar_narml_df.loc[des_nar_narml_df['video_id'] == video_id, tagless_narml_column_name] = tagless_narrativeml
        except Exception as e:
            logging.error(f"Error in ask_LLM during Tagless NarrativeML generation!")
        
        #Save checkpoint
        if step % checkpoint_steps == 0:
            des_nar_narml_df.to_csv(csv_file, index=False)
        
    #Final save
    des_nar_narml_df.to_csv(csv_file, index=False)

    #Finish!
    print("Finished generating tagless narrativeml for all videos from the dataset!")
    return 1

def add_answers_number(questions):
    for question_type in questions:
        for i in range(len(questions[question_type]['answer'])):
            questions[question_type]['answer'][i] = str(i) + ' . ' + questions[question_type]['answer'][i]
            if question_type == 'predictive' or question_type == "counterfactual":
                questions[question_type]['reason'][i] = str(i) + ' . ' + questions[question_type]['reason'][i]
    return questions

def generate_answer_causal_vidqa(generator, csv_file:str=None, qa_dir:str=None, output_dir:str=None, qa_config:dict=None, 
                                 input_mode:Literal["narrative", "narrativeml", "tagless_narrativeml", "both"]="both", dtd_file:str=None,
                                 examples_input_file:str=None, suffix:str=None,
                                 narml_type:Literal["full", "spatial", "temporal", "both"]="full", column_name:str=None,
                                 use_subtypes:bool=None, subtypes_examples_dir:str=None):
    #Initialize generator for answering questions
    if input_mode == "narrative" or input_mode == "tagless_narrativeml" or input_mode == "both":
        generator.init_qa_generator(qa_config=qa_config, dtd_file=dtd_file, examples_input_file=examples_input_file, input_mode=input_mode,
                                    use_subtypes=use_subtypes, subtypes_examples_dir=subtypes_examples_dir)
    else:
        generator.init_qa_generator(qa_config=qa_config, dtd_file=dtd_file, examples_input_file=examples_input_file, input_mode=input_mode,
                                    narml_type=narml_type, use_subtypes=use_subtypes, subtypes_examples_dir=subtypes_examples_dir)

    #Read the csv file and get video id
    des_nar_narml_df = pd.read_csv(csv_file)
    video_ids = des_nar_narml_df['video_id'].tolist()

    #Answer questions of each video
    test_subtypes = []
    for video_id in tqdm(video_ids):
        #Read the question file
        questions_json = os.path.join(qa_dir, video_id, "text.json")
        with open(questions_json, "r") as ifile:
            questions = json.load(ifile)
        questions = add_answers_number(questions=questions)

        #Classify subtypes for questions if set True
        if use_subtypes == True:
            temp_questions = {}
            for question_type in questions.keys():
                temp_questions[question_type] = {}
                temp_questions[question_type]['question'] = questions[question_type]['question']
            stcq = generator.classify_subtypes(stq=temp_questions)
            for question_type in stcq.keys():
                if question_type in set(list(questions.keys())):
                    questions[question_type]['subtype'] = stcq[question_type]['subtype']
                else:
                    print('Video id:', video_id)
                    print(stcq)
                    logging.error("There is a problem in classifying questions into subtypes!")
            questions['id'] = video_id #This is for debugging purpose only
            test_subtypes.append(questions)

        #Create output direction
        video_predict_dir = os.path.join(output_dir, video_id)
        if os.path.isdir(video_predict_dir) == False:
            os.mkdir(video_predict_dir)
        
        #Running answering based on input mode
        if input_mode == "narrative":
            narrative = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id]['narrative'].values[0]
            try:
                answer_json = generator.generate_answer(narrative=narrative, questions=questions)
                video_predict_json = os.path.join(video_predict_dir, f"prediction_narrative_{suffix}.json")
                with open(video_predict_json, 'w') as f:
                    f.write(answer_json)
            except Exception as e:
                logging.error(f"Error in generating asnwer!")
        elif input_mode == "narrativeml":
            #Getting narrativeML type
            if narml_type == "full":
                nar_type = "narrativeml"
                spatial_flag = False
                temporal_flag = False
            elif narml_type == "spatial":
                nar_type = "narml_spatial"
                spatial_flag = True
                temporal_flag = False
            elif narml_type == "temporal":
                nar_type = "narml_temporal"
                spatial_flag = False
                temporal_flag = True
            else:
                nar_type = "narml_spatial_temporal"
                spatial_flag = True
                temporal_flag = True

            narrativeml = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id][nar_type].values[0]
            try:
                answer_json = generator.generate_answer(narrativeml=narrativeml, questions=questions, spatial_flag=spatial_flag,
                                                        temporal_flag=temporal_flag)
                video_predict_json = os.path.join(video_predict_dir, f"prediction_narrativeml_{suffix}.json")
                with open(video_predict_json, 'w') as f:
                    f.write(answer_json)
            except Exception as e:
                logging.error(f"Error in generating asnwer!")
        elif input_mode == "tagless_narrativeml":
            tlnml = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id][column_name].values[0]
            try:
                answer_json = generator.generate_answer(narrative=tlnml, questions=questions)
                video_predict_json = os.path.join(video_predict_dir, f"prediction_tlnml_{suffix}.json")
                with open(video_predict_json, 'w') as f:
                    f.write(answer_json)
            except Exception as e:
                logging.error(f"Error in generating asnwer!")
        elif input_mode == "both":
            narrative = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id]['narrative'].values[0]
            tlnml = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id][column_name].values[0]
            try:
                answer_json = generator.generate_answer(narrative=narrative, narrativeml=tlnml, questions=questions)
                video_predict_json = os.path.join(video_predict_dir, f"prediction_both_{suffix}.json")
                with open(video_predict_json, 'w') as f:
                    f.write(answer_json)
            except Exception as e:
                logging.error(f"Error in generating asnwer!")
    return 1