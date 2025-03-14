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
        ***********************************************************************************
        Here are three examples to help you visualize your work better!
        Example 1:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        The video opens with Person_1 already in the water, wearing a red shirt, standing against a backdrop of churning waves. Person_3, a man with long hair and a blue shirt, and Person_2 are navigating the rocky terrain along the shore, making their way toward the water. The waves crash violently against the rocks, foaming and churning as they approach the group. The scene feels intense, and Person_1 appears unfazed by the turbulent water around them. It becomes clear that Person_1 and Person_3 have jumped into the water to play together, adding a sense of purpose to their actions amidst the chaos.

        Person_1, still in the water in a red shirt, now becomes more prominent in the scene as the water churns around them. Person_1 stretches out their arm toward Person_3, reaching through the foaming waves. Person_3 and Person_2 continue to move closer to the shore, carefully navigating the rocks. The waves crash with force, causing splashes around Person_1. The dynamic movement of the water emphasizes the intensity of the scene, creating a dramatic effect. The chaotic environment contrasts with Person_1's steady stance, reinforcing their confidence and presence in the rushing current.

        The moment of connection begins as Person_1, standing in the water, extends their arm further toward Person_3. Person_3, in his blue shirt with long hair flowing in the water's current, hesitates while navigating the rocky shore. The turbulent water churns around them as Person_1's gesture intensifies the interaction. Person_3 reaches out and clasps Person_1's extended hand, marking their first contact. However, if Person_3 falls, he will get hurt and wet because the water flow is very fast. This risk adds to the dramatic tension of the moment, heightening the stakes of their connection.

        Person_3, still standing on the rocks, begins to step forward hesitantly, inching closer to the water as the waves crash around him. The moment of tension builds as Person_1 steadies their grip, supporting Person_3 while the water continues its dynamic motion. The water churns violently, splashing and foaming as it rushes past them. Each movement feels deliberate as Person_3 carefully navigates the slippery rocks, mindful of the forceful waves that threaten to unbalance him, adding to the overall suspense.

        Person_3, still navigating the rocky terrain, starts closing the gap between himself and Person_1. The water churns more aggressively around them, with Person_3's long hair flowing in the current. Person_1, still in the water, maintains their stance, keeping a firm hold on Person_3's hand. The dynamic tension continues to rise as the water rushes past them, with both individuals positioned within the chaotic current, preparing for the next moment of interaction as the scene grows more intense. The environment amplifies the sense of risk and adventure they share.

        Person_3 is now poised to enter the water as the dynamic intensity of the waves increases. Person_1, wearing a red shirt, keeps a firm grip on Person_3's hand as Person_3 takes their first step into the churning current. Their arms begin to bend as the distance between them closes, the water rushing past them both. Person_3's long hair flows in the current as they prepare to step fully into the turbulent water together. The sense of anticipation grows as their connection strengthens, and the moment builds toward the shared plunge into the powerful forces surrounding them.

        The moment reaches its peak as Person_3 steps into the water, fully immersed alongside Person_1. The waves continue to churn violently, and Person_3's long hair flows with the current. Person_1, still in the water in a red shirt, supports Person_3, guiding him through the turbulent environment. The water rushes past them as the dynamic intensity of the scene escalates, with Person_1 helping to stabilize the moment as Person_3 becomes fully engaged in the turbulent waters. Their shared presence in the waves cements the emotional and physical connection between them.

        Person_3 is now completely immersed in the water, and Person_1, still in the water in a red shirt, continues to help guide him. The water churns aggressively around them, foaming as it rushes past. Person_1's arm bends slightly, supporting Person_3 as he navigates the current. The dynamic interaction between them, set against the backdrop of the turbulent water, creates a dramatic and intense atmosphere, with the water continuing to swirl and foam around them. It's clear that Person_1 and Person_3 are now fully engaged in their playful interaction, embracing the thrill of the churning waves together.

        Your narrative output:
        Person_1 stands firmly in the churning water, their red shirt contrasting sharply against the foaming waves. The ocean surges around them, each wave crashing against the nearby rocks with a relentless force. Along the shoreline, Person_3, a man with long hair and a blue shirt, and Person_2 carefully navigate the rocky terrain, inching closer to the water's edge. The intensity of the scene is undeniable, with the swirling waves adding an element of chaos to the environment. Despite the turbulence, Person_1 remains unfazed, their steady presence suggesting familiarity with the powerful currents.

        As the waves continue their rhythmic assault, Person_1 extends an arm toward Person_3, signaling an invitation. The movement is deliberate, cutting through the spray of water with purpose. Meanwhile, Person_3 and Person_2 cautiously make their way over the jagged rocks, their progress slow and calculated. The roaring waves surge around them, sending bursts of water into the air, heightening the dramatic atmosphere. The contrast between Person_1's composed stance and the violent motion of the sea emphasizes their confidence amidst the relentless current.

        Person_1's outstretched arm remains firm, beckoning Person_3 closer. Hesitation flickers in Person_3's expression as he navigates the slippery terrain, the risk of a fall ever-present. The water surges against the rocks with unrelenting energy, threatening to upset his balance. Still, he reaches forward, his long hair flowing in the strong currents. Their hands finally meet, fingers clasping tightly in the turbulent water. The brief but significant connection solidifies their intent—Person_3 is about to take the plunge.

        A moment of tension lingers as Person_3 steadies himself on the uneven rocks, preparing to step forward. The waves crash around him, each surge threatening to push him off balance. Person_1 tightens their grip, offering stability against the relentless pull of the sea. The water churns violently, sending white foam swirling around them. Every step Person_3 takes is measured, the treacherous footing requiring absolute focus. The stakes of their interaction heighten, the ocean's force adding an element of danger to their playful endeavor.

        Gradually, Person_3 inches closer, their connection remaining firm. The water grows more aggressive, swirling around them as if urging the moment forward. Person_1 maintains their stance, an anchor in the chaos, their firm grasp guiding Person_3 toward the inevitable. The surrounding environment amplifies the tension, the crashing waves reinforcing the thrill of their adventure. Then, with a decisive motion, Person_3 takes his first step into the water. The current immediately envelops him, his long hair swept into the rushing tide. Their arms bend as the distance between them closes, the moment building toward complete immersion in the unpredictable sea.

        The climactic moment arrives—Person_3 plunges fully into the water, joining Person_1 in the swirling embrace of the waves. The ocean roars around them, its force both exhilarating and relentless. Person_1 remains steady, their grip still firm as they guide Person_3 through the turbulent environment. The dynamic intensity of the moment reaches its peak, the two individuals now completely immersed in the powerful forces of nature.

        As the waves continue their rhythmic dance, Person_1 offers subtle support, helping Person_3 adjust to the chaotic currents. The water rushes past them, foaming as it collides with the rocks. Their interaction, a mix of exhilaration and caution, unfolds in the midst of the ocean's raw energy. The dramatic scene lingers, their shared presence in the untamed waters cementing the moment as an unforgettable experience. Fully engaged, they embrace the thrill of the churning waves together, reveling in the dynamic intensity of the sea.

        <<<Finished!>>>

        Example 2:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        In a gymnasium with a brown floor, person_2, a woman in a purple shirt and black pants, stands defensively, possibly preparing to kick as she practices martial arts. Person_3, a man in a blue shirt and pants, is nearby, training with her. Person_4 watches intently from the background, observing their movements. The scene is dynamic, with tension in the air as person_2 focuses on her defensive stance, ready to react to any attack.

        Person_2 maintains her defensive posture, her body poised and alert, as person_3 prepares to kick. His body tenses, and his focus sharpens as he readies his movement. Person_4 watches closely, analyzing their techniques. The brown floor contrasts with the vibrant colors of their clothing, highlighting the intensity of their martial arts practice. Person_2's eyes remain fixed on person_3, anticipating his next move.

        Person_2 stays defensive, her stance firm and balanced, as person_3 lifts one leg into the air, executing a controlled kick. His movement is precise, showcasing his skill. Person_4 observes silently, engrossed in the training session. The brown floor reflects the gym's utilitarian purpose, while the purple and blue clothing of the participants adds a splash of color to the disciplined atmosphere.

        Person_2 remains in her defensive stance, her focus unwavering as person_3 holds one leg in the air, mid-kick. His balance is impeccable, and his movements are fluid. Person_4 watches intently, studying their techniques. The brown floor grounds the scene, while the vibrant attire of person_2 and person_3 adds energy to the gymnasium, emphasizing their dedication to martial arts.

        Person_2 continues to hold her defensive position, her body ready to react, as person_3 brings his legs down, completing his kick. His movements are smooth and controlled. Person_4 observes closely, absorbing the details of their training. The brown floor provides a neutral backdrop, allowing the colorful clothing of the participants to stand out, symbolizing their focus and determination.

        Person_2 relaxes slightly, still maintaining a defensive posture, as person_3 stands calmly, having finished his kick. Person_4 continues to watch, his gaze shifting between them. The brown floor anchors the scene, while the purple and blue clothing of person_2 and person_3 adds vibrancy to the gymnasium, capturing the essence of their disciplined practice.

        Person_2 remains alert, her stance ready, as person_3 stands prepared for the next move, legs down and body relaxed. Person_4 observes intently, his attention fixed on their training. The brown floor contrasts with the bright clothing of the participants, highlighting the intensity and focus of their martial arts practice in the gymnasium.

        The scene remains unchanged: person_2 stands defensively, her body poised and ready, while person_3 stands calmly, prepared for the next action. Person_4 watches closely, his focus unwavering. The brown floor provides a neutral backdrop, while the vibrant colors of their clothing add life to the scene, encapsulating the dedication and energy of their martial arts training.
                
        Your narrative output:
        Person_2, a woman dressed in a purple shirt and black pants, stands in a defensive stance in the gymnasium, her posture firm and balanced as she prepares for her training session. The brown floor beneath her feet reflects the structured environment of the space, while the intensity in her eyes reveals her focus. Nearby, Person_3, a man wearing a blue shirt and matching pants, stands in preparation for his next move. Their training session unfolds under the watchful gaze of Person_4, who observes intently from the background, analyzing their techniques.

        As the session progresses, Person_3 shifts his weight, lifting one leg into the air in a controlled kick. His balance is steady, his movement precise, demonstrating skill and discipline. Person_2 remains locked in her defensive stance, her body poised to react to any sudden attacks. Her eyes remain fixed on Person_3, reading his movement and anticipating his next action. The contrast between the brown gymnasium floor and the vibrant colors of their clothing adds energy to the disciplined atmosphere of their martial arts practice.

        Person_3 holds his leg in the air momentarily, displaying complete control over his movement. Person_2, still in her defensive position, maintains her focus, prepared to respond at any moment. In the background, Person_4 continues to watch closely, studying their techniques with quiet concentration. The scene exudes a sense of dedication, with each participant playing their role in the structured training environment.

        With a controlled motion, Person_3 lowers his leg, completing the kick with fluid precision. His movements remain smooth and deliberate, reinforcing his command over his technique. Person_2 remains poised, her stance unwavering, though she begins to relax slightly after the execution of the attack. The gymnasium floor remains a neutral backdrop, allowing the interaction between the two martial artists to stand out.

        As the training sequence comes to a pause, Person_2 retains her defensive posture but eases into a more relaxed stance. Person_3, now standing with his legs down, appears calm and prepared for the next sequence of their practice. Person_4, who has been attentively following their session, continues to observe, shifting his gaze between them, ready to analyze the next move.

        The scene maintains its structured intensity, with Person_2 still ready to react, Person_3 composed and poised for further action, and Person_4 remaining an engaged observer. The gymnasium's brown floor grounds the setting, while the vibrant contrast of the participants' clothing enhances the energy of the training session. Their dedication to martial arts is evident in every precise movement and focused expression, encapsulating the essence of discipline and skill development.
        
        <<<Finished!>>>

        Example 3:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        [person_1] is standing in a track and field stadium, wearing a black tank top and black pants with white stripes. He is on a red track, holding a discus in his right hand. The tank top allows him to move his arms freely. A white line runs along the track's edge. Behind a fence in the background, a red car is parked, and a tree line stretches into the distance. The stadium appears calm as [person_1] focuses on his task.

        [person_1] is standing on the red track, gripping the discus firmly. He is wearing a black tank top for unrestricted arm movement and black pants with white stripes. The discus is round and heavy, designed for precise throws. A white line marks the track's boundary. Behind a fence, a red car is visible, and beyond it, trees form a natural backdrop. The stadium setting is quiet, with [person_1] preparing mentally for his upcoming throw.

        [person_1] is positioned on the red track, wearing a tank top that allows free arm movement. He holds the discus, focusing on his technique. A white line outlines the track's edge. In the background, a red car is parked behind a fence, and a row of trees provides a scenic boundary. His stance is firm as he prepares for a powerful throw, making small adjustments to his grip and posture to ensure maximum efficiency in his upcoming release.

        [person_1] is gripping the discus with his right hand while wearing a tank top and black pants with white stripes. His left hand remains out of view, possibly aiding his balance. The discus, a round, weighted object, is held carefully. The red track beneath him contrasts with the white boundary line. Behind a fence, a red car is parked, and a tree line stretches into the distance. The scene is still, with [person_1] entirely focused on his throw.

        [person_1] stands on the red track, his right hand gripping the discus. He is wearing a black tank top for ease of movement and black pants with white stripes. A white line runs along the track's edge, marking boundaries. In the background, a red car is parked beyond the stadium's fence, with trees providing a natural border. His stance suggests concentration as he adjusts his grip on the discus, ensuring proper positioning before beginning his throwing motion.

        [person_1] is on the red track, gripping the discus and preparing to throw. His black tank top allows unrestricted arm motion, essential for his technique. A white line outlines the track. Behind a fence, a red car is parked, with a row of trees forming the stadium's natural backdrop. He focuses, making sure his stance and balance are set. His feet are positioned for stability, and his eyes are fixed ahead, ready for the powerful swing that will launch the discus.

        [person_1] swings his arm, beginning the discus throw motion. His black tank top provides flexibility, allowing smooth movement. His right arm extends, guiding the discus into position. The red track beneath him has a white line marking the boundary. In the background, a red car remains parked behind the fence, with trees lining the distance. His body shifts, transferring weight from one foot to another, ensuring the momentum builds properly. His muscles are tense as he reaches the final phase.

        [person_1] is about to jump, using his momentum to throw the discus. His black tank top allows his arms to move freely, aiding in the throw. His body is coiled, and his feet are positioned for maximum push-off. The red track's white boundary line remains in view. Behind a fence, a red car is parked, and trees create a scenic background. His muscles are engaged as he prepares for the explosive release, ensuring the discus travels as far as possible.
                
        Your narrative output:
        [person_1] stands on the red track of the stadium, clad in a black tank top and black pants with white stripes. His right hand grips a discus, a round, weighted object designed for precision throwing. The tank top allows for unrestricted arm movement, ensuring flexibility for his upcoming performance. A white line marks the edge of the track, providing a clear boundary for his stance. Beyond the fence enclosing the field, a red car is parked, and a line of trees stretches into the distance, forming a natural backdrop. The stadium is calm, with no distractions—only the quiet focus of [person_1] as he prepares for his throw.

        Adjusting his grip, [person_1] ensures a firm hold on the discus. His stance is carefully positioned for stability, his body poised to generate the necessary power. His left hand remains out of view, possibly aiding his balance. Every movement is deliberate, each adjustment fine-tuning his technique. The red track beneath his feet contrasts with the white boundary line, reinforcing his position within the designated throwing zone. His gaze remains locked ahead, analyzing the trajectory of his upcoming throw.

        With a calculated shift of his weight, [person_1] initiates the throwing motion. His right arm swings, guiding the discus into position. His body twists, coiling energy in preparation for the explosive release. The muscles in his arms and legs engage as he transfers force from one foot to the other. The red track remains steady beneath him, its white boundary line still visible as he executes the motion. The discus, tightly gripped in his hand, is moments away from being launched.

        As his momentum peaks, [person_1] pushes off, utilizing the full force of his body to propel the discus forward. His arms move fluidly, unrestricted by his attire, maximizing the efficiency of the throw. The stadium remains quiet, the surrounding trees and parked red car in the background serving as the only stationary elements in the otherwise dynamic scene. His body extends fully, channeling power into the final motion, ensuring the discus achieves optimal distance.

        The moment of release is imminent, his entire form engaged in executing the perfect throw. Every movement, from his stance to his grip, has built up to this precise instant. The intensity of his focus is evident, his expression unwavering. The stadium, though still and quiet, feels charged with the anticipation of the discus's imminent flight.
                
        <<<Finished!>>>

        Example 4:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        "The scene begins with a person wearing a green shirt and dark pants, standing in front of a plant with a black trunk and long, thin leaves. The person's right arm is bent at the elbow, holding a black accordion with white buttons and a red stripe on the side. The accordion has the word ""Roland"" written in white letters on the side. The person's left arm is bent at the elbow, holding a dark-colored object, possibly a book or a musical instrument. 
        The scene continues with the person in the green shirt and dark pants, now holding the accordion and adjusting the buttons with their right hand. The person's left hand is still holding the dark-colored object, possibly a book or a musical instrument. The person's body language suggests they are preparing to play the accordion, with their arms bent at the elbows and their hands positioned on the instrument. The room remains the same, with the plant and wooden floor visible in the background. 
        The person in the green shirt and dark pants continues to play the accordion, their right hand adjusting the buttons with a gentle touch. The person's left hand, still holding the dark-colored object, seems to be a book or a musical instrument, now positioned near the person's chest. The room remains unchanged, with the plant and wooden floor visible in the background. The person's body language suggests they are fully immersed in the music, their arms bent at the elbows and hands moving in a fluid motion The person  does not know how to use the bass buttons.
        The scene begins with a person wearing a green shirt and dark pants, standing in front of a plant with a black trunk and long, thin leaves. The person's right arm is bent at the elbow, holding a black accordion with white buttons and a red stripe on the side. The accordion has the word ""Roland"" written in white letters on the side. The person's left arm is bent at the elbow, holding a dark-colored object, possibly a book or a musical instrument. The person  does not know how to use the bass buttons.
        The scene begins with a person wearing a green shirt and dark pants, standing in front of a plant with a black trunk and long, thin leaves. The person's right arm is bent at the elbow, holding a black accordion with white buttons and a red stripe on the side. The accordion has the word ""Roland"" written in white letters on the side. The person's left arm is bent at the elbow, holding a dark-colored object, possibly a book or a musical instrument.The person  does not know how to use the bass buttons.
        The person in the green shirt and dark pants is now holding the accordion and adjusting the buttons with their right hand. The person's left hand, still holding the dark-colored object, seems to be a book or a musical instrument, now positioned near the person's chest. The room remains unchanged, with the plant and wooden floor visible in the background. The person's body language suggests they are fully immersed in the music, their arms bent at the elbows and hands moving in a fluid motion.The person  does not know how to use the bass buttons.
        The scene begins with a person wearing a green shirt and dark pants, standing in front of a plant with a black trunk and long, thin leaves. The person's right arm is bent at the elbow, holding a black accordion with white buttons and a red stripe on the side. The accordion has the word ""Roland"" written in white letters on the side. The person's left arm is bent at the elbow, holding a dark-colored object, possibly a book or a musical instrument. The person  does not know how to use the bass buttons.

        The scene begins with a person wearing a green shirt and dark pants, standing in front of a plant with a black trunk and long, thin leaves. The person's right arm is bent at the elbow, holding a black accordion with white buttons and a red stripe on the side. The accordion has the word ""Roland"" written in white letters on the side. The person's left arm is bent at the elbow, holding a dark-colored object, possibly a book or a musical instrument. The person  does not know how to use the bass buttons."        
        
        Your narrative output:
        The person, dressed in a green shirt and dark pants, stands in a room with a wooden floor, positioned in front of a plant with a black trunk and long, thin leaves. Their right arm is bent at the elbow, holding a black accordion with white buttons and a distinctive red stripe on the side. The word "Roland" is printed in white letters on the instrument. In their left hand, they hold a dark-colored object, possibly a book or another musical instrument.

        As the scene progresses, the person begins adjusting the buttons on the accordion with their right hand. Their left hand remains occupied with the dark-colored object, now positioned closer to their chest. Their body language suggests they are preparing to play, their arms bent and hands positioned carefully on the instrument. The background remains unchanged, with the plant still visible against the wooden floor.

        The person continues handling the accordion, their right fingers lightly pressing the buttons in an attempt to play. Their left hand maintains its hold on the unidentified object, though its purpose remains unclear. Their movements appear somewhat fluid but uncertain—an indication that they are unfamiliar with the bass buttons. Despite this, they remain focused, fully engaged with the instrument.

        As they persist, their body remains in a controlled stance, their arms maintaining a consistent position. The plant and wooden floor provide a stable background, grounding the scene in an intimate, quiet setting. However, their interaction with the accordion lacks confidence, particularly in the bass section, suggesting they are still learning how to navigate its complexities.

        The scene repeats with minor variations—the person adjusting the buttons, shifting their grip slightly, and maintaining their concentration. Yet, their limited knowledge of the bass functions of the accordion remains evident. Their facial expression and body language reflect determination, but their movements indicate inexperience.

        Despite the repetition in their attempts, the person continues engaging with the accordion, occasionally pausing to reposition their hands. The setting remains unchanged, with the plant and wooden floor providing a familiar frame for their practice. The scene conveys both focus and hesitation, capturing a moment of musical exploration and the challenge of mastering a new instrument.                
        
        <<<Finished!>>>

        Example 5:
        Please write a narrative based on the descriptions below about scenes from a video!
        Video descriptions:
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.
        The image shows person_1, wearing a white outfit and a pair of comfortable trainers, standing on a treadmill in a gym. They appear to be running, as indicated by the motion blur around their legs. The treadmill has a silver frame and a black seat, and person_1's feet are off the ground, suggesting they are in mid-stride. The background reveals a well-lit gym with various exercise equipment, including treadmills and elliptical machines. Person_1 doesn't look tired, maintaining a steady pace.        
        
        Your narrative output:
        Person_1 is in a well-lit gym, standing on a treadmill with a silver frame and a black seat. Dressed in a white outfit and a pair of comfortable trainers, they appear to be running, as indicated by the motion blur surrounding their legs. Their feet are off the ground, suggesting they are mid-stride, maintaining a steady pace. The gym is equipped with various exercise machines, including treadmills and elliptical trainers, neatly arranged in the background.

        Despite the continuous motion, Person_1 does not appear fatigued. Their posture remains upright and balanced, indicating they are accustomed to this exercise. The treadmill beneath them hums steadily as they move, blending with the ambient sounds of the gym. The surrounding equipment and lighting contribute to the structured, active atmosphere of the space.

        Person_1's running form remains consistent, their feet rhythmically lifting off the treadmill's surface. The gym remains unchanged in the background, filled with other exercise machines. The repetition of movement and the lack of signs of exhaustion suggest they are engaged in a sustained workout session, fully immersed in their routine.
        
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
    
    def _narml_filter_(self, examples_text, spatial_flag:bool=False, temporal_flag:bool=False):
        if spatial_flag == False and temporal_flag == False:
            return examples_text
        
        patterns_to_keep = []
        if spatial_flag == True:
            patterns_to_keep = patterns_to_keep + [
                r"<\?xml.*?>",  # XML header
                r"<NarrativeML.*?>", r"</NarrativeML>",  # NarrativeML opening and closing
                r"<NARRATIVE.*?>", r"</NARRATIVE>",  # NARRATIVE opening and closing
                r"<CHARACTER.*?>",  # CHARACTER tags
                r"<MENTION.*?>",  # MENTION tags
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
                r"<MENTION.*?>",  # MENTION tags
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

        """#Define filter flag
        self.spatial_flag = spatial_flag
        self.temporal_flag = temporal_flag"""
        
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
            Starting from observer position \(a\) and looking to location \(b\), one can describe qualitatively the position of a location \(c\). In Figure 2, this location is to the left of the oriented line given by \((a, b)\) and on a line that is perpendicular to \((a, b)\) going through \(b\). Such a configuration is described using the relation \(lp\) (left-perpendicular). Similarly, we use:
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
        Here are some example input texts and their annotation:
        {ex_input}
        """

        self.examples_prompt = self.examples_prompt + examples
        
        self.narml_test_prompt = """
        Now, here is a new input text:
        $new_input

        Based on DTD, example inputs and XML outputs, generate a new NarrativeML XML output for the new input text. Fill out the CHARACTERs and their MENTIONs, as well as EVENTs. For each CHARACTER, record its mentionIDs and record the text offset values using textSpanStart and textSpanEnd on each MENTION. For each EVENT,  record the text offset values using textSpanStart and textSpanEnd.
        """

        if spatial_flag == True:
            self.narml_test_prompt = self.narml_test_prompt + """
            Also fill out PLACEs and SPATIALRELs (with RCC-8 as in the examples and Double Cross Calculus - DCC).  For each PLACE,  record the offset values using textSpanStart and textSpanEnd. No explanations needed.
            """
        
        if temporal_flag == True:
            self.narml_test_prompt = self.narml_test_prompt + """
            Also fill out TIMEs and TLINKs. For each TIME,  record the offset values using textSpanStart and textSpanEnd. Also fill out Narrative Event Chains(NECs). No explanations needed.
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
    
    def init_qa_generator(self, qa_config:dict=None, dtd_file:str=None, examples_input_file:str=None,
                          input_mode:Literal["narrative", "narrativeml", "both"]="both", narml_type:str=None):
        #Initialize Question Answer Sampling Params for the generator
        if qa_config == None:
            self.qa_params = SamplingParams(temperature = 0.7, max_tokens = 2500)
        else:
            self.qa_params = SamplingParams(
                temperature = qa_config['temperature'],
                max_tokens = qa_config['max_tokens']
            )
        
        #Main input prompt
        if input_mode == "narrative":
            system_prompt = """You are an extremely intelligent, advanced and knowledgeable artificial intelligence assistant, specially designed to act as a support for users to answer multiple choice questions.
            More specifically, the user will provide you with a narrative and questions along with the answer options for each question in json format. You will rely on the narrative and answer the questions by selecting only one correct answer from the list of answers for each question and recording the answer number (the answer number is the number in front of each answer option, these numbers can be 0, 1, 2, 3 and 4) in json format. One thing to note, some questions also ask you to choose the most correct reason option to best match the most correct answer option you have chosen for that question, so you will also need to record the number of the most correct reason option if the question requires it. Again you just need to write down the answer number or reason number you choose in json format, absolutely no need for any further explanation.
            """

            user_prompt = """
            After reading and thoroughly analyzing the narrative provided by the user, use these skills to completely answer all the multiple-choice questions:

            - Scene Recognition & Temporal Ordering: Identify key locations, objects, and actions occurring in the narrative. Pay attention to the sequence of events and their dependencies.
            - Causal & Procedural Reasoning: Understand why characters or objects act in a certain way and how their actions lead to observed outcomes. Identify explicit and implicit cause-and-effect relationships.
            - Future Inference & Motion Prediction: Anticipate the likely next actions based on movement trends, character intentions, and sequential dependencies within the story.
            - Conditional Analysis & Real-World Logic: Evaluate alternative scenarios by modifying key conditions in the story while maintaining logical consistency with real-world physics and human behavior.
            ********************************************************************************
            """

            examples = self._read_file_("Narrative.txt")

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

            system_prompt = f"""
            You are an extremely intelligent, advanced and knowledgeable artificial intelligence assistant, specially designed to act as a support for users to answer multiple choice questions.
            More specifically, the user will provide you with a narrativeML and questions along with the answer options for each question in json format. You will rely on the narrativeML and answer the questions by selecting only one correct answer from the list of answers for each question and recording the answer number (the answer number is the number in front of each answer option, these numbers can be 0, 1, 2, 3 and 4) in json format. One thing to note, some questions also ask you to choose the most correct reason option to best match the most correct answer option you have chosen for that question, so you will also need to record the number of the most correct reason option if the question requires it. Again you just need to write down the answer number or reason number you choose in json format, absolutely no need for any further explanation."""
            
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

            examples = self._read_file_("./narrativeml_files/narrativeml_only_examples.txt")

            examples = self._narml_filter_(examples, spatial_flag=spatial_flag, temporal_flag=temporal_flag)

            self.examples_prompt = system_prompt + user_prompt + examples

            self.qa_prompt = f"""
            Now, here is a new narrativeML and questions in json format!
            NarrativeML:
            $narrativeml
    
            Questions in json format:
            $questions

            Your answers to the questions in json format:
            """
        elif input_mode == "both":
            #dtd = self._read_file_(dtd_file)
            #ex_input = self._read_file_(examples_input_file)

            system_prompt = """You are an extremely intelligent, advanced and knowledgeable artificial intelligence assistant, specially designed to act as a support for users to answer multiple choice questions.
            More specifically, the user will provide you with a Narrative, a NarrativeML and questions along with the answer options for each question in json format. You will rely on the narrative and the narrativeML and then answer the questions by selecting only one correct answer from the list of answers for each question and recording the answer number (the answer number is the number in front of each answer option, these numbers can be 0, 1, 2, 3 and 4) in json format. One thing to note, some questions also ask you to choose the most correct reason option to best match the most correct answer option you have chosen for that question, so you will also need to record the number of the most correct reason option if the question requires it. Again you just need to write down the answer number or reason number you choose in json format, absolutely no need for any further explanation."""

            user_prompt = """
            After reading and thoroughly analyzing both the narrative and the structured NarrativeML data provided by the user, use these skills to completely answer all the multiple-choice questions:

            - Contextual Understanding Across Formats: Cross-reference details from both unstructured narrative text and structured NarrativeML data to build a complete understanding of the scene. Ensure consistency between different representations.
            - Causal & Procedural Reasoning: Identify motivations, causes, and procedural steps for observed actions using both textual descriptions and structured event relations.
            - Predictive & Temporal Analysis: Integrate sequential dependencies from the narrative flow and structured data to predict the most likely next event based on temporal relations including NarrativeML TLINKs and NECs, and character behaviors.
            - Counterfactual Reasoning & Logical Coherence: Modify conditions within both formats to assess realistic alternative outcomes while ensuring logical consistency across representations.
            
            Note:
            \t+ You just need to output the answers to the questions as a JSON file.
            \t+ There is no need to output your analysis process or thinking, and there is no need to explain anything more for your answers!
            ********************************************************************************
            """
            
            examples = self._read_file_("./narrativeml_files/both_examples.txt")

            #Filter the examples
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

            examples = self._narml_filter_(examples, spatial_flag=spatial_flag, temporal_flag=temporal_flag)

            self.examples_prompt = system_prompt + user_prompt + examples

            self.qa_prompt = f"""
            Now, here is a new Narrative and NarrativeML along with questions!
            Narrative:
            $narrative

            NarrativeML:
            $narrativeml
    
            Questions in json format:
            $questions

            Your answers to the questions in json format:
            """
        else:
            logging.error(f"Error in input mode switch!")

        #Save the input mode
        self.qa_input_mode = input_mode

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
        elif self.qa_input_mode == "both":
            if narrative == None or narrativeml == None:
                logging.error(f"Narrative or NarrativeML cannot be None when choosing {self.qa_input_mode} input mode!")
            narrativeml = self._narml_filter_(narrativeml, spatial_flag=spatial_flag, temporal_flag=temporal_flag)
            qa_prompt = qa_prompt.substitute(narrative=narrative, narrativeml=narrativeml, questions=questions)
        
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
                               checkpoint_steps:int=None, spatial_flag:bool=False, temporal_flag:bool=False, column_name:str=None):
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

def add_answers_number(questions):
    for question_type in questions:
        for i in range(len(questions[question_type]['answer'])):
            questions[question_type]['answer'][i] = str(i) + ' . ' + questions[question_type]['answer'][i]
            if question_type == 'predictive' or question_type == "counterfactual":
                questions[question_type]['reason'][i] = str(i) + ' . ' + questions[question_type]['reason'][i]
    return questions

def generate_answer_causal_vidqa(generator, csv_file:str=None, qa_dir:str=None, output_dir:str=None, qa_config:dict=None, 
                                 input_mode:Literal["narrative", "narrativeml", "both"]="both", dtd_file:str=None,
                                 examples_input_file:str=None, suffix:str=None,
                                 narml_type:Literal["full", "spatial", "temporal", "both"]="full"):
    #Initialize generator for answering questions
    if input_mode == "narrative":
        generator.init_qa_generator(qa_config=qa_config, dtd_file=dtd_file, examples_input_file=examples_input_file, input_mode=input_mode)
    else:
        generator.init_qa_generator(qa_config=qa_config, dtd_file=dtd_file, examples_input_file=examples_input_file, input_mode=input_mode,
                                    narml_type=narml_type)

    #Read the csv file and get video id
    des_nar_narml_df = pd.read_csv(csv_file)
    video_ids = des_nar_narml_df['video_id'].tolist()

    #Answer questions of each video
    for video_id in tqdm(video_ids):
        #Read the question file
        questions_json = os.path.join(qa_dir, video_id, "text.json")
        with open(questions_json, "r") as ifile:
            questions = json.load(ifile)
        questions = add_answers_number(questions=questions)

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
        elif input_mode == "both":
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

            narrative = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id]['narrative'].values[0]
            narrativeml = des_nar_narml_df[des_nar_narml_df['video_id'] == video_id][nar_type].values[0]
            try:
                answer_json = generator.generate_answer(narrative=narrative, narrativeml=narrativeml, questions=questions, spatial_flag=spatial_flag,
                                                        temporal_flag=temporal_flag)
                video_predict_json = os.path.join(video_predict_dir, f"prediction_both_{suffix}.json")
                with open(video_predict_json, 'w') as f:
                    f.write(answer_json)
            except Exception as e:
                logging.error(f"Error in generating asnwer!")
    return 1