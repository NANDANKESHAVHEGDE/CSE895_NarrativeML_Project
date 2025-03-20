import torch
import gc
import argparse
import os
from dotenv import load_dotenv
from videos_func import videos_framing
from huggingface_hub import login
from vision_llm_func import init_vision_llm, generate_descriptions
from description_to_narrative import generate_narratives, init_text_llm, generate_narrativeml_files, generate_answer_causal_vidqa

load_dotenv()
#Constants
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

#Main program
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--num_framing_segments", type=int, default=8)
    parser.add_argument("--num_frames_per_segment", type=int, default=16)
    parser.add_argument("--videos_dir", type=str, default="./datasets/Causal-VidQA/Test/test_videos")
    parser.add_argument("--videos_frames_dir", type=str, default="./datasets/Causal-VidQA/Test/frames")
    parser.add_argument("--huggingface_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"))
    parser.add_argument("--captioner_model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--description_narrative_csv", type=str, default="narrative_csv_file.csv")
    parser.add_argument("--des_start_pos", type=int, default=0)
    parser.add_argument("--des_end_pos", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="Causal-VidQA")
    parser.add_argument("--des_num_frames", type=int, default=16)
    parser.add_argument("--des_num_segments", type=int, default=8)
    parser.add_argument("--des_max_words", type=int, default=None)
    parser.add_argument("--des_checkpoint_steps", type=int, default=40)
    parser.add_argument("--framing_short", type=int, default=4)
    parser.add_argument("--framing_long", type=int, default=7)
    parser.add_argument("--narrator_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--nar_generate_temperature", type=float, default=0)
    parser.add_argument("--nar_generate_max_tokens", type=int, default=1600)
    parser.add_argument("--nar_checkpoint_steps", type=int, default=400)
    parser.add_argument("--narml_dtd_file", type=str, default="./Related_files/NarrativeML-latest.dtd.xml")
    parser.add_argument("--narml_examples_input_file", type=str, default="./Related_files/Gold0.NML copy.xml")
    parser.add_argument("--narml_temperature", type=float, default=None)
    parser.add_argument("--narml_max_tokens", type=int, default=None)
    parser.add_argument("--narml_checkpoint_steps", type=int, default=None)
    parser.add_argument("--qa_temperature", type=float, default=None)
    parser.add_argument("--qa_max_tokens", type=int, default=None)
    parser.add_argument("--qa_dir", type=str, default="./datasets/Causal-VidQA/Test/QA_test")
    parser.add_argument("--prediction_dir", type=str, default="./datasets/Causal-VidQA/Test/Prediction")
    parser.add_argument("--qa_input_mode", type=str, default=None, choices=["narrative", "narrativeml", "both"])
    parser.add_argument("--qa_file_suffix", type=str, default=None)
    parser.add_argument("--spatial_flag", type=bool, default=None)
    parser.add_argument("--temporal_flag", type=bool, default=None)
    parser.add_argument("--narml_column_name", type=str, default="narrativeml")
    parser.add_argument("--narrativeml_type", type=str, default=None, choices=["full", "spatial", "temporal", "both"])


    args = parser.parse_args()

    #Log in huggingface
    login(token=args.huggingface_token)

    #Framing the videos into x segemts each and each segment has y frames
    videos_framing(videos_dir=args.videos_dir, frames_dir=args.videos_frames_dir, video_extension=args.video_extension, num_segments=args.num_framing_segments,
                   frames_per_segment=args.num_frames_per_segment, short=args.framing_short, long=args.framing_long)
    
    #Generating descriptions from video segments and frames
    ##Initialize vision llm
    vision_llm = init_vision_llm(model_id=args.captioner_model_id, num_frames=args.des_num_frames)
    

    generate_descriptions(frames_folder_dir=args.videos_frames_dir, captioner=vision_llm, narrative_csv=args.description_narrative_csv,
                          start_pos=args.des_start_pos, end_pos=args.des_end_pos, dataset_name=args.dataset_name,
                          num_frames=args.des_num_frames,num_segments=args.des_num_segments,checkpoint_steps=args.des_checkpoint_steps,
                          max_words=args.des_max_words)

    #Generate text only large language model for three task
    ##Narrative Generating
    ##NarrativeML Generating
    ##Answering Question
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Run garbage collection to help free memory
    gc.collect()
    text_llm = init_text_llm(model_id=args.narrator_model_id)

    #Generating narrative from description of a video
    #Note: This one run for the full dataset
    generate_config = {
        "temperature": args.nar_generate_temperature,
        "max_tokens": args.nar_generate_max_tokens
    }
    generate_narratives(generator=text_llm, generate_config=generate_config, des_nar_csv_dir=args.description_narrative_csv,
                        checkpoint_steps=args.nar_checkpoint_steps)
    
    #Generating narrativeml from narrative of a video
    #Note: This one run for the full dataset
    # if args.narml_temperature == None or args.narml_max_tokens == None:
    #     narrativeml_config = None
    # else:
    #     narrativeml_config = {}
    
    # if args.narml_temperature != None:
    #     narrativeml_config['temperature'] = args.narml_temperature

    # if args.narml_max_tokens != None:
    #     narrativeml_config['max_tokens'] = args.narml_max_tokens

    # generate_narrativeml_files(generator=text_llm, csv_file=args.description_narrative_csv, dtd_file=args.narml_dtd_file,
    #                            examples_input_file=args.narml_examples_input_file, narrativeml_config=narrativeml_config,
    #                            checkpoint_steps=args.narml_checkpoint_steps, spatial_flag=args.spatial_flag,
    #                            temporal_flag=args.temporal_flag, column_name=args.narml_column_name)
    
    #Running question asnwering
    if args.qa_temperature == None or args.qa_max_tokens == None:
        qa_config = None
    else:
        qa_config = {}
    
    if args.qa_temperature != None:
        qa_config['temperature'] = args.qa_temperature

    if args.qa_max_tokens != None:
        qa_config['max_tokens'] = args.qa_max_tokens

    ##Narrative only
    generate_answer_causal_vidqa(generator=text_llm, csv_file=args.description_narrative_csv, qa_dir=args.qa_dir,
                                 output_dir=args.prediction_dir, qa_config=qa_config, input_mode="narrative",
                                 dtd_file=args.narml_dtd_file, examples_input_file=args.narml_examples_input_file,
                                 suffix=args.qa_file_suffix)
    
    """##NarrativeML only
    generate_answer_causal_vidqa(generator=text_llm, csv_file=args.description_narrative_csv, qa_dir=args.qa_dir,
                                 output_dir=args.prediction_dir, qa_config=qa_config, input_mode="narrativeml",
                                 dtd_file=args.narml_dtd_file, examples_input_file=args.narml_examples_input_file,
                                 suffix=args.qa_file_suffix, narml_type=args.narrativeml_type)
    
    ##Both
    generate_answer_causal_vidqa(generator=text_llm, csv_file=args.description_narrative_csv, qa_dir=args.qa_dir,
                                 output_dir=args.prediction_dir, qa_config=qa_config, input_mode="both",
                                 dtd_file=args.narml_dtd_file, examples_input_file=args.narml_examples_input_file,
                                 suffix=args.qa_file_suffix, narml_type=args.narrativeml_type)"""

if __name__ == "__main__":
    main()