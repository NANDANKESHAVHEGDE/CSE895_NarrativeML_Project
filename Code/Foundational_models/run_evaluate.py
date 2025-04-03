import argparse
from evaluate_helper import create_cuasal_vidqa_ground_truth_file, create_causal_vidqa_prediction_file, evaluate_csv

#Main program
def main():
    parser = argparse.ArgumentParser()

    #Define parser parameters
    parser.add_argument("--ground_truth_dir", type=str, default="./datasets/Causal-VidQA/Test/QA_test")
    parser.add_argument("--output_dir", type=str, default="./datasets/Causal-VidQA/Test")
    parser.add_argument("--prediction_dir", type=str, default="./datasets/Causal-VidQA/Test/Prediction")
    parser.add_argument("--prediction_data_mode", type=str, choices=["narrative", "narrativeml", "both"], default="narrative")
    parser.add_argument("--prediciton_suffix", type=str, default=None)
    parser.add_argument("--ground_truth_csv", type=str, default="./datasets/Causal-VidQA/Test/causal_vidqa_test_ground_truth.csv")
    parser.add_argument("--prediction_csv", type=str, default="./datasets/Causal-VidQA/Test/causal_vidqa_test_narrative_None_prediction.csv")
    parser.add_argument("--evaluate_mode", type=str, default="detail", choices=["detail", "full", "sample", "all"])
    parser.add_argument("--des_nar_narml_csv", type=str, default=None)


    #Parse arfument
    args = parser.parse_args()

    #Run create ground truth csv file
    #create_cuasal_vidqa_ground_truth_file(ground_truth_dir=args.ground_truth_dir, output_dir=args.output_dir)

    #Run create prediction csv file
    create_causal_vidqa_prediction_file(prediction_dir=args.prediction_dir, output_dir=args.output_dir, data_mode=args.prediction_data_mode, suffix=args.prediciton_suffix)
    
    #Run evaluate csv
    evaluate_csv(ground_truth_csv=args.ground_truth_csv, prediction_csv=args.prediction_csv, mode=args.evaluate_mode,                 des_nar_narml_csv=args.des_nar_narml_csv)

if __name__ == "__main__":
    main()