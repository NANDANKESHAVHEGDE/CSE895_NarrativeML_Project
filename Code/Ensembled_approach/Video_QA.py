import pandas as pd
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import re
import ast 
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MODEL_NAMES = [
    "mistralai/Mistral-7B-Instruct-v0.2",  # Good balance of size and performance
    "meta-llama/Llama-2-13b-chat-hf",      # Strong reasoning capabilities
    "Qwen/Qwen1.5-7B-Chat"                 # Good at following instructions
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Question subtypes and their characteristics
QUESTION_SUBTYPES = {
    "descriptive": [
        "appearance", 
        "location", 
        "identification",
        "counting", 
        "color", 
        "shape", 
        "size"
    ],
    "explanatory": [
        "cause_effect", 
        "motivation", 
        "reasoning", 
        "process_explanation", 
        "justification"
    ],
    "predictive": [
        "immediate_outcome", 
        "long_term_consequence", 
        "next_action", 
        "physical_reaction", 
        "emotional_response"
    ],
    "counterfactual": [
        "alternative_outcome", 
        "hypothetical_change", 
        "preventative_measure", 
        "impact_assessment", 
        "causal_variation"
    ]
}

class VideoQASystem:
    def __init__(self, model_names: List[str] = MODEL_NAMES, device: str = DEVICE):
        self.models = []
        self.tokenizers = []
        self.device = device
        self.model_names = model_names
        self.model_performance = {}  # Track model performance for adaptive weighting
        
        logger.info(f"Loading {len(model_names)} models...")
        for model_name in model_names:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Load in 8-bit precision for efficiency
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                # Initialize performance tracking
                self.model_performance[model_name] = {
                    "total": 0,
                    "correct": 0,
                    "weight": 1.0  # Start with equal weights
                }
                logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                
        if not self.models:
            raise ValueError("No models could be loaded. Please check your internet connection and model names.")
    
    def _detect_question_subtype(self, question: str, question_type: str) -> str:
        """Identify the subtype of question to better tailor the response"""
        
        # Use prompt engineering to detect question subtype
        prompt = f"""Analyze this question and determine its specific subtype.
Question: "{question}"
Main question type: {question_type}

For {question_type} questions, possible subtypes include:
{', '.join(QUESTION_SUBTYPES[question_type])}

Respond only with the most appropriate subtype from the list above."""
        
        # Use the first model for subtype detection
        model = self.models[0]
        tokenizer = self.tokenizers[0]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,  # Low temperature for more deterministic response
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip().lower()
        
        # Match response to closest subtype
        for subtype in QUESTION_SUBTYPES[question_type]:
            if subtype.lower() in response:
                return subtype
        
        # Try matching using keywords in the question
        keywords = {
            "appearance": ["look", "appear", "see", "visual", "shape", "color"],
            "location": ["where", "place", "location", "position"],
            "identification": ["what", "which", "identify", "recognize", "who"],
            "counting": ["how many", "count", "number of"],
            "color": ["color", "hue", "shade"],
            "shape": ["shape", "form", "structure", "figure"],
            "size": ["size", "how big", "how small", "dimension"],
            "cause_effect": ["why", "because", "cause", "effect", "result"],
            "motivation": ["why", "motivation", "reason for", "purpose"],
            "reasoning": ["explain", "reason", "logic", "rationale"],
            "process_explanation": ["how", "process", "procedure", "method"],
            "justification": ["justify", "support", "defend", "warrant"],
            "immediate_outcome": ["next", "soon", "immediate", "shortly", "about to"],
            "long_term_consequence": ["eventually", "ultimately", "long term", "in the future"],
            "next_action": ["do next", "next step", "action", "move"],
            "physical_reaction": ["physically", "physically react", "body", "motion"],
            "emotional_response": ["feel", "emotion", "react", "response"],
            "alternative_outcome": ["instead", "alternative", "different outcome"],
            "hypothetical_change": ["if", "would", "could", "might", "hypothetical"],
            "preventative_measure": ["prevent", "avoid", "stop", "precaution"],
            "impact_assessment": ["impact", "effect", "influence", "consequence"],
            "causal_variation": ["change", "alter", "modify", "vary"]
        }
        
        # Count keyword matches for each subtype
        subtype_scores = {subtype: 0 for subtype in QUESTION_SUBTYPES[question_type]}
        question_lower = question.lower()
        
        for subtype in QUESTION_SUBTYPES[question_type]:
            for keyword in keywords.get(subtype, []):
                if keyword in question_lower:
                    subtype_scores[subtype] += 1
        
        # Return the subtype with the highest score, or default to first subtype
        if max(subtype_scores.values(), default=0) > 0:
            return max(subtype_scores.items(), key=lambda x: x[1])[0]
        
        # Default to first subtype if no clear match
        return QUESTION_SUBTYPES[question_type][0]
    
    def _format_prompt(self, narrative: str, question: str, question_type: str, 
                      question_subtype: str, options: List[str] = None) -> str:
        """Format an enhanced prompt based on question type and subtype"""
        
        # Base examples for each question type
        type_examples = {
            "descriptive": {
                "appearance": """
Example Narrative: A woman with long brown hair is sitting on a bench in a park. She's wearing a red jacket and holding a book.
Example Question: What is the woman wearing?
Chain of Thought: I need to identify what the woman is wearing. The narrative specifically mentions she's wearing a red jacket.
Answer: The woman is wearing a red jacket.
""",
                "shape": """
Example Narrative: The child is playing with a blue toy that has four wheels, a rectangular body, and a round dome on top.
Example Question: What is the shape of the toy's body?
Chain of Thought: I need to identify the shape of the toy's body. The narrative mentions the toy has a rectangular body.
Answer: The toy's body is rectangular.
""",
                "color": """
Example Narrative: On the kitchen counter are three fruit bowls: one with green apples, one with yellow bananas, and one with red strawberries.
Example Question: What color are the apples?
Chain of Thought: I need to identify the color of the apples. The narrative mentions green apples.
Answer: The apples are green.
"""
            },
            "explanatory": {
                "cause_effect": """
Example Narrative: A man runs across the street when the traffic light turns red. Cars quickly brake to avoid hitting him.
Example Question: Why did the cars brake suddenly?
Chain of Thought: The narrative states the man ran across the street when the light was red, meaning the cars had a green light and weren't expecting a pedestrian. They braked suddenly to avoid hitting him.
Answer: The cars braked suddenly to avoid hitting the man who was crossing against the red light.
""",
                "motivation": """
Example Narrative: Despite the rain, the woman stood outside the concert venue for three hours before the doors opened.
Example Question: Why did the woman wait so long in the rain?
Chain of Thought: The narrative doesn't explicitly state her motivation, but it implies she was willing to endure discomfort (standing in the rain for hours) to achieve something valuable to her - likely getting a good spot at the concert.
Answer: She likely wanted to get a good position inside the venue for the concert.
"""
            },
            "predictive": {
                "next_action": """
Example Narrative: A chef has just finished chopping vegetables and has placed a pan on the stove. He's reaching for a bottle of olive oil.
Example Question: What will the chef do next?
Chain of Thought: The chef has prepared vegetables and has a pan on the stove. He's reaching for olive oil, which is typically used to start cooking in a pan.
Answer: The chef will likely pour olive oil into the pan to begin cooking the vegetables.
Reason: The typical cooking process involves adding oil to a pan before adding ingredients, and the chef has completed the preparation steps and is setting up to cook.
""",
                "immediate_outcome": """
Example Narrative: A glass of water is teetering on the edge of a table. A cat is pawing at the base of the glass.
Example Question: What will happen next?
Chain of Thought: The glass is in an unstable position (teetering on the edge), and the cat is creating additional force by pawing at it. Following the laws of physics, objects in unstable positions with additional force applied tend to fall.
Answer: The glass will probably fall off the table and spill.
Reason: The glass is already in an unstable position, and the cat's pawing will likely provide enough force to push it over the edge.
"""
            },
            "counterfactual": {
                "hypothetical_change": """
Example Narrative: A woman is walking her dog in the park when it starts to rain. She doesn't have an umbrella, so she and her dog quickly get soaked.
Example Question: What would happen if the woman had brought an umbrella?
Chain of Thought: The narrative states the woman and her dog got soaked because she didn't have an umbrella during unexpected rain. An umbrella would provide protection from rainfall.
Answer: The woman would stay relatively dry, though her dog might still get wet.
Reason: An umbrella would shield the woman from the rain, but unlikely to cover both her and her dog completely, especially if the dog is walking on a leash.
""",
                "alternative_outcome": """
Example Narrative: A chef is preparing a soufflé. He opens the oven door too quickly and the soufflé collapses.
Example Question: What would have happened if the chef had opened the oven door slowly?
Chain of Thought: The narrative indicates the soufflé collapsed because the oven door was opened too quickly. Soufflés are sensitive to sudden temperature changes.
Answer: The soufflé might have maintained its structure better.
Reason: Opening the oven door slowly would reduce the sudden temperature change and air pressure difference that causes soufflés to collapse.
"""
            }
        }
        
        # Get the example based on question type and subtype
        example = type_examples.get(question_type, {}).get(question_subtype, None)
        if not example:
            # Fall back to first subtype if specific subtype example not available
            first_subtype = next(iter(type_examples.get(question_type, {}).keys()), None)
            example = type_examples.get(question_type, {}).get(first_subtype, "")
        
        narrative_focus = self._extract_relevant_parts(narrative, question)
        
        base_prompt = f"""You are an expert at analyzing video narratives and answering questions about them.

Narrative: {narrative_focus}

Question: {question}

This is a {question_type} question, specifically a {question_subtype} type question.

{example}

Now for the current question:
Think step by step before answering. First understand what the narrative describes, then analyze the question carefully.
"""
        
        if options:
            base_prompt += f"\nPlease select the most appropriate answer from these options:\n"
            for i, option in enumerate(options):
                base_prompt += f"{i}: {option}\n"
            base_prompt += "\nProvide your answer as the option number only, followed by your detailed reasoning with reference to specific parts of the narrative."
        else:
            base_prompt += "\nProvide a concise answer based on the narrative."
            
        return base_prompt
    
    def _extract_relevant_parts(self, narrative: str, question: str) -> str:
        """Extract the most relevant parts of the narrative for the question"""
        # Split the narrative into sentences
        sentences = re.split(r'(?<=[.!?])\s+', narrative)
        
        # Clean up the question and extract key terms
        question_clean = question.lower().replace("[person_1]", "person").replace("[person_2]", "person")
        question_keywords = set(re.findall(r'\b\w{4,}\b', question_clean))
        
        # Filter out common words that aren't very discriminative
        common_words = {'what', 'when', 'where', 'why', 'how', 'would', 'could', 'will', 'does', 'did', 'person'}
        question_keywords = question_keywords - common_words
        
        # Score each sentence based on keyword matches
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.lower()
            score = 0
            for keyword in question_keywords:
                if keyword in sentence_clean:
                    score += 1
            sentence_scores.append((i, score, sentence))
        
        # Sort sentences by score (higher is better)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top scoring sentences (up to 5) and sort them by original order
        top_sentences = sentence_scores[:5]
        top_sentences.sort(key=lambda x: x[0])
        
        # If no good matches, return the full narrative
        if max([s[1] for s in sentence_scores], default=0) == 0:
            return narrative
        
        # Return the relevant parts
        return " ".join([s[2] for s in top_sentences])
    
    def _get_model_response(self, model_idx: int, prompt: str) -> str:
        """Get a response from a specific model"""
        model = self.models[model_idx]
        tokenizer = self.tokenizers[model_idx]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part (not including the prompt)
        response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        return response.strip()
    
    def _extract_answer_and_confidence(self, response: str, options: List[str] = None) -> Tuple[Any, float]:
        """Extract the answer and confidence from the model response with improved parsing"""
        if options:
            # First look for the pattern "Answer: X" or "The answer is X"
            answer_patterns = [
                r'(?:Answer|The answer is)[:\s]+(\d+)',
                r'(?:Option|Choice|Select|Selected)[:\s]+(\d+)',
                r'^(\d+)[:\s]',
                r'(\d+)\s+is the (?:correct|best|most appropriate) (?:answer|option)'
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        answer_idx = int(match.group(1))
                        if 0 <= answer_idx < len(options):
                            # Higher confidence if reasoning follows the answer
                            confidence = 0.95 if len(response) > 50 else 0.8
                            return answer_idx, confidence
                    except ValueError:
                        pass
            
            # Look for the option text verbatim
            for i, option in enumerate(options):
                if option.lower() in response.lower():
                    return i, 0.8
            
            # Count keyword matches with improved weighting for discriminative terms
            highest_confidence = 0
            best_match = 0
            
            for i, option in enumerate(options):
                # Extract meaningful keywords (words with 4+ chars)
                option_keywords = set(re.findall(r'\b\w{4,}\b', option.lower()))
                response_keywords = set(re.findall(r'\b\w{4,}\b', response.lower()))
                
                if option_keywords:
                    # Weight matches by keyword rarity
                    all_keywords = []
                    for opt in options:
                        all_keywords.extend(re.findall(r'\b\w{4,}\b', opt.lower()))
                    
                    keyword_counts = Counter(all_keywords)
                    match_score = 0
                    
                    for keyword in option_keywords:
                        if keyword in response_keywords:
                            # Rarer keywords get higher weight
                            match_score += 1 / (keyword_counts[keyword] if keyword_counts[keyword] > 0 else 1)
                    
                    normalized_score = match_score / len(option_keywords) if option_keywords else 0
                    
                    if normalized_score > highest_confidence:
                        highest_confidence = normalized_score
                        best_match = i
            
            if highest_confidence > 0.3:
                return best_match, min(highest_confidence, 0.9)  # Cap at 0.9
            
            # Try extracting number with regex as last resort
            numbers = re.findall(r'\b(\d+)\b', response)
            for num in numbers:
                try:
                    answer_idx = int(num)
                    if 0 <= answer_idx < len(options):
                        return answer_idx, 0.5
                except ValueError:
                    continue
            
            # Default to most mentioned option with low confidence
            option_mentions = [0] * len(options)
            for i, option in enumerate(options):
                # Count mentions of option text in response
                option_parts = re.findall(r'\b\w{4,}\b', option.lower())
                for part in option_parts:
                    if part in response.lower():
                        option_mentions[i] += 1
            
            if max(option_mentions) > 0:
                best_option = option_mentions.index(max(option_mentions))
                return best_option, 0.4
            
            # Truly fallback to first option with very low confidence
            return 0, 0.1
        else:
            # For free-form answers, we'll just return the text and a default confidence
            # Clean up the response to get core answer
            lines = response.strip().split('\n')
            answer_line = ""
            for line in lines:
                if line.startswith('Answer:'):
                    answer_line = line[7:].strip()
                    break
            
            # If no "Answer:" prefix found, use the first non-empty line
            if not answer_line and lines:
                for line in lines:
                    if line.strip():
                        answer_line = line.strip()
                        break
            
            final_answer = answer_line if answer_line else response
            # Confidence based on answer specificity
            confidence = 0.8 if len(final_answer.split()) > 3 else 0.6
            
            return final_answer, confidence
    
    def _get_model_weight(self, model_idx: int, question_type: str) -> float:
        """Get dynamic weight for model based on past performance"""
        model_name = self.model_names[model_idx]
        model_stats = self.model_performance.get(model_name, {"weight": 1.0})
        
        # Base weight is current stored weight
        weight = model_stats.get("weight", 1.0)
        
        # Adjust for question type if we have specific data
        type_correct = model_stats.get(f"{question_type}_correct", 0)
        type_total = model_stats.get(f"{question_type}_total", 0)
        
        if type_total > 5:  # Only adjust if we have enough data
            type_accuracy = type_correct / type_total
            # Scale by accuracy compared to baseline (0.5)
            type_factor = max(0.5, min(2.0, 2 * type_accuracy))
            weight *= type_factor
        
        return weight
    
    def _update_model_performance(self, model_idx: int, question_type: str, correct: bool) -> None:
        """Update model performance metrics"""
        if model_idx >= len(self.model_names):
            return
            
        model_name = self.model_names[model_idx]
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {"total": 0, "correct": 0, "weight": 1.0}
        
        self.model_performance[model_name]["total"] = self.model_performance[model_name].get("total", 0) + 1
        
        if correct:
            self.model_performance[model_name]["correct"] = self.model_performance[model_name].get("correct", 0) + 1
        
        # Update type-specific stats
        self.model_performance[model_name][f"{question_type}_total"] = self.model_performance[model_name].get(f"{question_type}_total", 0) + 1
        
        if correct:
            self.model_performance[model_name][f"{question_type}_correct"] = self.model_performance[model_name].get(f"{question_type}_correct", 0) + 1
        
        # Update weight based on overall accuracy
        total = self.model_performance[model_name]["total"]
        correct_count = self.model_performance[model_name]["correct"]
        
        if total >= 5:  # Only update if we have enough data
            accuracy = correct_count / total
            # Set weight proportional to accuracy, with a floor of 0.5
            self.model_performance[model_name]["weight"] = 0.5 + accuracy / 2
    
    def answer_question(self, narrative: str, question: str, question_type: str, 
                        options: List[str] = None, ground_truth: Any = None) -> Dict[str, Any]:
        """Generate an answer for a question using improved model ensemble"""
        # Detect question subtype
        question_subtype = self._detect_question_subtype(question, question_type)
        logger.info(f"Detected question subtype: {question_subtype} for question: {question}")
        
        # Format prompt with subtype information
        prompt = self._format_prompt(narrative, question, question_type, question_subtype, options)
        
        # Get responses from all models with dynamic weighting
        all_responses = []
        all_answers = []
        all_confidences = []
        all_weights = []
        
        for i in range(len(self.models)):
            try:
                response = self._get_model_response(i, prompt)
                answer, confidence = self._extract_answer_and_confidence(response, options)
                weight = self._get_model_weight(i, question_type)
                
                all_responses.append(response)
                all_answers.append(answer)
                all_confidences.append(confidence)
                all_weights.append(weight)
                
            except Exception as e:
                logger.error(f"Error getting response from model {i}: {e}")
                all_responses.append("")
                all_answers.append(0 if options else "")
                all_confidences.append(0.1)
                all_weights.append(0.1)
        
        # For multiple-choice, use weighted voting
        if options:
            # Check for unanimous agreement (strong signal)
            unique_answers = set(all_answers)
            if len(unique_answers) == 1:
                final_answer = all_answers[0]
                confidence = max(all_confidences)
            else:
                # Enhanced weighted voting
                answer_weights = {}
                for ans, conf, weight in zip(all_answers, all_confidences, all_weights):
                    # Weighted score combines confidence and model weight
                    combined_weight = conf * weight
                    answer_weights[ans] = answer_weights.get(ans, 0) + combined_weight
                
                if answer_weights:
                    final_answer = max(answer_weights.items(), key=lambda x: x[1])[0]
                    confidence = answer_weights[final_answer] / sum([c * w for c, w in zip(all_confidences, all_weights)])
                else:
                    # Fallback if something went wrong
                    final_answer = 0
                    confidence = 0.1
        else:
            # For free-form answers, use the answer from the most confident model weighted by model performance
            weighted_confidences = [c * w for c, w in zip(all_confidences, all_weights)]
            if weighted_confidences:
                max_confidence_idx = weighted_confidences.index(max(weighted_confidences))
                final_answer = all_answers[max_confidence_idx]
                confidence = all_confidences[max_confidence_idx]
            else:
                # Fallback
                final_answer = all_answers[0] if all_answers else ""
                confidence = 0.1
        
        # Check if correct (if ground truth is provided)
        correct = None
        if ground_truth is not None:
            if isinstance(ground_truth, int) and isinstance(final_answer, int):
                correct = ground_truth == final_answer
            elif isinstance(final_answer, str) and isinstance(ground_truth, str):
                # Simple string matching for free-form answers
                correct = final_answer.lower() == ground_truth.lower()
            
            # Update model performance based on correctness
            for i in range(len(all_answers)):
                model_correct = False
                if isinstance(ground_truth, int) and isinstance(all_answers[i], int):
                    model_correct = ground_truth == all_answers[i]
                elif isinstance(all_answers[i], str) and isinstance(ground_truth, str):
                    model_correct = all_answers[i].lower() == ground_truth.lower()
                
                self._update_model_performance(i, question_type, model_correct)
        
        return {
            "question": question,
            "question_type": question_type,
            "question_subtype": question_subtype,
            "final_answer": final_answer,
            "confidence": confidence,
            "model_responses": all_responses,
            "model_answers": all_answers,
            "model_confidences": all_confidences,
            "model_weights": all_weights,
            "correct": correct,
            "ground_truth": ground_truth
        }
    
    def process_csv_file(self, csv_path: str, output_path: str = "results.json") -> List[Dict[str, Any]]:
        """Process a CSV file with video narratives and questions with improved handling"""
        # Load the CSV file
        df = pd.read_csv(csv_path, encoding='latin1')
        
        results = []
        
        # Process each row
        for _, row in tqdm(df.iterrows(), total=len(df)):
            video_id = row['video_id']
            narrative = row['narrative']
            
            # Parse the question JSON
            question_data = ast.literal_eval(row['Question'])
            ground_truth_data = ast.literal_eval(row['Ground_truth_answers'])
            
            video_results = {"video_id": video_id}
            
            # Process each question type
            for q_type in ["descriptive", "explanatory", "predictive", "counterfactual"]:
                if q_type in question_data:
                    try:
                        question = question_data[q_type].get("question", "").strip()
                        options = question_data[q_type].get("answer", [])
                        
                        # Clean up options
                        options = [opt.strip() for opt in options if opt.strip()]
                        
                        # Get ground truth
                        ground_truth = None
                        if q_type in ground_truth_data:
                            ground_truth = ground_truth_data[q_type].get("answer")
                        
                        # For predictive and counterfactual questions, we also need the reason
                        if q_type in ["predictive", "counterfactual"] and "reason" in ground_truth_data.get(q_type, {}):
                            ground_truth_reason = ground_truth_data[q_type].get("reason")
                        else:
                            ground_truth_reason = None
                        
                        # Skip if no valid question or options
                        if not question or not options:
                            logger.warning(f"Skipping {q_type} question for {video_id}: Missing question or options")
                            continue
                        
                        # Answer the question
                        result = self.answer_question(
                            narrative=narrative,
                            question=question,
                            question_type=q_type,
                            options=options,
                            ground_truth=ground_truth
                        )
                        
                        # For predictive and counterfactual questions, we need to get the reason too
                        if q_type in ["predictive", "counterfactual"] and "reason" in question_data.get(q_type, {}):
                            reasons = question_data[q_type].get("reason", [])
                            reasons = [r.strip() for r in reasons if r.strip()]
                            
                            if options and result["final_answer"] < len(options) and reasons:
                                # Create a new prompt specifically for the reasoning with improved context
                                selected_answer = options[result["final_answer"]]
                                reason_prompt = f"""Based on the narrative and the question, we've determined that the answer is:
                                
    Narrative: {narrative}
    
    Question: {question}
    
    Selected Answer: {selected_answer}
    
    Now, select the most appropriate reasoning from these options:
    """
                                # Add reasons to the prompt
                                for i, reason in enumerate(reasons):
                                    reason_prompt += f"{i}: {reason}\n"
                                
                                # Get reasonings from all models
                                reason_responses = []
                                reason_answers = []
                                reason_confidences = []
                                
                                for i in range(len(self.models)):
                                    response = self._get_model_response(i, reason_prompt)
                                    answer, confidence = self._extract_answer_and_confidence(response, reasons)
                                    
                                    reason_responses.append(response)
                                    reason_answers.append(answer)
                                    reason_confidences.append(confidence)
                                
                                # Weighted voting for reason
                                reason_weights = {}
                                for ans, conf in zip(reason_answers, reason_confidences):
                                    reason_weights[ans] = reason_weights.get(ans, 0) + conf
                                
                                final_reason = max(reason_weights.items(), key=lambda x: x[1])[0]
                                reason_confidence = reason_weights[final_reason] / sum(reason_confidences)
                                
                                # Check if reason is correct
                                reason_correct = None
                                if ground_truth_reason is not None:
                                    reason_correct = ground_truth_reason == final_reason
                                
                                # Add reason results
                                result["reason"] = {
                                    "final_reason": final_reason,
                                    "confidence": reason_confidence,
                                    "model_responses": reason_responses,
                                    "model_answers": reason_answers,
                                    "model_confidences": reason_confidences,
                                    "correct": reason_correct,
                                    "ground_truth": ground_truth_reason
                                }
                        
                        video_results[q_type] = result
                    except Exception as e:
                        logger.error(f"Error processing {q_type} question for {video_id}: {e}")
                        continue
                        
            results.append(video_results)
            
            # Save results after each video to avoid losing progress
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_results(self, results: List[Dict[str, Any]]):
        """Evaluate the results of the question answering"""
        metrics = {}
        
        # Overall accuracy
        correct_answers = 0
        total_answers = 0
        
        # Accuracy by question type
        for q_type in ["descriptive", "explanatory", "predictive", "counterfactual"]:
            metrics[q_type] = {
                "correct": 0,
                "total": 0,
                "accuracy": 0.0
            }
            
            # For predictive and counterfactual, also track reason accuracy
            if q_type in ["predictive", "counterfactual"]:
                metrics[f"{q_type}_reason"] = {
                    "correct": 0,
                    "total": 0,
                    "accuracy": 0.0
                }
        
        # Calculate metrics
        for result in results:
            for q_type in ["descriptive", "explanatory", "predictive", "counterfactual"]:
                if q_type in result and result[q_type].get("correct") is not None:
                    metrics[q_type]["total"] += 1
                    total_answers += 1
                    
                    if result[q_type]["correct"]:
                        metrics[q_type]["correct"] += 1
                        correct_answers += 1
                    
                    # For predictive and counterfactual, also check reason
                    if q_type in ["predictive", "counterfactual"] and "reason" in result[q_type]:
                        if result[q_type]["reason"].get("correct") is not None:
                            metrics[f"{q_type}_reason"]["total"] += 1
                            
                            if result[q_type]["reason"]["correct"]:
                                metrics[f"{q_type}_reason"]["correct"] += 1
        
        # Calculate accuracies
        overall_accuracy = correct_answers / total_answers if total_answers > 0 else 0
        metrics["overall"] = {
            "correct": correct_answers,
            "total": total_answers,
            "accuracy": overall_accuracy
        }
        
        for metric in metrics:
            if metrics[metric]["total"] > 0:
                metrics[metric]["accuracy"] = metrics[metric]["correct"] / metrics[metric]["total"]
        
        return metrics

# Function to run the entire pipeline
def run_video_qa_pipeline(csv_path: str, output_dir: str = "results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the QA system
    system = VideoQASystem()
    
    # Process the CSV file
    results_path = os.path.join(output_dir, "qa_results.json")
    results = system.process_csv_file(csv_path, results_path)
    
    # Evaluate the results
    metrics = system.evaluate_results(results)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Overall accuracy: {metrics['overall']['accuracy']:.2%}")
    
    return results, metrics

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Narrative QA System")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    results, metrics = run_video_qa_pipeline(args.csv, args.output)
