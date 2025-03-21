Advancing Causal Video Question Answering: Integrating Language Models for Enhanced Reasoning

Project Description | Abstract:

Video Question Answering (VidQA) is a challenging task in artificial intelligence (AI) that requires a model to understand a video and answer questions about it. Traditional VidQA models primarily focus on recognizing objects and actions but struggle with causal reasoning, which involves understanding why events occur and predicting what might happen next. This limitation makes it difficult for AI to infer relationships between actions and their effects, leading to poor performance in real-world applications that require deep understanding.
Causal Video Question Answering (CVidQA) addresses this challenge by extending VidQA to reasoning-based questions. These questions go beyond simple descriptions and require explanations, predictions, and counterfactual thinking (i.e., "What would happen if..."). However, current CVidQA models still struggle due to a lack of structured causal knowledge and limited training data that explicitly encode these relations.

Importance and Relevance of the Problem:


The ability to answer causal questions from videos is crucial for various AI applications, including:
•	Autonomous Systems: To avoid accidents, self-driving cars need to predict human actions and understand cause-effect relationships.
•	Healthcare: AI can assist in medical diagnosis by analyzing procedural videos and inferring outcomes.
•	Education: Intelligent tutoring systems can evaluate student experiments and provide feedback on cause-and-effect relationships.
•	Surveillance & Security: AI models can predict potential security threats based on observed behavior.
Understanding causality is fundamental to human-like reasoning, making CVidQA an essential research problem in AI and machine learning.

Input Space:

1.	Video clips: Short video sequences containing human actions, interactions, and objects.
2.	Natural language questions: These questions fall into four categories:
o	Description: "What is happening in the video?"
o	Explanation: "Why did person A do X?"
o	Prediction: "What will happen next?"
o	Counterfactual: "What if event Y had not occurred?"
The model must process both video and text, requiring multimodal learning capabilities.

Output Space:

The output is a text-based answer to the given question. Depending on the task, the model may:
•	Select the correct answer from multiple choices (e.g., "A, B, C, or D").
•	Generate an open-ended answer (e.g., "The person will fall because they lost their balance").
•	Explain its reasoning (especially for prediction and counterfactual questions).
For evaluation, metrics such as accuracy, BLEU, METEOR, and human judgment are used to measure performance.

Prior Work Using Language Models:

Researchers have attempted to use pre-trained language models (PLMs) to solve CVidQA. Some notable approaches include:
•	CaKE-LM: Extracts causal commonsense knowledge from large language models (LLMs) like GPT-3 to generate high-quality QA pairs and improve zero-shot performance.
•	Visual Causal Scene Refinement (VCSR): Enhances video question answering by refining key visual cues and reducing spurious correlations.
•	Collaborative Bidirectional Semantic Reasoning (CBSR): Uses multi-granularity semantic aggregation to dynamically mine essential video segments.
However, these methods still face challenges, including:
•	Poor generalization: Pre-trained LMs struggle to adapt causal reasoning to unseen scenarios.
•	Lack of video-text alignment: While LMs excel in textual reasoning, integrating them with video understanding remains a major challenge.
•	Data efficiency: Many models require large-scale annotated datasets that are expensive.

Available Data for Training and Testing:

Internal data: Available with the Professor for the ongoing research work within her lab 

External data:
•	Causal-VidQA: A large-scale dataset with 26,900 videos and 107,600 QA pairs categorized into description, explanation, prediction, and counterfactual questions.
•	NExT-QA: A benchmark dataset focusing on causal and temporal reasoning in video QA.
•	MSRVTT-QA & MSVD-QA: Popular datasets for general video question answering, often used to test baseline performance

## Dataset
The complete dataset can be found [here](https://drive.google.com/drive/folders/1Zdcv9-jyi7y8f5IV-yReygtXntFTJyhe).
