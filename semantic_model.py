import argparse
import os
import sys
import pandas as pd
import torch
import re
import random
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------
# Load or download retriever model (Sentence-BERT)
# ------------------------------------------------------
def load_retriever(model_dir="models/retriever"):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if not os.path.exists(model_dir):
        print("Downloading retriever model...")
        model = SentenceTransformer(model_name)
        model.save(model_dir)
    else:
        print(f"Loading retriever from local folder: {model_dir}")
        model = SentenceTransformer(model_dir)
    return model

# ------------------------------------------------------
# Load or download paraphraser model (Pegasus)
# ------------------------------------------------------
def load_paraphraser(model_dir="models/paraphraser"):
    model_name = "tuner007/pegasus_paraphrase"
    if not os.path.exists(model_dir):
        print("Downloading paraphraser model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        print(f"Loading paraphraser from local folder: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

# ------------------------------------------------------
# Preprocess text for better matching
# ------------------------------------------------------
def preprocess_text(text):
    """Clean and normalize text for better semantic matching"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?\!\.\,]', '', text)
    return text

# ------------------------------------------------------
# Handle greetings and casual interactions
# ------------------------------------------------------
def handle_greetings_and_casual(user_query):
    """Handle greetings, thanks, and casual questions"""
    query_lower = user_query.lower().strip()

    greeting_patterns = [
        r'^(hi|hello|hey|hiya)!?$',
        r'^(good morning|good afternoon|good evening)!?$',
        r'^(whats up|how are you|hows it going)[\?\!]?$'
    ]

    for pattern in greeting_patterns:
        if re.match(pattern, query_lower):
            responses = [
                "Hello! I'm here to help you with questions about NUS. What would you like to know?",
                "Hi there! Feel free to ask me anything about NUS courses, programs, or general information.",
                "Hey! I'm ready to answer your questions about NUS. What can I help you with?",
                "Hello! Ask me about NUS courses, requirements, or anything else you'd like to know."
            ]
            return random.choice(responses)

    if re.match(r'^(thanks?|thank you|thx)!?$', query_lower):
        responses = [
            "You're welcome! Feel free to ask if you have more questions.",
            "Happy to help! Is there anything else you'd like to know?",
            "No problem! Let me know if you need more information."
        ]
        return random.choice(responses)

    if re.match(r'^(help|what can you do|what do you know)[\?\!]?$', query_lower):
        return ("I can help you with questions about NUS including courses, programs, requirements, "
                "college information, and general university topics. Just ask me anything!")

    return None

# ------------------------------------------------------
# Semantic retrieval + answer selection
# ------------------------------------------------------
def select_best_answer(user_query, questions, answers, question_embeddings, retriever, args):
    """Find the best matching answer for a query with fallback logic"""

    # Check greetings/casual first
    casual_response = handle_greetings_and_casual(user_query)
    if casual_response:
        return casual_response, 1.0, None

    processed_query = preprocess_text(user_query)

    # Handle very short queries
    if len(processed_query.split()) < 2:
        important_keywords = [
            'nus', 'courses', 'modules', 'requirements', 'fees',
            'admission', 'engineering', 'computing', 'business',
            'medicine', 'science'
        ]
        if processed_query not in important_keywords:
            return None, 0.0, "Could you ask a more detailed question? For example, you could ask about NUS courses, requirements, or specific programs."

    # Encode and compute similarity
    query_embedding = retriever.encode(processed_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]

    top_indices = torch.topk(scores, min(3, len(scores)))[1]
    top_scores = [float(scores[idx]) for idx in top_indices]

    best_idx = int(top_indices[0])
    best_score = top_scores[0]

    if best_score < args.threshold:
        if len(top_scores) > 1 and top_scores[1] > args.threshold * 0.8:
            return answers[int(top_indices[1])], top_scores[1], None
        else:
            fallback_responses = [
                "I'm not sure about that specific topic. Could you ask about NUS courses, requirements, or programs?",
                "I don't have information on that. Try asking about NUS academics, admissions, or student life.",
                "That's not in my knowledge base. Feel free to ask about university courses, policies, or general information.",
                "I'm here to help with NUS-related questions. What would you like to know about the university?"
            ]
            return None, best_score, random.choice(fallback_responses)

    return answers[best_idx], best_score, None

# ------------------------------------------------------
# Smart paraphrasing logic
# ------------------------------------------------------
def smart_paraphrase(answer, user_query, tokenizer, model, similarity_score):
    """
    Paraphrase answers only when appropriate:
    - Skip short answers, factual/numeric answers, or high-confidence matches
    - Otherwise rephrase for naturalness
    """
    if len(answer.split()) < 5:
        return answer
    if similarity_score > 0.9:
        return answer
    if re.search(r'\b\d+\b', answer) or len(re.findall(r'[A-Z]{2,}', answer)) > 0:
        return answer

    try:
        inputs = tokenizer([answer], return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=min(80, len(answer.split()) * 2),
            num_beams=2,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if len(paraphrased.split()) < len(answer.split()) * 0.5:
            return answer
        return paraphrased
    except Exception as e:
        print(f"Paraphrasing failed: {e}")
        return answer

# ------------------------------------------------------
# Main chatbot loop
# ------------------------------------------------------
def run_chatbot(args):
    # Load models
    print("Loading retriever model...")
    retriever = load_retriever()

    print("Loading paraphrasing model...")
    para_tokenizer, para_model = load_paraphraser()

    # Load datasets
    print("Loading datasets...")
    try:
        df1 = pd.read_csv("data/Conversation.csv")
        df2 = pd.read_csv("data/nus_qna_augmented.csv")
        df = pd.concat([df1, df2], ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure your CSV files are in the 'data/' directory")
        return

    df = df.dropna(subset=['question', 'answer'])
    questions = [preprocess_text(q) for q in df["question"].astype(str).tolist()]
    answers = df["answer"].astype(str).tolist()

    valid_pairs = [(q, a) for q, a in zip(questions, answers) if q.strip() and a.strip()]
    questions, answers = zip(*valid_pairs) if valid_pairs else ([], [])

    if not questions:
        print("No valid question-answer pairs found!")
        return

    print(f"Loaded {len(questions)} question-answer pairs")
    print("Encoding dataset questions...")
    question_embeddings = retriever.encode(questions, convert_to_tensor=True)

    print(f"\nChatbot ready! (Threshold: {args.threshold})")
    print("Type 'quit', 'exit', or 'bye' to end.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print("Chatbot: Goodbye! Have a great day.")
            break

        if not user_input:
            print("Chatbot: I'm here to help! Ask me anything about NUS.")
            continue

        best_answer, similarity_score, fallback_msg = select_best_answer(
            user_input, questions, answers, question_embeddings, retriever, args
        )

        if fallback_msg:
            print(f"Chatbot: {fallback_msg}")
        else:
            final_answer = smart_paraphrase(
                best_answer, user_input, para_tokenizer, para_model, similarity_score
            )
            print(f"Chatbot: {final_answer}")

            if args.show_confidence:
                print(f"[Confidence: {similarity_score:.3f}]")

# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic retrieval chatbot with paraphrasing")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Similarity threshold (default=0.4)")
    parser.add_argument("--show-confidence", action="store_true",
                        help="Show confidence scores for debugging")

    args = parser.parse_args()

    if not 0.1 <= args.threshold <= 1.0:
        print("Warning: Threshold should be between 0.1 and 1.0")

    try:
        run_chatbot(args)
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
