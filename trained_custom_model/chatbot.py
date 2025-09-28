import argparse
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run chatbot with sampling options.")
parser.add_argument("--model_path", type=str, default="../models/custom/chatbot_epoch_200.h5",
                    help="Path to trained .h5 model file")
parser.add_argument("--tokenizer_path", type=str, default="../models/custom/tokenizer_epoch_200.pkl",
                    help="Path to tokenizer.pkl file")
parser.add_argument("--top_k", type=int, default=5,
                    help="Top-k sampling (default=5)")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (default=1.0)")
parser.add_argument("--max_len", type=int, default=50,
                    help="Fallback max sequence length if not saved in pickle")
args = parser.parse_args()

# -----------------------------
# Load model and tokenizer
# -----------------------------
model = load_model(args.model_path)

with open(args.tokenizer_path, "rb") as f:
    data = pickle.load(f)

# Backward compatibility: old pickles only stored tokenizer
if isinstance(data, dict):
    tokenizer = data["tokenizer"]
    MAX_LEN = data.get("max_len", args.max_len)
else:
    tokenizer = data
    # Fallback: use CLI default if max_len not available
    MAX_LEN = args.max_len

word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

# -----------------------------
# Sampling helper
# -----------------------------
def sample_next(preds, top_k=5, temperature=1.0):
    """Sample next token index using top-k + temperature."""
    preds = np.asarray(preds).astype("float64")

    # Apply temperature
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # Select top-k tokens
    top_indices = preds.argsort()[-top_k:]
    top_probs = preds[top_indices]
    top_probs = top_probs / np.sum(top_probs)

    # Randomly pick one of the top-k
    return np.random.choice(top_indices, p=top_probs)

# -----------------------------
# Decode function
# -----------------------------
def decode_sequence(input_text, top_k=5, temperature=1.0):
    """Generate response for a given input text."""
    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    # Start token
    target_seq = tokenizer.texts_to_sequences(["<start>"])[0]
    decoded_sentence = []

    for _ in range(MAX_LEN):
        target_seq_pad = pad_sequences([target_seq], maxlen=MAX_LEN, padding="post")

        preds = model.predict([seq, target_seq_pad], verbose=0)
        next_probs = preds[0, len(target_seq) - 1, :]

        next_index = sample_next(next_probs, top_k=top_k, temperature=temperature)
        next_word = index_word.get(next_index, "")

        if next_word == "<end>" or next_word == "":
            break
        decoded_sentence.append(next_word)

        target_seq = np.append(target_seq, next_index)

    return " ".join(decoded_sentence)

# -----------------------------
# Chat loop
# -----------------------------
print("Chatbot ready! Type 'quit' to stop.")
while True:
    user_in = input("\nYou: ")
    if user_in.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    response = decode_sequence(user_in, top_k=args.top_k, temperature=args.temperature)
    print("Chatbot:", response)
