import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import scrolledtext
import threading

# Optional voice I/O
import speech_recognition as sr
import pyttsx3

from model import Model  # your trained model class

# ---------------------------
# Load vocab
# ---------------------------
def load_vocab(save_dir):
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.pkl not found in {save_dir}")
    with open(vocab_path, "rb") as f:
        vocab, inv_vocab = pickle.load(f)
    return vocab, inv_vocab

# ---------------------------
# Labels
# ---------------------------
ACT_LABELS = ["Inform", "Question", "Directive", "Commissive"]
EMO_LABELS = ["No Emotion", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

# ---------------------------
# Fast generation with tf.function
# ---------------------------
@tf.function
def generate_step(model, inputs):
    outputs = model(inputs, training=False)
    logits = outputs["lm"][:, -1, :]
    probs = tf.nn.softmax(logits)
    return outputs["act"], outputs["emo"], probs

def generate_response(model, vocab, inv_vocab, text,
                      seq_length=20, max_len=20, temperature=0.7, top_k=10):
    tokens = text.lower().split()
    token_ids = [vocab.get(tok, 1) for tok in tokens]

    if len(token_ids) < seq_length:
        token_ids = [0] * (seq_length - len(token_ids)) + token_ids
    else:
        token_ids = token_ids[-seq_length:]

    x = np.array([token_ids], dtype=np.int32)

    # First step: classify act & emotion
    act_logits, emo_logits, _ = generate_step(model, x)
    act_pred = ACT_LABELS[tf.argmax(act_logits[0]).numpy()]
    emo_pred = EMO_LABELS[tf.argmax(emo_logits[0]).numpy()]

    # Autoregressive generation
    response_ids = []
    prev_input = tf.convert_to_tensor(x)
    for _ in range(max_len):
        act_logits, emo_logits, probs = generate_step(model, prev_input)
        probs = probs[0].numpy()

        # top-k sampling
        top_k_indices = probs.argsort()[-top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)
        next_id = np.random.choice(top_k_indices, p=top_k_probs)

        response_ids.append(next_id)
        word = inv_vocab.get(next_id, "")
        if word in [".", "!", "?", "<eos>"]:
            break

        # roll input window
        arr = prev_input.numpy()
        arr = np.roll(arr, -1, axis=1)
        arr[0, -1] = next_id
        prev_input = tf.convert_to_tensor(arr)

    response = " ".join([inv_vocab.get(i, "<unk>") for i in response_ids])
    return response.strip(), act_pred, emo_pred

# ---------------------------
# Voice I/O
# ---------------------------
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# ---------------------------
# GUI
# ---------------------------
class ChatGUI:
    def __init__(self, root, model, vocab, inv_vocab, seq_length):
        self.model = model
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.seq_length = seq_length
        self.voice_mode = False

        # Default settings
        self.max_len = tk.IntVar(value=15)
        self.temperature = tk.DoubleVar(value=0.8)
        self.top_k = tk.IntVar(value=8)

        root.title("AI Companion")
        root.geometry("700x600")
        root.configure(bg="#1e1e1e")

        # Chat area
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, bg="#252526", fg="#d4d4d4",
            insertbackground="white", font=("Consolas", 11)
        )
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, "Companion Ready!\n")

        # Controls
        control_frame = tk.Frame(root, bg="#1e1e1e")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Max Len:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        tk.Entry(control_frame, textvariable=self.max_len, width=4).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Temp:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        tk.Entry(control_frame, textvariable=self.temperature, width=4).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Top-K:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        tk.Entry(control_frame, textvariable=self.top_k, width=4).pack(side=tk.LEFT, padx=5)

        # Input box
        bottom_frame = tk.Frame(root, bg="#1e1e1e")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.entry = tk.Entry(bottom_frame, font=("Consolas", 11),
                              bg="#333333", fg="white", insertbackground="white")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(bottom_frame, text="Send", command=self.send_message,
                                  bg="#007acc", fg="white", font=("Consolas", 10, "bold"))
        self.send_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.voice_btn = tk.Button(bottom_frame, text="Voice", command=self.toggle_voice,
                                   bg="#007acc", fg="white", font=("Consolas", 10, "bold"))
        self.voice_btn.pack(side=tk.LEFT)

    def toggle_voice(self):
        self.voice_mode = not self.voice_mode
        if self.voice_mode:
            self.display_message("System", "Voice mode enabled")
            threading.Thread(target=self.voice_loop, daemon=True).start()
        else:
            self.display_message("System", "Text mode enabled")

    def voice_loop(self):
        while self.voice_mode:
            user_text = listen()
            if not user_text.strip():
                continue
            self.process_message(user_text, sender="You", speak_out=True)

    def send_message(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, tk.END)
        self.process_message(user_text, sender="You")

    def process_message(self, user_text, sender="You", speak_out=False):
        self.display_message(sender, user_text)
        response, act, emo = generate_response(
            self.model, self.vocab, self.inv_vocab,
            user_text,
            seq_length=self.seq_length,
            max_len=self.max_len.get(),
            temperature=self.temperature.get(),
            top_k=self.top_k.get()
        )
        self.display_message("Companion", f"{response}\n   Act: {act} | Emotion: {emo}")
        if speak_out:
            speak(response)

    def display_message(self, sender, message):
        self.text_area.insert(tk.END, f"{sender}: {message}\n")
        self.text_area.see(tk.END)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = argparse.Namespace(
        save_dir="models/dailydialog",
        seq_length=20,
        block_size=256,
        model="gru",
        num_layers=3,
        batch_size=1,
        learning_rate=0.0001,
        vocab_size=0
    )

    vocab, inv_vocab = load_vocab(args.save_dir)
    args.vocab_size = len(vocab)

    model = Model(args)
    model.build(input_shape=(None, args.seq_length))
    model.load_weights(os.path.join(args.save_dir, "model.weights.h5"))

    root = tk.Tk()
    gui = ChatGUI(root, model, vocab, inv_vocab, args.seq_length)
    root.mainloop()
