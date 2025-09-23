import argparse
import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tkinter as tk
from tkinter import scrolledtext
import threading

# For speech
import speech_recognition as sr
import pyttsx3

from model import Model   # ✅ updated multi-task model


# ---------------------------
# Load vocab + model
# ---------------------------
def load_vocab(save_dir):
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"❌ vocab.pkl not found in {save_dir}")
    with open(vocab_path, "rb") as f:
        vocab, inv_vocab = pickle.load(f)
    return vocab, inv_vocab


def load_model(session, args, saver):
    checkpoint = tf.train.latest_checkpoint(args.save_dir)
    if checkpoint:
        saver.restore(session, checkpoint)
        print(f"✅ Restored from {checkpoint}")
    else:
        raise RuntimeError(f"❌ No checkpoint found in {args.save_dir}. Train the model first.")


# ---------------------------
# Sampling helper
# ---------------------------
def sample_with_temperature(probs, temperature=1.0):
    probs = np.asarray(probs).astype("float64")
    probs = np.log(probs + 1e-8) / temperature
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    return np.random.choice(len(probs), p=probs)


# ---------------------------
# Generate response + classify act/emotion
# ---------------------------
ACT_LABELS = ["Inform", "Question", "Directive", "Commissive"]
EMO_LABELS = ["No Emotion", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

def generate_response(sess, model, vocab, inv_vocab, text,
                      seq_length=20, max_len=40, block_size=128, temperature=0.8):
    tokens = text.lower().split()
    token_ids = [vocab.get(tok, 1) for tok in tokens]  # <UNK>=1

    # Pad/truncate
    if len(token_ids) < seq_length:
        token_ids = [0] * (seq_length - len(token_ids)) + token_ids
    else:
        token_ids = token_ids[-seq_length:]

    x = np.array([token_ids])

    feed = {
        model.input_data: x,
        model.targets: np.zeros((1, seq_length)),   # dummy
        model.act_targets: np.zeros(1),             # dummy
        model.emotion_targets: np.zeros(1)          # dummy
    }

    # Step 1: classify act + emotion
    act_logits, emo_logits = sess.run([model.act_logits, model.emotion_logits], feed_dict=feed)
    act_pred = ACT_LABELS[np.argmax(act_logits[0])]
    emo_pred = EMO_LABELS[np.argmax(emo_logits[0])]

    # Step 2: generate text
    response_ids = []
    prev_input = x.copy()
    for _ in range(max_len):
        preds = sess.run(model.probs, feed_dict=feed)
        next_id = sample_with_temperature(preds[0], temperature)
        response_ids.append(next_id)
        prev_input = np.roll(prev_input, -1, axis=1)
        prev_input[0, -1] = next_id
        feed[model.input_data] = prev_input

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
    def __init__(self, root, sess, model, vocab, inv_vocab, block_size):
        self.sess = sess
        self.model = model
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.block_size = block_size
        self.voice_mode = False

        root.title("🤖 AI Companion")
        root.geometry("600x500")
        root.configure(bg="#1e1e1e")

        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, bg="#252526", fg="#d4d4d4",
            insertbackground="white", font=("Consolas", 11)
        )
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, "🤖 Companion Ready!\n")

        bottom_frame = tk.Frame(root, bg="#1e1e1e")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.entry = tk.Entry(bottom_frame, font=("Consolas", 11),
                              bg="#333333", fg="white", insertbackground="white")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(bottom_frame, text="Send", command=self.send_message,
                                  bg="#007acc", fg="white")
        self.send_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.voice_btn = tk.Button(bottom_frame, text="🎤 Voice", command=self.toggle_voice,
                                   bg="#007acc", fg="white")
        self.voice_btn.pack(side=tk.LEFT)

    def toggle_voice(self):
        self.voice_mode = not self.voice_mode
        if self.voice_mode:
            self.display_message("System", "🎤 Voice mode enabled")
            threading.Thread(target=self.voice_loop, daemon=True).start()
        else:
            self.display_message("System", "⌨️ Text mode enabled")

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
            self.sess, self.model, self.vocab, self.inv_vocab,
            user_text, block_size=self.block_size
        )
        self.display_message("Companion", f"{response}\n   🏷 Act: {act} | Emotion: {emo}")
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
        block_size=128,
        model="gru",
        num_layers=2,
        num_blocks=2,
        seq_length=20,
        batch_size=1,
        learning_rate=0.002,
        grad_clip=5.0,
        vocab_size=0
    )

    vocab, inv_vocab = load_vocab(args.save_dir)
    args.vocab_size = len(vocab)

    model = Model(args, infer=True)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        load_model(sess, args, saver)
        root = tk.Tk()
        gui = ChatGUI(root, sess, model, vocab, inv_vocab, args.block_size)
        root.mainloop()
