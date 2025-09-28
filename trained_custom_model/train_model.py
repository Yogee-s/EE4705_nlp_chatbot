"""
Train a Seq2Seq chatbot model with periodic and selective checkpoint saving.
Command line arguments allow customizing training hyperparameters and save schedule.
"""

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, Callback
import pickle
import datetime

# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a Seq2Seq chatbot model.")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=0.003, help="Optimizer learning rate")
parser.add_argument("--save_every", type=int, default=0,
                    help="Save checkpoint every N epochs (0 = disabled)")
parser.add_argument("--save_epochs", type=str, default="",
                    help="Comma-separated list of epochs to save checkpoints (e.g., 10,50,100)")
parser.add_argument("--data", type=str, default="data/Conversation_augmented.csv",
                    help="Path to dataset CSV")
args = parser.parse_args()

# Parse explicit save epochs into a set of integers
save_epochs = set()
if args.save_epochs:
    save_epochs = set(int(x.strip()) for x in args.save_epochs.split(",") if x.strip().isdigit())

# Enable mixed precision and XLA for GPU speedup
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

# ----------------------------------------------------------------------
# 1. Load and preprocess dataset
# ----------------------------------------------------------------------
df = pd.read_csv(args.data)
questions = df["question"].astype(str).values
answers = df["answer"].astype(str).values

# Add start and end tokens to answers
answers_in = ["<start> " + ans for ans in answers]
answers_out = [ans + " <end>" for ans in answers]

# Build tokenizer
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(list(questions) + list(answers_in) + list(answers_out))
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Convert to padded sequences
encoder_in = pad_sequences(tokenizer.texts_to_sequences(questions), padding="post")
decoder_in = pad_sequences(tokenizer.texts_to_sequences(answers_in), padding="post")
decoder_out = pad_sequences(tokenizer.texts_to_sequences(answers_out), padding="post")

# Ensure outputs are shaped correctly for sparse categorical crossentropy
decoder_out = np.expand_dims(decoder_out, -1)

# ----------------------------------------------------------------------
# 2. Build Seq2Seq model
# ----------------------------------------------------------------------
EMB_DIM = 512
LATENT_DIM = 512
DROPOUT = 0.3

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(VOCAB_SIZE, EMB_DIM)(encoder_inputs)

encoder_bi = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, dropout=DROPOUT))(enc_emb)
encoder_lstm2 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=DROPOUT)
enc_output2, state_h2, state_c2 = encoder_lstm2(encoder_bi)

encoder_lstm3 = LSTM(LATENT_DIM, return_state=True, dropout=DROPOUT)
_, state_h3, state_c3 = encoder_lstm3(enc_output2)
encoder_states = [state_h3, state_c3]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(VOCAB_SIZE, EMB_DIM)(decoder_inputs)

decoder_lstm1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=DROPOUT)
dec_output1, _, _ = decoder_lstm1(dec_emb, initial_state=[state_h2, state_c2])

decoder_lstm2 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=DROPOUT)
dec_output2, _, _ = decoder_lstm2(dec_output1, initial_state=encoder_states)

decoder_lstm3 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=DROPOUT)
decoder_outputs, _, _ = decoder_lstm3(dec_output2, initial_state=encoder_states)

decoder_dense = Dense(VOCAB_SIZE, activation="softmax", dtype="float32")
decoder_outputs = decoder_dense(decoder_outputs)

# Full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
              loss="sparse_categorical_crossentropy")

# ----------------------------------------------------------------------
# 3. Callbacks
# ----------------------------------------------------------------------
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

class EpochSaver(Callback):
    """
    Saves the model and tokenizer at regular intervals (--save_every)
    and/or specific epochs (--save_epochs).
    """
    def __init__(self, save_every=0, save_epochs=None, tokenizer=None,
                 save_path="chatbot_epoch_{epoch:03d}.h5"):
        super().__init__()
        self.save_every = save_every
        self.save_epochs = save_epochs or set()
        self.save_path = save_path
        self.tokenizer = tokenizer

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        should_save = False

        # Save if matches interval
        if self.save_every and epoch_num % self.save_every == 0:
            should_save = True
        # Save if explicitly listed
        if epoch_num in self.save_epochs:
            should_save = True

        if should_save:
            # Save model
            path = self.save_path.format(epoch=epoch_num)
            self.model.save(path)
            print(f"Model checkpoint saved at epoch {epoch_num}: {path}")

            # Save tokenizer to match the checkpoint
            if self.tokenizer is not None:
                tok_path = f"tokenizer_epoch_{epoch_num:03d}.pkl"
                with open(tok_path, "wb") as f:
                    pickle.dump(self.tokenizer, f)
                print(f"Tokenizer saved at epoch {epoch_num}: {tok_path}")

epoch_saver_cb = EpochSaver(save_every=args.save_every,
                            save_epochs=save_epochs,
                            tokenizer=tokenizer)

# ----------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------
model.fit([encoder_in, decoder_in], decoder_out,
          batch_size=args.batch_size,
          epochs=args.epochs,
          validation_split=0.1,
          callbacks=[tensorboard_cb, epoch_saver_cb])

# ----------------------------------------------------------------------
# 5. Save final model and tokenizer
# ----------------------------------------------------------------------
model.save("chatbot_seq2seq.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training complete. Run TensorBoard with:")
print("    tensorboard --logdir logs/fit")
