import os
import re
import ast
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from collections import Counter
from model import Model
from tqdm import tqdm
import csv

# ---------------------------
# Tokenization
# ---------------------------
def tokenize(sentence):
    return re.findall(r"\w+|[^\w\s]", sentence.lower(), re.UNICODE)

# ---------------------------
# Load dataset (train/val/test)
# ---------------------------
def parse_labels(x):
    # Handles strings like "[3 4 2 2 2 3]" or "[3,4,2,2]"
    x = x.strip("[]")
    if not x:
        return []
    return [int(tok) for tok in re.split(r"[ ,]+", x.strip()) if tok]

def load_dailydialog(data_path, split, seq_length=20, vocab=None):
    df = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
    df['dialog'] = df['dialog'].apply(ast.literal_eval)
    df['act'] = df['act'].apply(parse_labels)
    df['emotion'] = df['emotion'].apply(parse_labels)

    all_sentences, all_acts, all_emotions = [], [], []
    for conv, acts, emos in zip(df['dialog'], df['act'], df['emotion']):
        for sent, act, emo in zip(conv, acts, emos):
            tokens = tokenize(sent)
            all_sentences.extend(tokens)
            all_acts.extend([act] * len(tokens))
            all_emotions.extend([emo] * len(tokens))

    tokens = []
    for sent in all_sentences:
        tokens.extend(tokenize(sent))

    if vocab is None:
        word_counts = Counter(tokens)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, (word, _) in enumerate(word_counts.most_common(), start=2):
            vocab[word] = idx
        inv_vocab = {idx: word for word, idx in vocab.items()}
    else:
        inv_vocab = {idx: word for word, idx in vocab.items()}

    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

    X, Y, A, E = [], [], [], []
    for i in range(0, len(token_ids) - seq_length):
        X.append(token_ids[i:i+seq_length])
        Y.append(token_ids[i+1:i+seq_length+1])
        A.append(all_acts[i+seq_length-1])
        E.append(all_emotions[i+seq_length-1])

    return np.array(X), np.array(Y), np.array(A), np.array(E), vocab, inv_vocab

# ---------------------------
# Zero state helper
# ---------------------------
def zero_state_feed(initial_state, batch_size, block_size):
    feed_dict = {}

    def fill(state):
        if isinstance(state, (list, tuple)):
            return [fill(s) for s in state]
        else:
            return np.zeros((batch_size, block_size), dtype=np.float32)

    def assign(phs, vals):
        if isinstance(phs, (list, tuple)) and isinstance(vals, (list, tuple)):
            for p, v in zip(phs, vals):
                assign(p, v)
        else:
            feed_dict[phs] = vals

    zero_vals = fill(initial_state)
    assign(initial_state, zero_vals)
    return feed_dict

# ---------------------------
# Evaluate helper
# ---------------------------
def evaluate(sess, model, X, Y, A, E, args, split_name="Validation"):
    num_batches = max(1, len(X) // args.batch_size)
    total_loss, total_acc = 0.0, 0.0
    total_act_acc, total_emo_acc = 0.0, 0.0
    count = 0

    for i in range(0, len(X), args.batch_size):
        x_batch = X[i:i+args.batch_size]
        y_batch = Y[i:i+args.batch_size]
        a_batch = A[i:i+args.batch_size]
        e_batch = E[i:i+args.batch_size]
        if len(x_batch) < args.batch_size:
            continue

        feed = {
            model.input_data: x_batch,
            model.targets: y_batch,
            model.act_targets: a_batch,
            model.emotion_targets: e_batch
        }
        feed.update(zero_state_feed(model.initial_state, args.batch_size, args.block_size))

        loss, acc, act_acc, emo_acc = sess.run(
            [model.cost, model.accuracy, model.act_accuracy, model.emotion_accuracy],
            feed_dict=feed
        )
        total_loss += loss
        total_acc += acc
        total_act_acc += act_acc
        total_emo_acc += emo_acc
        count += 1

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    avg_act_acc = total_act_acc / count
    avg_emo_acc = total_emo_acc / count
    perplexity = np.exp(avg_loss)

    print(f"🔍 {split_name} | Loss: {avg_loss:.4f} | LM Acc: {avg_acc:.4f} | "
          f"Act Acc: {avg_act_acc:.4f} | Emo Acc: {avg_emo_acc:.4f} | PPL: {perplexity:.2f}")
    return avg_loss, avg_acc, avg_act_acc, avg_emo_acc, perplexity

# ---------------------------
# Main Training Loop
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dailydialog")
    parser.add_argument("--save_dir", type=str, default="models/dailydialog")
    parser.add_argument("--model", type=str, default="gru", help="rnn | gru | lstm")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    args = parser.parse_args()

    print("📂 Loading dataset...")
    X_train, Y_train, A_train, E_train, vocab, inv_vocab = load_dailydialog(args.data_dir, "train", args.seq_length)
    X_val, Y_val, A_val, E_val, _, _ = load_dailydialog(args.data_dir, "validation", args.seq_length, vocab)
    X_test, Y_test, A_test, E_test, _, _ = load_dailydialog(args.data_dir, "test", args.seq_length, vocab)
    args.vocab_size = len(vocab)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump((vocab, inv_vocab), f)

    # 📊 CSV logger
    log_path = os.path.join(args.save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_act_acc", "train_emo_acc", "train_ppl",
            "val_loss", "val_acc", "val_act_acc", "val_emo_acc", "val_ppl"
        ])

    print(f"✅ Vocab size: {len(vocab)}")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    model = Model(args)
    saver = tf.train.Saver(max_to_keep=3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # TensorBoard writer
    tb_writer = tf.summary.FileWriter(os.path.join(args.save_dir, "logs"))

    with tf.Session(config=config) as sess, open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        sess.run(tf.global_variables_initializer())
        num_batches = max(1, len(X_train) // args.batch_size)

        print(f"🚀 Training for {args.num_epochs} epochs, {num_batches} batches/epoch")

        best_val_loss = float("inf")

        for epoch in range(args.num_epochs):
            epoch_loss, epoch_acc, epoch_act_acc, epoch_emo_acc = 0.0, 0.0, 0.0, 0.0
            count = 0

            for i in tqdm(range(0, len(X_train), args.batch_size),
                          desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch"):
                x_batch = X_train[i:i+args.batch_size]
                y_batch = Y_train[i:i+args.batch_size]
                a_batch = A_train[i:i+args.batch_size]
                e_batch = E_train[i:i+args.batch_size]
                if len(x_batch) < args.batch_size:
                    continue

                feed = {
                    model.input_data: x_batch,
                    model.targets: y_batch,
                    model.act_targets: a_batch,
                    model.emotion_targets: e_batch
                }
                feed.update(zero_state_feed(model.initial_state, args.batch_size, args.block_size))

                train_loss, train_acc, train_act_acc, train_emo_acc, summary, _ = sess.run(
                    [model.cost, model.accuracy, model.act_accuracy,
                     model.emotion_accuracy, model.summary_op, model.train_op],
                    feed_dict=feed
                )
                tb_writer.add_summary(summary, epoch * num_batches + count)

                epoch_loss += train_loss
                epoch_acc += train_acc
                epoch_act_acc += train_act_acc
                epoch_emo_acc += train_emo_acc
                count += 1

            avg_loss = epoch_loss / count
            avg_acc = epoch_acc / count
            avg_act_acc = epoch_act_acc / count
            avg_emo_acc = epoch_emo_acc / count
            perplexity = np.exp(avg_loss)

            print(f"\n📊 Epoch {epoch+1}/{args.num_epochs} | "
                  f"Train Loss: {avg_loss:.4f} | LM Acc: {avg_acc:.4f} | "
                  f"Act Acc: {avg_act_acc:.4f} | Emo Acc: {avg_emo_acc:.4f} | PPL: {perplexity:.2f}")

            val_loss, val_acc, val_act_acc, val_emo_acc, val_ppl = evaluate(
                sess, model, X_val, Y_val, A_val, E_val, args, "Validation"
            )

            # 🔥 Save CSV row
            writer.writerow([
                epoch+1, avg_loss, avg_acc, avg_act_acc, avg_emo_acc, perplexity,
                val_loss, val_acc, val_act_acc, val_emo_acc, val_ppl
            ])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.save_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=epoch+1)
                print(f"💾 Model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        print("\n🏁 Training complete. Evaluating on test set...")
        evaluate(sess, model, X_test, Y_test, A_test, E_test, args, "Test")
        
if __name__ == "__main__":
    main()
