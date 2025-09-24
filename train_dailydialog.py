import argparse
import os
import time
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import TextLoader
from model import Model
import pandas as pd


def build_input_file(data_dir):
    input_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(input_path):
        csv_files = ["train.csv", "validation.csv", "test.csv"]
        lines = []

        for csv_file in csv_files:
            csv_path = os.path.join(data_dir, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Assume dialogue text is in the first column
                text_col = df.columns[0]

                for row in df[text_col]:
                    if isinstance(row, str) and row.strip():
                        lines.append(row.strip())

        if not lines:
            raise FileNotFoundError(
                f"No usable text found in {data_dir}/train.csv, validation.csv, or test.csv"
            )

        with open(input_path, "w", encoding="utf-8") as f_out:
            for line in lines:
                f_out.write(line + "\n")

        print(f"[INFO] Created {input_path} from train/validation/test CSV files.")

    return input_path


def train(args):
    # Make sure input.txt exists or create it
    input_file = build_input_file(args.data_dir)

    # Load the dataset
    data_loader = TextLoader(
        args.data_dir,
        args.batch_size,
        args.seq_length,
        input_file=input_file,
        encoding='utf-8'
    )

    args.vocab_size = data_loader.vocab_size
    args.word_level = data_loader.word_level

    # Save model configuration
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Save vocab
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.tokens, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=args.keep_last_n)

        for epoch in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate))
            state = sess.run(model.zero_state)

            epoch_loss = 0
            start = time.time()

            for b in range(data_loader.num_batches):
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                model.add_state_to_feed_dict(feed, state)

                train_loss, state, _ = sess.run(
                    [model.cost, model.final_state, model.train_op],
                    feed_dict=feed
                )
                epoch_loss += train_loss

            duration = time.time() - start
            avg_loss = epoch_loss / data_loader.num_batches
            print(f"Epoch {epoch+1}/{args.num_epochs}, "
                  f"Loss: {avg_loss:.4f}, Time: {duration:.2f}s")

            # Save checkpoints according to --save_every
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, "model.ckpt")
                saver.save(sess, ckpt_path, global_step=epoch+1)
                print(f"[INFO] Saved checkpoint at epoch {epoch+1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dailydialog',
                        help='data directory containing train/validation/test CSVs')
    parser.add_argument('--save_dir', type=str, default='models/dailydialog',
                        help='directory to store checkpoints')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--block_size', type=int, default=128,
                        help='size of each RNN block/hidden state')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='number of horizontal partitions per layer')
    parser.add_argument('--seq_length', type=int, default=64,
                        help='sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='save frequency (epochs)')
    parser.add_argument('--keep_last_n', type=int, default=3,
                        help='number of checkpoints to keep')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')

    args = parser.parse_args()

    # Ensure save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)


if __name__ == '__main__':
    main()
