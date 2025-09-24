from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import os
import pickle
import copy
import sys
import html
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models/reddit',
                        help='model directory to load checkpoints from')
    parser.add_argument('-n', type=int, default=200,
                        help='number of tokens (chars or words) to sample')
    parser.add_argument('--prime', type=str, default='',
                        help='prime text')
    parser.add_argument('--beam_width', type=int, default=2,
                        help='Width of the beam for beam search')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='sampling temperature')
    parser.add_argument('--topn', type=int, default=-1,
                        help='top-n filtering; set <0 to disable')
    args = parser.parse_args()
    sample_main(args)


def get_paths(input_path):
    if os.path.isfile(input_path):
        model_path = input_path
        save_dir = os.path.dirname(model_path)
    elif os.path.exists(input_path):
        save_dir = input_path
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if checkpoint:
            model_path = checkpoint.model_checkpoint_path
        else:
            raise ValueError(f'Checkpoint not found in {save_dir}')
    else:
        raise ValueError('save_dir is not a valid path.')
    return model_path, os.path.join(save_dir, 'config.pkl'), os.path.join(save_dir, 'chars_vocab.pkl')


def sample_main(args):
    model_path, config_path, vocab_path = get_paths(args.save_dir)

    # Load training-time arguments
    with open(config_path, 'rb') as f:
        saved_args = pickle.load(f)

    # Load vocab
    with open(vocab_path, 'rb') as f:
        tokens, vocab = pickle.load(f)

    # Auto-detect word-level or char-level
    word_level = getattr(saved_args, "word_level", False)

    print(f"[INFO] Loaded model from {args.save_dir} | Word-level: {word_level}")

    # Build model
    saved_args.batch_size = args.beam_width
    saved_args.seq_length = 1
    net = Model(saved_args, infer=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(net.save_variables_list())
        saver.restore(sess, model_path)

        while True:
            user_input = input("\n> ")
            if not user_input.strip():
                continue

            # Convert input to tokens
            if word_level:
                prime_tokens = [t for t in user_input.split() if t in vocab]
            else:
                prime_tokens = [c for c in user_input if c in vocab]

            state = sess.run(net.zero_state)

            # Prime the model
            for tok in prime_tokens:
                _, state = net.forward_model(sess, state, vocab[tok])

            # Generate response
            out_tokens = []
            sample = vocab.get(" ", list(vocab.values())[0])
            for i in range(args.n):
                probs, state = net.forward_model(sess, state, sample)
                if args.topn > 0:
                    probs[np.argsort(probs)[:-args.topn]] = 0
                probs = probs / np.sum(probs)
                sample = np.random.choice(len(probs), p=probs)
                token = tokens[sample]
                out_tokens.append(token)
                if token in ['\n', '<eos>']:
                    break

            if word_level:
                print(" ".join(out_tokens))
            else:
                print("".join(out_tokens))


if __name__ == '__main__':
    main()
