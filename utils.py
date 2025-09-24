import codecs
import os
import collections
import pickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length,
                 input_file=None, encoding='utf-8', word_level=False):
        """
        TextLoader for training RNNs.

        Args:
            data_dir (str): directory to save/load vocab and tensor files
            batch_size (int): number of sequences per batch
            seq_length (int): sequence length
            input_file (str): path to input.txt (preprocessed text)
            encoding (str): text encoding
            word_level (bool): if True, use word-level tokens, else char-level
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.word_level = word_level

        if input_file is None:
            input_file = os.path.join(data_dir, "input.txt")

        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # If no cache, preprocess input file
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("Reading text file and creating vocab...")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("Loading preprocessed files...")
            self.load_preprocessed(vocab_file, tensor_file)

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()

        # Tokenize
        if self.word_level:
            # Split into words
            tokens = data.split()
        else:
            # Split into characters
            tokens = list(data)

        # Build vocab (sorted by frequency)
        counter = collections.Counter(tokens)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.tokens, _ = zip(*count_pairs)
        self.vocab_size = len(self.tokens)
        self.vocab = dict(zip(self.tokens, range(len(self.tokens))))

        # Save vocabulary
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.tokens, f)

        # Convert text to integer array
        self.tensor = np.array(list(map(self.vocab.get, tokens)), dtype=np.int32)

        # Save tensor
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.tokens = pickle.load(f)
        self.vocab_size = len(self.tokens)
        self.vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.tensor = np.load(tensor_file)

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        if self.num_batches == 0:
            raise ValueError("Not enough data. Reduce seq_length and/or batch_size.")

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
