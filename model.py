import tensorflow as tf
from tensorflow.keras import layers, Model as KerasModel


class Model(KerasModel):
    def __init__(self, args):
        super(Model, self).__init__()
        self.vocab_size = args.vocab_size
        self.block_size = args.block_size
        self.num_layers = args.num_layers

        # Embedding layer for words
        self.embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.block_size,
            mask_zero=True,
            name="embedding"
        )

        # Stacked GRU layers
        self.rnn_layers = []
        for i in range(self.num_layers):
            self.rnn_layers.append(
                layers.GRU(
                    self.block_size,
                    return_sequences=True if i < self.num_layers - 1 else True,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    name=f"gru_layer_{i+1}"
                )
            )

        # Language Modeling head
        self.lm_dense = layers.Dense(self.vocab_size, name="lm")

        # Classification heads for dialog act and emotion
        self.act_dense = layers.Dense(5, activation="softmax", name="act")   # DailyDialog has 4-5 acts
        self.emo_dense = layers.Dense(7, activation="softmax", name="emo")   # DailyDialog has 7 emotions

    def call(self, inputs, training=False):
        """
        Forward pass
        Args:
            inputs: token IDs of shape (batch, seq_len)
        Returns:
            dict with lm, act, emo outputs
        """
        x = self.embedding(inputs)

        for rnn in self.rnn_layers:
            x = rnn(x, training=training)

        # Language modeling: predict next token for each time step
        lm_logits = self.lm_dense(x)

        # For act and emotion, take only the last hidden state
        last_state = x[:, -1, :]  # shape: (batch, hidden)

        act_logits = self.act_dense(last_state)
        emo_logits = self.emo_dense(last_state)

        return {"lm": lm_logits, "act": act_logits, "emo": emo_logits}
