import numpy as np
import tensorflow as tf
import tiktoken
import matplotlib.pyplot as plt
import re
from keras.layers import *
from keras import Sequential, Layer
from keras.preprocessing.sequence import pad_sequences

class PositionalEncoding(Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        pos_enc = np.zeros((sequence_length, embed_dim))
        for pos in range(sequence_length):
            for i in range(0, embed_dim, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / embed_dim)))
                if i + 1 < embed_dim:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / embed_dim)))
                    
        self.pos_enc = tf.constant(pos_enc, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)
        return inputs + self.pos_enc[:seq_len, :]

class Decoder(Layer):
    def __init__(self, vocabsize, sequence_length, embed_dim, ff_dim, num_heads, dropout=0.1):
        super().__init__()
        self.vocabsize = vocabsize 
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim, sequence_length=self.sequence_length)
        self.embeddings = Embedding(input_dim=self.vocabsize, output_dim=embed_dim)
        self.linear_layer = Dense(units=self.vocabsize, activation='softmax')
        
        self.masked_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )   
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.fnn = Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(self.embed_dim)
        ])
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)

    def call(self, inputs, training=False):
        inputs = self.embeddings(inputs)
        inputs = self.pos_encoding(inputs)
        
        seq_len = tf.shape(inputs)
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        
        m_attention_output = self.masked_attention(
            key=inputs,
            query=inputs,
            value=inputs,
            attention_mask=causal_mask
        )
        m_attention_output = self.dropout1(m_attention_output, training=training)
        x1 = self.norm1(m_attention_output + inputs)
        
        fnn_output = self.fnn(x1)
        fnn_output = self.dropout2(fnn_output, training=training)
        x2 = self.norm2(fnn_output + x1)
        
        return self.linear_layer(x2)

maxlen = 50 

with open('transformers-and-beyond/transformers/word_predictor/data.txt', 'r', encoding="utf-8") as file:
    data = file.read()

data = re.sub(r'\[.*?\]', '', data) 
data = " ".join(data.split())  

encoder = tiktoken.get_encoding('gpt2')
token = encoder.encode(data)

print(f"Total processed tokens: {len(token)}")

input_sequences = []
target_sequences = []

for i in range(0, len(token) - maxlen, 1):
    input_sequences.append(token[i:i + maxlen])
    target_sequences.append(token[i + 1:i + maxlen + 1])

padded_input_sequences = pad_sequences(input_sequences, maxlen=maxlen, padding='post', truncating='post')
target_sequences = np.array(target_sequences)

print(f"Total training patterns: {len(padded_input_sequences)}")

BATCH_SIZE = 16
dataset = tf.data.Dataset.from_tensor_slices((padded_input_sequences, target_sequences))
dataset = dataset.shuffle(buffer_size=len(padded_input_sequences)).batch(BATCH_SIZE, drop_remainder=True)

inputs = Input(shape=(maxlen,))
x = Decoder(vocabsize=encoder.n_vocab, sequence_length=maxlen, embed_dim=128, ff_dim=256, num_heads=4)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(dataset, epochs=10)

plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.show()

starting_text = "messi is"
generated_text = starting_text

print(f"\nStarting with: '{starting_text}'\n")
print("Generating next 10 words...")
print(starting_text, end="", flush=True)

for _ in range(10):
    current_tokens = encoder.encode(generated_text)
    
    if len(current_tokens) < maxlen:
        padded_tokens = current_tokens + [0] * (maxlen - len(current_tokens))
    else:
        padded_tokens = current_tokens[-maxlen:]
        
    model_input = np.array([padded_tokens])
    predictions = model.predict(model_input, verbose=0)
    
    active_index = min(len(current_tokens) - 1, maxlen - 1)
    next_token_id = np.argmax(predictions[0, active_index, :])
    
    next_word = encoder.decode([next_token_id])
    generated_text += next_word
    print(next_word, end="", flush=True)

print("\n\n booooom! generation complete.")
