from keras.layers import *
from keras import Sequential
import numpy as np
import tensorflow as tf
import tiktoken
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
        return inputs + self.pos_enc

class Encoder(Layer):
    def __init__(self,sequence_length,vocabsize,embed_dim,num_heads,ff_dim,dropout=0.1):
        super().__init__()
        self.vocabsize = vocabsize
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads =num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim,sequence_length=self.sequence_length)
        self.embeddings = Embedding(input_dim=self.vocabsize,output_dim=embed_dim)
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim= self.embed_dim // self.num_heads
            
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.fnn = Sequential(
            layers=[Dense(self.ff_dim,activation='relu'),
            Dense(self.embed_dim)],
            trainable=True
        )
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)

    def call(self,inputs,training=False):
        inputs = self.embeddings(inputs)
        inputs = inputs + self.pos_encoding(inputs)
        attention_output = self.attention(
            key = inputs,
            value = inputs,
            query = inputs
        )
        attention_output = self.dropout1(attention_output,training=training)
        x = self.norm1(attention_output + inputs)
        fnn_output  = self.fnn(x)
        fnn_output = self.dropout2(fnn_output,training=training)

        return self.norm2(x+fnn_output)

class Decoder (Layer):
    def __init__(self,vocabsize,sequence_length,embed_dim,ff_dim,num_heads,dropout=0.1):
        super().__init__()
        self.vocabsize =  vocabsize 
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads =num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim,sequence_length=self.sequence_length)
        self.embeddings = Embedding(input_dim=self.vocabsize,output_dim=embed_dim)
        self.linear_layer = Dense(units=self.vocabsize, activation='softmax')
        self.masked_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim//num_heads,
            
        )   
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim= self.embed_dim // self.num_heads
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.fnn = Sequential([
            Dense(self.ff_dim,activation='relu'),
            Dense(self.embed_dim)]
        )
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)
        self.dropout3 = Dropout(self.dropout)
    def call(self,inputs,training=False):
        inputs = self.embeddings(inputs)
        inputs = inputs + self.pos_encoding(inputs)
        causal_mask = tf.linalg.band_part(tf.ones((self.sequence_length, self.sequence_length)), -1, 0)
        m_attention_output = self.masked_attention(
            key = inputs,
            query = inputs,
            value = inputs,
            attention_mask = causal_mask
        )
        m_attention_output = self.dropout1(m_attention_output,training=training)
        x1 = self.norm1(m_attention_output + inputs)
        attention_output = self.attention(
            key = x1,
            query = x1,
            value = x1
        )
        attention_output = self.dropout2(attention_output,training=training)
        x2 = self.norm2(attention_output + x1)
        fnn_output = self.fnn(x2)
        fnn_output = self.dropout3(fnn_output,training=training)

        x3 = self.norm3(fnn_output+x2)
        return self.linear_layer(x3)
maxlen = 150

with open('transformers-and-beyond/transformers/basic_projects/data.txt','r',encoding="utf-8") as file :
    data = file.read()

encoder = tiktoken.get_encoding('gpt2')
token = encoder.encode(data)

input_sequences = []
target_sequences = []
for i in range(0, len(token), maxlen):
    input_seq = token[i:i + maxlen]
    target_seq = token[i + 1:i + maxlen + 1]
    if len(input_seq) == maxlen and len(target_seq) == maxlen:
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)


padded_input_sequences = pad_sequences(input_sequences, maxlen=maxlen, padding='post')
target_sequences =np.array(target_sequences)

print(f"Number of input sequences: {len(padded_input_sequences)}")

BATCH_SIZE = 16

dataset = tf.data.Dataset.from_tensor_slices((padded_input_sequences, target_sequences))
dataset = dataset.shuffle(buffer_size=len(padded_input_sequences)).batch(BATCH_SIZE, drop_remainder=True)

x_train, y_train = padded_input_sequences, target_sequences

inputs = Input(shape=(maxlen,))
x = Decoder(vocabsize=encoder.n_vocab,
            sequence_length=maxlen, 
            embed_dim=128, ff_dim=512, 
            num_heads=8)(inputs)

model = tf.keras.Model(inputs=inputs, outputs=x)

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset, epochs=30, batch_size=BATCH_SIZE)

import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'], label='Training Loss')

import numpy as np
import tensorflow as tf

import numpy as np

# 1. Shuruati input aur setup
starting_text = "print"
generated_text = starting_text

print(f"Starting with: '{starting_text}'\n")
print("Generating next 10 words...")
print(starting_text, end="", flush=True)

for _ in range(10):
    current_tokens = encoder.encode(generated_text)
    if len(current_tokens) < maxlen:
        padded_tokens = [0] * (maxlen - len(current_tokens)) + current_tokens
    else:
        padded_tokens = current_tokens[-maxlen:]
        
    model_input = np.array([padded_tokens])
    predictions = model(model_input, training=False)
    next_token_id = np.argmax(predictions[0, -1, :])
    next_word = encoder.decode([next_token_id])
    generated_text += next_word
    print(next_word, end="", flush=True)

print("\n\n🚀 Boom! Generation Complete.")

