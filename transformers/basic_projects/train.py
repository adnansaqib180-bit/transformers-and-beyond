from keras.layers import *
from keras import Sequential
import numpy as np
import tensorflow as tf
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
    def __init__(self,embed_dim,num_heads,ff_dim,dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim= embed_dim // num_heads
            
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.fnn = Sequential(
            layers=[Dense(ff_dim,activation='relu'),
            Dense(embed_dim)],
            trainable=True
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self,inputs,training=False):
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
    def __init__(self,embed_dim,ff_dim,num_heads,dropout=0.1):
        super().__init__()
        self.embeddings = Embedding(input_dim=4,output_dim=embed_dim)
        self.masked_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim//num_heads,
        )   
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim= embed_dim // num_heads
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.fnn = Sequential(
            Dense(ff_dim,activation='relu'),
            Dense(embed_dim)
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
    def softmax(num):
        n = np.exp(num - np.max(num, axis=-1, keepdims=True))
        result = n / np.sum(n, axis=-1, keepdims=True)
        return result
    def call(self,inputs,training=False):
        inputs = self.embeddings(inputs)
        m_attention_output = self.masked_attention(
            key = inputs,
            query = inputs,
            value = inputs
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

        x3 = self.norm3(fnn_output,training=training)
        return self.softmax(x3)