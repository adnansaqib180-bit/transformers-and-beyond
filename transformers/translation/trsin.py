import pandas as pd 
from keras.layers import *
import numpy as np
import tensorflow as tf
from keras import Sequential, Layer
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


with open('transformers-and-beyond/transformers/translation/data.csv', 'r') as file:
    df = pd.read_csv(file)

print(df.head())

