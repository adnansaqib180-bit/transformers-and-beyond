import pandas as pd 
from keras.layers import *
import numpy as np
import tensorflow as tf
from keras import Sequential, Layer,Model
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
        result = self.norm2(x+fnn_output)
        return result

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

    def call(self, inputs,encoder_input, training=False):
        inputs = self.embeddings(inputs)
        inputs = self.pos_encoding(inputs)
        
        seq_len = tf.shape(inputs)
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        
        m_attention_output = self.masked_attention(
            key=encoder_input,
            query=encoder_input,
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

encoder = tiktoken.get_encoding("cl100k_base")

df['encoded_english'] = df['english'].apply(lambda x: encoder.encode(x))
df['encoded_urdu'] = df['roman_hindi'].apply(lambda x: encoder.encode(x))

max_length_english = max(df['encoded_english'].apply(len))
max_length_urdu = max(df['encoded_urdu'].apply(len))

PAD_ID = encoder.n_vocab
BOS_ID = encoder.n_vocab + 1
EOS_ID = encoder.encode('<|endoftext|>',allowed_special={'<|endoftext|>'})[0]

vocab_size = encoder.n_vocab + 2  
target_seq = [
    [BOS_ID] + tokens + [EOS_ID] for tokens in df['encoded_urdu']
]
encoder_input = [
    seq[:-1] for seq in df['encoded_english']
]
decoder_target = [
    seq[1:] for seq in target_seq
    
]
df['encoded_english'] = pad_sequences(encoder_input, maxlen=max_length_english, padding='post', truncating='post', value=PAD_ID).tolist()
df['decoder_input'] = pad_sequences(target_seq, maxlen=max_length_urdu + 2, padding='post', truncating='post', value=PAD_ID).tolist()
df['decoder_target'] = pad_sequences(decoder_target, maxlen=max_length_urdu + 2, padding='post', truncating='post', value=PAD_ID).tolist()
print(df.head())
encoder_input = np.array(df['encoded_english'])
decoder_input = np.array(df['decoder_input'])
decoder_target = np.array(df['decoder_target'])

print(df.columns)
df = df.drop(columns=['english', 'roman_hindi', 'encoded_urdu',])
print(df.head())
encoder_input = np.stack(df["encoded_english"].to_numpy())
decoder_input = np.stack(df["decoder_input"].to_numpy())
decoder_target = np.stack(df["decoder_target"].to_numpy())
print(encoder_input.shape, decoder_input.shape, decoder_target.shape)

inputs = Input(shape=(max_length_english,))
x = Encoder(sequence_length=max_length_english, vocabsize=vocab_size, embed_dim=128, num_heads=4, ff_dim=512)(inputs)
y = Decoder(vocabsize=vocab_size, sequence_length=max_length_urdu + 2, embed_dim=128, ff_dim=512, num_heads=4)(inputs=x, encoder_input=x)

model = Model(inputs=inputs, outputs=y)
print(model.summary())