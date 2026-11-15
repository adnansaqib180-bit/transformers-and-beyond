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
        inputs = self.pos_encoding(inputs)
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
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.fnn = Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(self.embed_dim)
        ])
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)
        self.dropout3 = Dropout(self.dropout)

    def call(self, inputs,encoder_input, training=False):
        inputs = self.embeddings(inputs)
        inputs = self.pos_encoding(inputs)
        
        seq_len = tf.shape(inputs)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len),dtype=tf.int32), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        
        m_attention_output = self.masked_attention(
            key=inputs,
            query=inputs,
            value=inputs,
            attention_mask=causal_mask
        )
        m_attention_output = self.dropout1(m_attention_output, training=training)
        x1 = self.norm1(m_attention_output + inputs)
        cross_attention_output = self.cross_attention(
            key=encoder_input,
            query=inputs,
            value=encoder_input
        )
        cross_attention_output = self.dropout2(cross_attention_output, training=training)
        x2 = self.norm2(cross_attention_output + x1)
        fnn_output = self.fnn(x2)
        fnn_output = self.dropout3(fnn_output, training=training)
        x3 = self.norm3(fnn_output + x2)

        return self.linear_layer(x3)


with open('/kaggle/input/datasets/adnansaqib1212/english-to-hindi-translation/data.csv', 'r') as file:
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
df['decoder_input'] = pad_sequences(target_seq, maxlen=max_length_urdu + 1, padding='post', truncating='post', value=PAD_ID).tolist()
df['decoder_target'] = pad_sequences(decoder_target, maxlen=max_length_urdu + 1, padding='post', truncating='post', value=PAD_ID).tolist()
print(df.head())
X_encoder = np.array(df['encoded_english'])
X_decoder = np.array(df['decoder_input'])
decoder_target = np.array(df['decoder_target'])

print(df.columns)
df = df.drop(columns=['english', 'roman_hindi', 'encoded_urdu',])
print(df.head())
X_encoder = np.stack(df["encoded_english"].to_numpy())
X_decoder = np.stack(df["decoder_input"].to_numpy())
decoder_target = np.stack(df["decoder_target"].to_numpy())
print(X_encoder.shape, X_decoder.shape, decoder_target.shape)

encoder_input = Input(shape=(max_length_english,),dtype = 'int32')
decoder_input = Input(shape=(max_length_urdu + 1,),dtype = 'int32')

encoder_output = Encoder(sequence_length=max_length_english, vocabsize=vocab_size, embed_dim=256, num_heads=8, ff_dim=1024)(encoder_input)

decoder_output = Decoder(vocabsize=vocab_size, sequence_length=max_length_urdu + 1, embed_dim=256, ff_dim=1024, num_heads=8)(decoder_input, encoder_output)

model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
sample_weights = (decoder_target != PAD_ID).astype("float32")
model.fit([X_encoder, X_decoder], decoder_target, batch_size=32, epochs=50, validation_split=0.2, sample_weight=sample_weights)

test_sentence = "How are you?"

# 1. Encode the source text
tokens = encoder.encode(test_sentence)
encoder_input = pad_sequences(
    [tokens],
    maxlen=max_length_english,
    padding="post",
    truncating="post",
    value=PAD_ID
).astype("int32")

decoder_input_seq = np.full((1, max_length_urdu + 1), PAD_ID, dtype="int32")
decoder_input_seq[0, 0] = BOS_ID  

generated_tokens = []

for i in range(max_length_urdu):
    prediction = model.predict([encoder_input, decoder_input_seq], verbose=0)
    next_token_logits = prediction[0, i, :]
    next_token_id = int(np.argmax(next_token_logits))
    if next_token_id == EOS_ID or next_token_id == PAD_ID:
        break
        
    generated_tokens.append(next_token_id)
    if i + 1 < (max_length_urdu + 1):
        decoder_input_seq[0, i + 1] = next_token_id

clean_tokens = [t for t in generated_tokens if t < encoder.n_vocab]
decoded_output = encoder.decode(clean_tokens)


print('Input Sentence:', test_sentence)
print('Predicted Roman Hindi:', decoded_output)
