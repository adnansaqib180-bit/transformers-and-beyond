import pandas as pd 
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

data = pd.read_csv('transformers-and-beyond/transformers/sentiment analyser/sentiment_analysis.csv')
print(data.head())

data = data.drop(columns=['Year', 'Month', 'Day', 'Time of Tweet', 'Platform'])

encoder = tiktoken.get_encoding("cl100k_base")
data['encoded_tweet'] = data['text'].apply(lambda x: encoder.encode(x))

max_length = max(data['encoded_tweet'].apply(len))
data['padded_tweet'] = pad_sequences(data['encoded_tweet'], maxlen=max_length, padding='post', truncating='post').tolist()

x_text = np.array(data['padded_tweet'].tolist())
y_labels = data['sentiment'].map({'positive':2, 'neutral': 1, 'negative': 0}).values

vocab_size = encoder.n_vocab

inputs = Input(shape=(max_length,))
x = Encoder(sequence_length=max_length, vocabsize=vocab_size, embed_dim=64, num_heads=2, ff_dim=128, dropout=0.5)(inputs)
x = GlobalAveragePooling1D()(x)
outputs = Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_text, y_labels, epochs=15, batch_size=32, validation_split=0.2)
text = 'I hate this product! It works perfectly and exceeded my expectations.'
model_input = encoder.encode(text)
model_input = pad_sequences([model_input], maxlen=max_length, padding='post', truncating='post')
model_input = np.array(model_input).reshape(1, -1)
prediction = model.predict(model_input)
predicted_class = np.argmax(prediction, axis=1)[0]
predicted_sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[predicted_class]
print(f"Predicted sentiment: {predicted_sentiment}")


# this model is predicting very well on clear sentences like "I hate this product! i just
# build it for my learning" it can be improved by training on more data and using a more complex model architecture.