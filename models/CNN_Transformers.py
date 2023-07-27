import tensorflow as tf
from tensorflow import keras


class SpeechFeatureEmbedding(keras.Model):
    def __init__(self, num_hid=64):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
    

class TransformerEncoder(keras.Model):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(feed_forward_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class CNN_T(keras.Model):
    def __init__(self, transformer_enc_num, num_class=3):
        super(CNN_T, self).__init__()
        
        self.cnn_enc = SpeechFeatureEmbedding()
        
        self.transformer_enc = keras.Sequential([TransformerEncoder(embed_dim=64, num_heads=2, feed_forward_dim=128) for _ in range(transformer_enc_num)])
        
        self.pool = keras.layers.GlobalMaxPool1D()
        
        self.fc = keras.layers.Dense(num_class, activation='softmax')
        
    def call(self, input):
        x = self.cnn_enc(input)
        
        x = self.transformer_enc(x)
        
        x = self.pool(x)
        
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    model = CNN_T(4)
    model(tf.random.uniform(shape=(1, 241, 185)))
    
    model.summary()