# %%
import re
import os
import string
import numpy as np
import pandas as pd

from tensorflow.python.keras import backend as K
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet import ResNet152

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Concatenate, Embedding, LSTM, Dropout, Bidirectional, Layer, InputSpec, Dot, Activation, Multiply

# %%
from sentence_transformers import SentenceTransformer, util
from PIL import Image

sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# %%
dataset = pd.read_excel("./Aspect_Complain_web_entity_clip_new2.xlsx")
#dataset.drop(columns = ['Complaint Overall', 'Emotion Overall', 'Sentiment Overall','Emotion.0', 'Emotion.1', 'Emotion.2', 'Emotion.3', 'Comments'], inplace = True)
#dataset = dataset.iloc[0:1096,:]
dataset.rename(columns = {' ': 'Domain'}, inplace = True)
dataset = dataset.drop('Unnamed: 0',axis = 1)
# dataset = dataset.drop('Unnamed: 0.1',axis = 1)
dataset.columns

# %%
dataset

# %%
import pandas as pd


df_gemini = pd.read_csv('./output1.csv')
df_blip = pd.read_csv('./blip.csv')

# Keep only necessary columns ('Image Path' and 'Caption')
df_gemini = df_gemini[['Image Path', 'Caption']]
df_blip = df_blip[['Image Path', 'Caption']]

# Typecast the df_gemini['Caption'] to string
df_gemini['Caption'] = df_gemini['Caption'].astype(str)
df_blip['Caption'] = df_blip['Caption'].astype(str)

# Typecast other columns if needed
df_gemini['Image Path'] = df_gemini['Image Path'].astype(str)
df_blip['Image Path'] = df_blip['Image Path'].astype(str)

# Remove the [ ] from the df_gemini['Caption']
df_gemini['Caption'] = df_gemini['Caption'].str.replace("['", '')
df_gemini['Caption'] = df_gemini['Caption'].str.replace("']", '')
df_blip['Caption'] = df_blip['Caption'].str.replace("['", '')
df_blip['Caption'] = df_blip['Caption'].str.replace("']", '')

# Replace the /DATA/toshit_2101ai39/output in df_gemini['Image Path'] with .
df_gemini['Image Path'] = df_gemini['Image Path'].str.replace("/DATA/toshit_2101ai39/output", "./output")

# Merge dataframes on the common column ('Image Link' and 'Image Path')
merged_df = pd.merge(dataset, df_gemini, left_on='Image Link', right_on='Image Path', how='left')
merged_df_final = pd.merge(merged_df, df_blip, left_on='Image Link', right_on='Image Path', how='left')



# If Image Path_x in merged_df_final is null, then replace it with Image Path_y
merged_df_final['Image Path_x'] = merged_df_final['Image Path_x'].fillna(merged_df_final['Image Path_y'])
# If Caption_x in merged_df_final is null, then replace it with Caption_y
merged_df_final['Caption_x'] = merged_df_final['Caption_x'].fillna(merged_df_final['Caption_y'])

# drop Image Path_y and Caption_y
merged_df_final.drop(['Image Path_y', 'Caption_y'], axis=1, inplace=True)

# Rename Image Path_x and Caption_x to Image Path and Caption
merged_df_final.rename(columns={'Image Path_x': 'Image Path', 'Caption_x': 'Caption'}, inplace=True)

# Create a new column 'review' based on conditions
merged_df_final['review'] = merged_df_final.apply(
    lambda row: f"Review: {row['Complaint/ Opinion']} Image Info: {row['Caption']}" if pd.notnull(row['Caption']) else '',
    axis=1
)

# If you want to update the 'Complaint/ Opinion' column in the original df:
dataset['Complaint/ Opinion'] = merged_df_final['review']
# merged_df_final.head()

# %%
images_path = dataset['Image Link']

# %% [markdown]
# ## Getting data into lists

# %%
dataset.iloc[2,0]

# %%
dataset.info()

# %%
dataset['Domain'] = dataset['Domain'].astype(str)
dataset['Complaint/ Opinion'] = dataset['Complaint/ Opinion'].astype(str)

# %%
def get_title_review_comb(X_reviews):
    title_review_comb = []
    for i in range(X_reviews.shape[0]):
        if(str(type(X_reviews.iloc[i,0])) == "<class 'str'>" and str(type(X_reviews.iloc[i,1])) == "<class 'str'>"):
          title_review_comb.append(str(X_reviews.iloc[i,0])+" "+str(X_reviews.iloc[i,1]))
        else:
          # print(i)
          title_review_comb.append(X_reviews.iloc[i,0])

    text_reviews = []
    for i in range(X_reviews.shape[0]):
        words = re.split(r'\W+', title_review_comb[i])

        words = [word.lower() for word in words]
        text = ' '.join(words)
        words = text.split()
        text = ' '.join(words)
        text_reviews.append(text)

    return text_reviews

# def check_aspect_comp(aspect,complaint_label,j):
#   aspects = []
#   comp_labels = []
#   for i in [0,1,2]:
#     val1 = aspect[i][j]
#     val2 = complaint_label[i][j]
#     if(str(val1) == 'nan') or val1 == ' ':
#       continue
#     elif(val1 == 'fabric'):
#       aspects.append('quality')
#     else:
#       aspects.append(val1)
#     comp_labels.append(val2)

#   return aspects,comp_labels

def check_aspect_comp(aspect,complaint_label,j):
  asp = []
  comp_labels = []
  for i in [0,1,2]:
    val = aspect[i][j]
    val2 = complaint_label[i][j]
    if(str(val) != 'nan'):
      val = str(val).lower().strip()
    if(str(val) == 'nan'):
      continue
    elif val in ['freedback','feeedback','feedbck','feedaback','fedback','feedabck','feedback','personal opinion','personal opinion,','general opinion','advise','genera review','general review']:
      asp.append('review')
    elif val in ['miscelllaneous information','miscellaneous inforamation','miscellaneous inforation','misceallneous information','miscellaneous information, inflation','misellaneous infromation','miscelleneos information','miscellaneous information','miscellaneous','miscellameous information','miscellaneus information','misellaneous information','miscelaneous information','misceklaneous information','miscellaneous iformation','misceallaneous information','financial advertisement','financial advertiement','financial advertisemnt']:
      asp.append('misc_info')
    elif val in ['delay, response','delay resoponse','delay respose','delay response','dealy response','delay resonse','delay response,','delay reponse','apologise']:
      asp.append('provider_response')
    elif val in ['payment','paymet','payent issue','payment issue','payment issue, bank issue','payment failure','payment failure, miscellaneous information','banking','bankinhg','bank issue','net banking','bank issue, harassment','internet banking','refund issue']:
      asp.append('net_banking_issue')
    # elif val in ['banking','bankinhg','bank issue','net banking','bank issue, harassment','internet banking']:
    #   asp.append('banking_issue')
    elif val in ['haraasment','harrasment','harassment','harrashment','fruad','fraud','manipulation']:
      asp.append('consumer_safety')
    # elif val in ['fruad','fraud']:
    #   asp.append('fraud')
    elif val in ['trade balance','news','dept collection','debt collection','exports']:
      asp.append('financial_info')
    # elif val in ['inflation']:
    #   asp.append('inflation')
    elif val in ['crdentials','credentials','credentials issue','credential','cedentials','credential related','technicality','techinicality','technical issue']:
      asp.append('credential_error')
    elif val in ['finacial policy','financial policy','financial policies','financial poilicy','financial stability','financial gain','inflation']:
      asp.append('financial_situation')
    # elif val in ['apologise']:
    #   asp.append('apologise')
    # elif val in ['dept collection','debt collection']:
    #   asp.append('debt_collection')
    # elif val in ['equal taxation']:
    #   asp.append('equal_taxation')
    elif val in ['genearl query','general qurey','general query','genral query','genaral query','general qurery','tax increases','equal taxation']:
      asp.append('general_query')
    # elif val in ['personal opinion','personal opinion,','general opinion','advise','genera review','general review']:
    #   asp.append('personal_opinion')
    # elif val in ['financial stability']:
    #   asp.append('financial_stability')
    # elif val in ['financial gain']:
    #   asp.append('financial_gain')
    # elif val in ['news']:
    #   asp.append('news')
    # elif val in ['manipulation']:
    #   asp.append('manipulation')
    # elif val in ['financial advertisement','financial advertiement','financial advertisemnt']:
    #   asp.append('financial_advertisement')
    # elif val in ['exports']:
    #   asp.append('exports')
    # elif val in ['tax increases','equal taxation']:
    #   asp.append('tax_implication')
    # elif val in ['technicality','techinicality','technical issue']:
    #   asp.append('technical_issue')
    # elif val in ['debt collection']:
    #   asp.append('debt collection')
    # elif val in ['refund issue']:
    #   asp.append('refund_issue')
    # else:
    #   print(val)
    else:
      asp.append('no_aspects')
    comp_labels.append(val2)
  return asp,comp_labels


def get_labels(df):
    aspect = [list(df['Aspects_1']), list(df['Aspect_2']), list(df['Aspect_3'])]
    complaint_label = [list(df['Complaint_1']), list(df['Complaint_2']), list(df['Complaint_3'])]
    comp_labels = []
    aspects = []

    for i in range(df.shape[0]):
        out = check_aspect_comp(aspect,complaint_label,i)
        aspects.append(out[0])
        comp_labels.append(out[1])

    return aspects, comp_labels

def get_web_entities(dataset):
    web_entities = []
    for mul_entities in list(dataset['Web_entities']):
      lables_list = (re.sub("[^a-zA-Z0-9,.)]", " ", mul_entities)).split(',')
      # print(lables_list)
      lables_list_mod = []
      for label in lables_list:
        label_mod = " ".join(label.split())
        # print(label_mod)
        if(label_mod == ''):
          continue
        lables_list_mod.append(label_mod.lower())

      web_entities.append(lables_list_mod)

    return web_entities


text_review = get_title_review_comb(dataset)
#web_entities = get_web_entities(dataset)
aspect_labels, complaint_labels = get_labels(dataset)

print(len(aspect_labels))
print(len(complaint_labels))
print(len(text_review))
#print(len(web_entities))
print(len(images_path))

# %%
for i in range(len(complaint_labels)):
  for j in range(len(complaint_labels[i])):
    if(str(complaint_labels[i][j]) == 'nan'):
      complaint_labels[i][j] = 0.0

# %%
complaint_labels[3][1]

# %%
for item in complaint_labels:
    print(item)

# %%
# for each of the aspect_labels find the number of complaints and non complaint

comp_dict = {}
for i in range(len(aspect_labels)):
    for j in range(len(aspect_labels[i])):
        if aspect_labels[i][j] not in comp_dict:
            comp_dict[aspect_labels[i][j]] = complaint_labels[i][j]
        comp_dict[aspect_labels[i][j]] += complaint_labels[i][j]
print(comp_dict)

# %%
review_aspect_merged = []
images_path_new = []
comp_labels = []
# for r,a,c,i in zip(text_review, aspect_labels, complaint_labels, images_path):
#     for aspect,complaint in zip(a,c):
#         r_a = aspect+' [SEP] '+r
#         review_aspect_merged.append(r_a)
#         comp_labels.append(complaint)
#         images_path_new.append(i)
merged_aspect = []
for item in aspect_labels:
  temp = ""
  for j in item:
    temp = temp + " " + j
  merged_aspect.append(temp)
for i in range(len(text_review)):
    review_aspect_merged.append(text_review[i])
    comp_labels.append(aspect_labels[i])
    images_path_new.append(images_path[i])

print(len(review_aspect_merged))
print(len(images_path_new))
print(len(comp_labels))

# %%
from collections import Counter
def get_labels_and_frequencies(aspects):
    label_freqs = Counter()
    data_labels = aspects
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs

aspect_label, aspect_label_freqs = get_labels_and_frequencies(aspect_labels)
# aspect_label_1, aspect_label_freqs_1 = get_labels_and_frequencies(aspect_label_new)
# aspect_label, aspect_label_freqs = get_labels_and_frequencies(aspect_label_final)
print(aspect_label_freqs)
# print(aspect_label_freqs_1)
# print(aspect_label_freqs_2)



# %%
# Delete all the text reviews from review_text_merged which has either no_aspect or financial_situation as aspect
# Delete the corresponding image path and complaint label
review_aspect_merged_new = []
images_path_new_new = []
comp_labels_new = []
aspect_label_new = []
for r,a,c,i in zip(review_aspect_merged, aspect_labels, complaint_labels, images_path_new):
    if('no_aspects' not in a and 'financial_situation' not in a):
        review_aspect_merged_new.append(r)
        comp_labels_new.append(c)
        images_path_new_new.append(i)
        aspect_label_new.append(a)

# %%
# Repeat the sample for provider_response 2 times and net_banking_issue 3 times and change the complaint label, image_path_new_new, aspect_label_new accordingly
review_aspect_final = []
images_path_final = []
comp_labels_final = []
aspect_label_final = []
for r,a,c,i in zip(review_aspect_merged_new, aspect_label_new, comp_labels_new, images_path_new_new):
    if('provider_response' in a):
        for j in range(2):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    elif('consumer_safety' in a):
        for j in range(6):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    elif('credential_error' in a):
        for j in range(6):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    elif('general_query' in a):
        for j in range(7):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    elif('financial_info' in a):
        for j in range(8):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    else:
        review_aspect_final.append(r)
        comp_labels_final.append(c)
        images_path_final.append(i)
        aspect_label_final.append(a)

# %%
review_aspect_merged = review_aspect_final
images_path_new = images_path_final
comp_labels = comp_labels_final
aspect_label = aspect_label_final

# %%
complaint_labels = comp_labels_final

# %% [markdown]
# ## Encoding text and images

# %%
sentence_model.max_seq_length = 512
review_embeddings = []
for review in review_aspect_merged:
  text_emb = sentence_model.encode(review)
  review_embeddings.append(text_emb)

# %%
def image_preprocessing(path):
    img = keras_image.load_img(path, target_size=(224, 224), interpolation='bicubic')
    img = keras_image.img_to_array(img)
    img = img//255.0
    return img

images = []
for img_path in images_path_new:
  img = image_preprocessing(img_path)
  images.append(img)

images_list_new = []
for a,i in zip(aspect_labels, images):
    for aspect in a:
        images_list_new.append(i)

# %%
# Count the distinct labels in the aspect labels
aspect_label, aspect_label_freqs = get_labels_and_frequencies(aspect_label_final)

# %%
aspect_label


# %%
# Count number of distinct aspect labels
aspect_labels_count = [item for sublist in aspect_label_final for item in sublist]
aspect_labels_count = list(set(aspect_labels_count))
print((aspect_labels_count))



# %%
from collections import Counter
def get_labels_and_frequencies(aspects):
    label_freqs = Counter()
    data_labels = aspects
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs

aspect_label, aspect_label_freqs = get_labels_and_frequencies(aspect_label_final)
print(aspect_label_freqs)



# %%
# plot the aspect_count in a vertical bar graph
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
plt.bar(range(len(aspect_label_freqs)), list(aspect_label_freqs.values()), align='center',color='seagreen')
plt.xticks(range(len(aspect_label_freqs)), list(aspect_label_freqs.keys()))
plt.title('Aspect Count')
# Format the plt figure with sea green color
# plt.rcParams['figure.facecolor'] = 'xkcd:sea green'
# Make the bars in pink
plt.show()





# %%

# %%
aspect_label_final[0]

# %%
len(aspect_label_final)

# %%
aspect_labels_count
# 

# %%
# Create a list of list as the size of complaint_labels and one list in it should be of size 10 in which 1 is present at the index of the complaint label
complaint_labels_count = []
for i in range(len(complaint_labels)):
  temp = [0]*8
  for j in range(8):
    if(aspect_label_final[i][0] == aspect_labels_count[j]):
      temp[j] = 1
    if(len(aspect_label_final[i]) > 1):
      if(aspect_label_final[i][1] == aspect_labels_count[j]):
        temp[j] = 1
    if(len(aspect_label_final[i]) > 2):
      if(aspect_label_final[i][2] == aspect_labels_count[j]):
        temp[j] = 1
  complaint_labels_count.append(temp)


# %%
import tensorflow as tf
# # comp_labels = list(pd.Series(comp_labels).fillna(0.))
# # complaint_labels_reshaped = tf.keras.utils.to_categorical(comp_labels)
# review_embeddings = np.array(review_embeddings).reshape(-1, 1, 768)

# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# max_length = max(len(seq) for seq in review_embeddings)

# padded_review_embeddings = pad_sequences(review_embeddings, maxlen=max_length, padding='post', truncating='post', dtype='float32')

review_embeddings = np.array(review_embeddings).reshape(-1, 1, 768)
img_converted = tf.convert_to_tensor([tf.convert_to_tensor(i, dtype=tf.float32) for i in images], dtype = tf.float32)

# %%
img_converted.shape

# %%
complaint_labels_0 = [item[0] for item in complaint_labels]

# %%
RANDOM_SEED = 42
# First Split for Train and Test
img_train,img_test, x_train,x_test, y_train,y_test, comp_label_train,comp_label_test = train_test_split( np.array(img_converted),
                                                              np.array(review_embeddings), np.array(complaint_labels_count), np.array(complaint_labels_0),
                                                              test_size=0.1, random_state=RANDOM_SEED,shuffle=True)

# Next split Train in to training and validation
img_tr,img_val, x_tr,x_val, y_tr,y_val, comp_label_tr,comp_label_val = train_test_split(img_train, x_train, y_train,comp_label_train, test_size=0.2,
                                                          random_state = RANDOM_SEED,shuffle=True)

print(img_tr.shape)
print(img_test.shape)
print(img_val.shape)
print(x_tr.shape)
print(x_test.shape)
print(x_val.shape)
print(y_tr.shape)
print(y_test.shape)
print(y_val.shape)

# %% [markdown]
# ## Modeling

# %%
class AttLayer(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttLayer, self).__init__(** kwargs)

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
        }
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self._trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

# %%
import tensorflow as tf
import keras
from keras import layers

# %%


# %%
resnet152 = ResNet152(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
for layer in resnet152.layers:
    layer.trainable = False
input_1 = Input(shape=(1, 768), dtype='float32')
image_input = Input(shape=(224, 224, 3), name='Images_input')
resnet152_output = resnet152(image_input)
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(resnet152_output)
img_repr = (tf.reshape(pooling_layer, [-1, 1, 2048]))
# Add a dense layer to the pooled feature representation to make it the same shape as input_1
img_repr = Dense(768, activation = 'relu')(img_repr)

# Concatenate text and image features
concatenated_features = layers.Concatenate(axis=1)([input_1, img_repr])
# concatenated_features = input_1
# concatenated_features = img_repr
concatenated_features = Dense(768, activation = 'relu')(concatenated_features)

# VAE Encoder
encoder_inputs = concatenated_features
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(128, name="z_mean")(x)
z_log_var = layers.Dense(128, name="z_log_var")(x)

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

# VAE Decoder
decoder_inputs = z
x = layers.Dense(256, activation="relu")(decoder_inputs)
x = layers.Reshape((1, 256))(x)

input_1_dense = layers.Dense(128, name="input_1_dense")(input_1)
# Concatenate with img_repr for further processing
merged_features = layers.Concatenate(axis=-1)([input_1_dense, x])

# Linear layer
linear_layer = layers.Dense(768)(merged_features)


Bi_lstm = Bidirectional(tf.keras.layers.GRU(256, return_sequences = True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer = l2(0.000001)))(concatenated_features)
Bi_lstm = AttLayer()(Bi_lstm)
out_aspects = Dropout(0.3)(Bi_lstm)
out_aspects = Dense(128, activation = 'relu')(out_aspects)
out_aspects = Dropout(0.2)(out_aspects)
output = Dense(8, activation = 'sigmoid')(out_aspects)

MM_model = Model([input_1, image_input], output)
MM_model.summary()

# %%
from keras.layers import Input, Dense, Flatten, Concatenate, Attention

# %%
# %pip install tensorflow-addons

# %%
# import tensorflow_addons as tfa
# from tensorflow import keras

# %%
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# %%
def asymmetric_loss(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    def loss(y_true, y_pred):
        # Calculating Probabilities
        y_true = tf.cast(y_true, tf.float32)
        y_pred_sigmoid = tf.sigmoid(y_pred)
        y_pred_pos = y_pred_sigmoid
        y_pred_neg = 1 - y_pred_sigmoid

        # Asymmetric Clipping
        if clip is not None and clip > 0:
            y_pred_neg = tf.clip_by_value(y_pred_neg + clip, clip_value_min=0.0, clip_value_max=1.0)

        # Basic CE calculation
        los_pos = y_true * tf.math.log(tf.clip_by_value(y_pred_pos, eps, 1.0))
        los_neg = (1 - y_true) * tf.math.log(tf.clip_by_value(y_pred_neg, eps, 1.0))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if gamma_neg > 0 or gamma_pos > 0:
            pt0 = y_pred_pos * y_true
            pt1 = y_pred_neg * (1 - y_true)
            pt = pt0 + pt1
            one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
            one_sided_w = tf.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -tf.reduce_sum(loss)

    return loss

# %%
m = 'accuracy'
MM_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss = 'binary_crossentropy',
                  loss_weights = 0.5,
                  metrics = m)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=13, verbose=1, restore_best_weights=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./save_dir",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

MM_model.fit(
    [x_tr, img_tr], y_tr,
    batch_size=16,
    epochs=50,
    validation_data = ([x_val, img_val], y_val),
    steps_per_epoch = x_tr.shape[0] // 16,
    validation_steps = x_val.shape[0] // 16,
    shuffle=True,
    verbose=1,
    callbacks=[earlystopping, model_checkpoint_callback]
)

# %%
MM_model.save('multi_aspect_detection_text_image_with_blip.h5')

# %%
# MM_model = load_model('multi_aspect_detection_text_image.h5', custom_objects={'AttLayer': AttLayer})
# MM_model = load_model('./save_dir')

# %%
complaint_labels_predicted = MM_model.predict([x_test, img_test])
complaint_labels_pred = np.argmax(complaint_labels_predicted, axis = -1)
complaint_labels_predicted

# %%
threshold = 0.5
y_pred = [np.where(prob >= threshold, 1, 0) for prob in complaint_labels_predicted]

# %%
y_test

# %%
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# %%
aspect_labels_count[1] = "No Aspect"

# %%
print(classification_report(y_test, y_pred, target_names=aspect_labels_count))

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

mcm = multilabel_confusion_matrix(y_test, y_pred)


# %%
fig, axes = plt.subplots(1, len(mcm), figsize=(20, 2))
for i, (label, cm) in enumerate(zip(aspect_labels_count, mcm)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
    # axes[i].set_xlabel('Predicted')
    # axes[i].set_ylabel('True')
    axes[i].set_title(f'{label}')
# plt.tight_layout()
plt.show()

# %%
ax= plt.subplot()

cm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation


# %%
domain_list = list(dataset['Domain'])
(domain_list)

# %%
x_test_domain = []
img_test_domain = []
y_test_domain = []
for i in range(len(domain_list)):
    if domain_list[i] == 'Transactional Deficiency':
        # print("adding")
        x_test_domain.append(review_embeddings[i])
        img_test_domain.append(img_converted[i])
        y_test_domain.append(complaint_labels_count[i])
complaint_labels_predicted_domain = MM_model.predict([np.array(x_test_domain), np.array(img_test_domain)])
y_pred_domain = [np.where(prob >= threshold, 1, 0) for prob in complaint_labels_predicted_domain]
accuracy = accuracy_score(np.array(y_test_domain), np.array(y_pred_domain))
print(accuracy)
print(classification_report(np.array(y_test_domain), np.array(y_pred_domain), target_names=aspect_labels_count))
        
        

# %%
domain = []
for i in dataset['Domain']:
    domain.append(i.strip().lower())
domain = list(domain)
from collections import Counter
domain_counter = Counter(domain)
print(domain_counter)

# %%
# Code for ignoring warning
import warnings
warnings.filterwarnings("ignore")

# %%
for domain in domain_count.keys():
    x_test_domain = []
    img_test_domain = []
    y_test_domain = []
    for i in range(len(domain_list)):
        if domain_list[i].lower().strip() == domain:
            # print("adding")
            x_test_domain.append(review_embeddings[i])
            img_test_domain.append(img_converted[i])
            y_test_domain.append(complaint_labels_count[i])
    complaint_labels_predicted_domain = MM_model.predict([np.array(x_test_domain), np.array(img_test_domain)])
    y_pred_domain = [np.where(prob >= threshold, 1, 0) for prob in complaint_labels_predicted_domain]
    accuracy = accuracy_score(np.array(y_test_domain), np.array(y_pred_domain))
    print(domain)
    print(accuracy)
    print(classification_report(np.array(y_test_domain), np.array(y_pred_domain), target_names=aspect_labels_count))
        
        

# %%
complaint_labels_predicted = MM_model.predict([np.array(review_embeddings), np.array(img_converted)])
complaint_labels_pred = np.argmax(complaint_labels_predicted, axis = -1)
complaint_labels_predicted

# %%
threshold = 0.5
y_pred = [np.where(prob >= threshold, 1, 0) for prob in complaint_labels_predicted]

# %%
accuracy = accuracy_score(np.array(complaint_labels_count), y_pred)
print(accuracy)

# %%
print(classification_report(np.array(complaint_labels_count), y_pred, target_names=aspect_labels_count))

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
mcm = multilabel_confusion_matrix(np.array(complaint_labels_count), y_pred)

# %%
fig, axes = plt.subplots(1, len(mcm), figsize=(20, 2))
for i, (label, cm) in enumerate(zip(aspect_labels_count, mcm)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
    # axes[i].set_xlabel('Predicted')
    # axes[i].set_ylabel('True')
    axes[i].set_title(f'{label}')
# plt.tight_layout()
plt.show()

# %%
ax= plt.subplot()

cm = confusion_matrix(np.asarray(np.array(complaint_labels_count)).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation



