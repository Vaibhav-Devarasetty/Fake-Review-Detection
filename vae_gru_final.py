
# %%
import re
import os
import string
import numpy as np
import pandas as pd


# from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# %%
from sentence_transformers import SentenceTransformer, util
from PIL import Image


sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# %%
dataset = pd.read_csv("./Aspect_Complain_web_entity_clip.csv")
#dataset.drop(columns = ['Complaint Overall', 'Emotion Overall', 'Sentiment Overall','Emotion.0', 'Emotion.1', 'Emotion.2', 'Emotion.3', 'Comments'], inplace = True)
#dataset = dataset.iloc[0:1096,:]

dataset.head()

data = pd.read_csv("local_icdar.csv")
data = data.dropna(subset=['Image', 'Aspect_1'])
data = data.reset_index()

data.head(10)

dataset = data

#dataset.rename(columns = {' ': 'Domain'}, inplace = True)
dataset = dataset.drop('Unnamed: 0',axis = 1)
# dataset = dataset.drop('Unnamed: 0.1',axis = 1)
dataset.columns

# %%
dataset.head(10)

dataset.keys()

unique_count = dataset['Domain'].unique()
print(unique_count)


# %%
images_path = dataset['Image']

images_path[8]

# %% [markdown]
# ## Getting data into lists

# %%
dataset.iloc[2,0]

# %%
dataset.info()

dataset.keys()

# %%
dataset['Domain'] = dataset['Domain'].astype(str)
dataset['Review Content'] = dataset['Review Content'].astype(str)

dataset.head()

import pandas as pd
from collections import Counter

# Assuming df is your pandas DataFrame
# For example, let's say you have columns 'column1' and 'column2'
# You can replace them with the actual column names in your DataFrame

df = dataset

# Select the columns you are interested in
selected_columns = ['Aspect_1', 'Aspect_2', 'Aspect_3', 'Aspect_4', 'Aspect_5']

# Concatenate the values of selected columns into a single Series
concatenated_series = pd.concat([df[col] for col in selected_columns])

# Create a Counter object
value_counter = Counter(concatenated_series)

sorted_counter = dict(sorted(value_counter.items(), key=lambda x: x[1], reverse=True))


# Print the count of each unique value
for value, count in sorted_counter.items():
    print(f"{value}: {count}")

dataset.keys()

num_of_labels = 9

# %%
def get_title_review_comb(X_reviews):
    title_review_comb = []
    for i in range(X_reviews.shape[0]):
        if(str(type(X_reviews.iloc[i,3])) == "<class 'str'>" and str(type(X_reviews.iloc[i,5])) == "<class 'str'>"):
          title_review_comb.append(str(X_reviews.iloc[i,3])+" "+str(X_reviews.iloc[i,5]))
        else:
          # print(i)
          title_review_comb.append(X_reviews.iloc[i,3])

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

    elif val in ['battery', 'charging']:
      asp.append('Energy_Reservior')

    elif val in ['camera']:
      asp.append('Camera')

    elif val in ['color']:
      asp.append('Others')

    elif val in ['customer_service']:
      asp.append('Customer_Service')

    elif val in ['cost']:
      asp.append('Price')

    elif val in ['product_quality', 'product_quantity']:
      asp.append('Product_Quality')

    elif val in ['user experience', 'feedback', 'review']:
      asp.append('Feedback')

    elif val in ['security', 'consumer_safety']:
      asp.append('Others')

    elif val in ['display']:
      asp.append('Display')

    elif val in ['hardware', 'software']:
      asp.append('Others')

    elif val in ['media', 'function', 'speaker', 'user_interface', 'size', 'noise', 's_pen', 'texture']:
      asp.append('Functionality')

    elif val in ['packaging', 'look', 'texture']:
      asp.append('Others')

    elif val in ['service']:
      asp.append('Others')

    elif val in ['weight', 'storage']:
      asp.append('Others')


    # elif val in ['freedback','feeedback','feedbck','feedaback','fedback','feedabck','feedback','personal opinion','personal opinion,','general opinion','advise','genera review','general review']:
    #   asp.append('review')
    # elif val in ['miscelllaneous information','miscellaneous inforamation','miscellaneous inforation','misceallneous information','miscellaneous information, inflation','misellaneous infromation','miscelleneos information','miscellaneous information','miscellaneous','miscellameous information','miscellaneus information','misellaneous information','miscelaneous information','misceklaneous information','miscellaneous iformation','misceallaneous information','financial advertisement','financial advertiement','financial advertisemnt']:
    #   asp.append('misc_info')
    # elif val in ['delay, response','delay resoponse','delay respose','delay response','dealy response','delay resonse','delay response,','delay reponse','apologise']:
    #   asp.append('provider_response')
    # elif val in ['payment','paymet','payent issue','payment issue','payment issue, bank issue','payment failure','payment failure, miscellaneous information','banking','bankinhg','bank issue','net banking','bank issue, harassment','internet banking','refund issue']:
    #   asp.append('net_banking_issue')
    # # elif val in ['banking','bankinhg','bank issue','net banking','bank issue, harassment','internet banking']:
    # #   asp.append('banking_issue')
    # elif val in ['haraasment','harrasment','harassment','harrashment','fruad','fraud','manipulation']:
    #   asp.append('consumer_safety')
    # # elif val in ['fruad','fraud']:
    # #   asp.append('fraud')
    # elif val in ['trade balance','news','dept collection','debt collection','exports']:
    #   asp.append('financial_info')
    # # elif val in ['inflation']:
    # #   asp.append('inflation')
    # elif val in ['crdentials','credentials','credentials issue','credential','cedentials','credential related','technicality','techinicality','technical issue']:
    #   asp.append('credential_error')
    # elif val in ['finacial policy','financial policy','financial policies','financial poilicy','financial stability','financial gain','inflation']:
    #   asp.append('financial_situation')
    # # elif val in ['apologise']:
    # #   asp.append('apologise')
    # # elif val in ['dept collection','debt collection']:
    # #   asp.append('debt_collection')
    # # elif val in ['equal taxation']:
    # #   asp.append('equal_taxation')
    # elif val in ['genearl query','general qurey','general query','genral query','genaral query','general qurery','tax increases','equal taxation']:
    #   asp.append('general_query')
    # # elif val in ['personal opinion','personal opinion,','general opinion','advise','genera review','general review']:
    # #   asp.append('personal_opinion')
    # # elif val in ['financial stability']:
    # #   asp.append('financial_stability')
    # # elif val in ['financial gain']:
    # #   asp.append('financial_gain')
    # # elif val in ['news']:
    # #   asp.append('news')
    # # elif val in ['manipulation']:
    # #   asp.append('manipulation')
    # # elif val in ['financial advertisement','financial advertiement','financial advertisemnt']:
    # #   asp.append('financial_advertisement')
    # # elif val in ['exports']:
    # #   asp.append('exports')
    # # elif val in ['tax increases','equal taxation']:
    # #   asp.append('tax_implication')
    # # elif val in ['technicality','techinicality','technical issue']:
    # #   asp.append('technical_issue')
    # # elif val in ['debt collection']:
    # #   asp.append('debt collection')
    # # elif val in ['refund issue']:
    # #   asp.append('refund_issue')
    # # else:
    # #   print(val)
    else:
      asp.append('No_Aspects')
    comp_labels.append(val2)
  return asp,comp_labels


def get_labels(df):
    aspect = [list(df['Aspect_1']), list(df['Aspect_2']), list(df['Aspect_3']), list(df['Aspect_4']), list(df['Aspect_5'])]
    complaint_label = [list(df['Label_1']), list(df['Label_2']), list(df['Label_3']), list(df['Label_4']), list(df['Label_5'])]
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


for i in range(len(complaint_labels)):
  for j in range(len(complaint_labels[i])):
    if(str(complaint_labels[i][j]) == 'nan'):
      complaint_labels[i][j] = 0.0

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

for i in range(len(complaint_labels)):
  for j in range(len(complaint_labels[i])):
      if is_float(complaint_labels[i][j]):
        complaint_labels[i][j] = int(complaint_labels[i][j])
      else:
          complaint_labels[i][j] = int(1)

comp_dict = {}
for i in range(len(aspect_labels)):
    for j in range(len(aspect_labels[i])):
        if aspect_labels[i][j] not in comp_dict:
            comp_dict[aspect_labels[i][j]] = complaint_labels[i][j]
        comp_dict[aspect_labels[i][j]] += complaint_labels[i][j]

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
# print(aspect_label_freqs)
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
    if('No_Aspects' not in a):
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
    if('Display' in a):
        for j in range(4):
            #print("yes")
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    if('Energy_Reservior' in a):
        for j in range(3):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    if('Camera' in a):
        for j in range(4):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    if('Customer_Service' in a):
        for j in range(4):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)
    if('Feedback' in a):
        for j in range(4):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)

    if('Functionality' in a):
        for j in range(2):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)

    if('Price' in a):
        for j in range(2):
            review_aspect_final.append(r)
            comp_labels_final.append(c)
            images_path_final.append(i)
            aspect_label_final.append(a)

    if('Others' in a):
        for j in range(4):
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

review_aspect_final = review_aspect_merged
images_path_final = images_path_new
comp_labels_final = comp_labels
aspect_label_final = aspect_label
# %%
complaint_labels = comp_labels_final

# %% [markdown]
# ## Encoding text and images



# # %%
# sentence_model.max_seq_length = 512
# review_embeddings = []
# for review in review_aspect_merged:
#   text_emb = sentence_model.encode(review)
#   review_embeddings.append(text_emb)

# import pickle

# # Assuming review_embeddings is a list of numerical data or arrays

# # Specify the file path where you want to save the embeddings
# file_path = "review_embeddings.pkl"

# # Writing the embeddings to a Pickle file
# with open(file_path, 'wb') as pickle_file:
#     pickle.dump(review_embeddings, pickle_file)

import pickle

# Specify the file path where you saved the embeddings
file_path = "review_embeddings.pkl"

# Reading the embeddings from the Pickle file
with open(file_path, 'rb') as pickle_file:
    review_embeddings = pickle.load(pickle_file)

# %%
def image_preprocessing(path):
    #print(path)
    img = keras_image.load_img(path, target_size=(224, 224), interpolation='bicubic')
    img = keras_image.img_to_array(img)
    img = img//255.0
    return img

images = []
for img_path in images_path_new:
  img = image_preprocessing(img_path)
  images.append(img)

images_list_new = []
for a,i in zip(aspect_label_final, images):
    for aspect in a:
        images_list_new.append(i)

# %%
# Count the distinct labels in the aspect labels
aspect_label, aspect_label_freqs = get_labels_and_frequencies(aspect_label_final)

# %%
# Count number of distinct aspect labels
aspect_labels_count = [item for sublist in aspect_label_final for item in sublist]
aspect_labels_count = list(set(aspect_labels_count))

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

# Create a list of list as the size of complaint_labels and one list in it should be of size 10 in which 1 is present at the index of the complaint label
complaint_labels_count = []
for i in range(len(complaint_labels)):
  temp = [0]*num_of_labels
  for j in range(num_of_labels):
    #print(i)
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
# # comp_labels = list(pd.Series(comp_labels).fillna(0.))
# # complaint_labels_reshaped = tf.keras.utils.to_categorical(comp_labels)
# review_embeddings = np.array(review_embeddings).reshape(-1, 1, 768)

# import numpy as np
# max_length = max(len(seq) for seq in review_embeddings)

# padded_review_embeddings = pad_sequences(review_embeddings, maxlen=max_length, padding='post', truncating='post', dtype='float32')

review_embeddings = np.array(review_embeddings).reshape(-1, 1, 768)
img_converted = tf.convert_to_tensor([tf.convert_to_tensor(i, dtype=tf.float32) for i in images], dtype = tf.float32)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayer(nn.Module):
    def __init__(self, input_dim, return_attention=False):
        super(AttLayer, self).__init__()
        self.return_attention = return_attention

        # Initialize the learnable weight matrix
        self.W = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.W)  # You can use other initialization methods

    def forward(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        logits = torch.matmul(x, self.W)
        ai = torch.exp(logits - torch.max(logits, dim=-1, keepdim=True).values)

        # masked timesteps have zero weight
        if mask is not None:
            ai = ai * mask
        att_weights = ai / (torch.sum(ai, dim=1, keepdim=True) + 1e-8)
        weighted_input = x * att_weights.unsqueeze(2)
        result = torch.sum(weighted_input, dim=1)
        if self.return_attention:
            return result, att_weights
        return result

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models

# Define ResNet152
resnet152 = models.resnet152(pretrained=True)
num_ftrs = resnet152.fc.in_features
#print(num_ftrs)
resnet152 = nn.Sequential(*list(resnet152.children())[:-1])  # Remove the last fully connected layer

# Freeze ResNet layers
for param in resnet152.parameters():
    param.requires_grad = False

resnet152.eval()
with torch.no_grad():
    # Assuming img_tr[0] is a single image, you can add a batch dimension
    input_image = torch.tensor(img_tr[0]).permute(2, 0, 1).unsqueeze(0)

    # Pass the input through the ResNet model
    output = resnet152(input_image)
    #print(output.shape)
    output = output.squeeze(dim=2).squeeze(dim=2)
    #print(output.shape)

import torch
import torch.nn as nn

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch_size, dim = z_mean.size()
        epsilon = torch.randn(batch_size, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

# Define the model
class VAE_GRU(nn.Module):
    def __init__(self, num_labels, resnet152):
        super(VAE_GRU, self).__init__()

        # ResNet layer
        self.resnet152 = resnet152
        self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1))

        # Text and image features concatenation
        self.img_repr_dense = nn.Linear(2048, 768)
        self.concatenated_dense = nn.Linear(1536, 768)

        self.relu = nn.ReLU()

        self.encoder1 = nn.Linear(768, 256)
        self.z_mean = nn.Linear(256, 128)
        self.z_log_var = nn.Linear(256, 128)

        self.sampling = Sampling()

        # VAE Decoder
        self.decoder_dense = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
        self.input_1_dense = nn.Linear(768, 128)

        # Linear layer
        self.linear_layer = nn.Linear(896, 768)

        # Bi-directional GRU
        self.bi_lstm = nn.GRU(768, 256, bidirectional=True, batch_first=True, dropout=0.3)

        self.att_layer = AttLayer(512)

        # Final output layers
        self.out_aspects_dense = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
            nn.Sigmoid()
        )

    def forward(self, input_1, image_input):
        # ResNet feature extraction
        #print(image_input.shape)
        resnet_output = self.resnet152(image_input)
        pooling_output = self.pooling_layer(resnet_output)
        img_repr = self.img_repr_dense(pooling_output.view(-1, 2048))

        # Concatenate text and image features
        #print(input_1.shape)
        #print(img_repr.shape)
        concatenated_features = torch.cat((input_1.squeeze(dim=1), img_repr), dim=1)
        concatenated_features = self.concatenated_dense(concatenated_features)
        #print(concatenated_features.shape)
        # VAE Encoder
        encoder_output = self.relu(self.encoder1(concatenated_features))
        z_mean = self.z_mean(encoder_output)
        z_log_var = self.z_log_var(encoder_output)
        z = self.sampling(z_mean, z_log_var)

        # VAE Decoder
        decoder_output = self.decoder_dense(z)
        #print(decoder_output.shape)
        #decoder_output = decoder_output.view(-1, 1, 256)

        #print(decoder_output.shape)

        input_1 = input_1.squeeze()
        #print(input_1.shape)

        # Concatenate with input_1_dense for further processing
        merged_features = torch.cat((self.input_1_dense(input_1), decoder_output), dim=-1)

        # Linear layer
        linear_output = self.linear_layer(merged_features)

        # Bi-directional GRU
        bi_lstm_output, _ = self.bi_lstm(linear_output)
        # Attention layer
        #print(bi_lstm_output.shape)
        attention_output = self.att_layer(bi_lstm_output)

        # Final output layers
        out_aspects = self.out_aspects_dense(attention_output)

        return out_aspects, linear_output, concatenated_features, z_mean, z_log_var

device = torch.device("cpu")

# Instantiate the model
num_of_labels = 9  # Replace with the actual number of labels
vae_gru_model = VAE_GRU(num_of_labels,resnet152)
vae_gru_model = vae_gru_model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(vae_gru_model.parameters(), lr=1e-4)

import torch
from torch.utils.data import DataLoader, TensorDataset

# Assuming x_tr, img_tr, y_tr, x_val, img_val, y_val are your training and validation data

# Convert numpy arrays to PyTorch tensors
x_tr_tensor = torch.Tensor(x_tr)
img_tr_tensor = torch.Tensor(img_tr)
#print(img_tr_tensor.shape)
img_tr_tensor = torch.tensor(img_tr_tensor).permute(0, 3, 1, 2)
y_tr_tensor = torch.Tensor(y_tr)

x_val_tensor = torch.Tensor(x_val)
img_val_tensor = torch.Tensor(img_val)
img_val_tensor = torch.tensor(img_val_tensor).permute(0, 3, 1, 2)
y_val_tensor = torch.Tensor(y_val)

# Create TensorDatasets
train_dataset = TensorDataset(x_tr_tensor, img_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(x_val_tensor, img_val_tensor, y_val_tensor)

# Create DataLoader
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model_name = "vae_gru.pt"
min_val_loss = 100
best_epoch = 0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# Assuming vae_gru_model, optimizer, train_loader, val_loader are defined

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    vae_gru_model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Training
    for text, image, labels in train_loader:
        #print(image.shape)
        #image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        #print(image.shape)
        text, image, labels = text.to(device), image.to(device), labels.to(device)
        outputs, linear_output, concatenated_features, z_mean, z_log_var = vae_gru_model.forward(text, image)

        reconstruction_loss = nn.MSELoss()(concatenated_features, linear_output)

        # KL Divergence Loss
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))

        # Total Loss
        vae_loss = reconstruction_loss + kl_loss

        # Binary Cross-Entropy Loss
        bce_loss = nn.BCEWithLogitsLoss()(outputs, labels)

        # Combined Loss
        combined_loss = bce_loss + 0.5 * vae_loss

        loss = combined_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        #print(outputs[0])
        # Apply threshold
        threshold = 0.5
        predicted_labels = (outputs > threshold).float()

        # Calculate accuracy for each sample
        accuracy_per_sample = (predicted_labels == labels).float()

        # Calculate overall accuracy
        overall_accuracy = accuracy_per_sample.mean().item()
        # predicted, _ = torch.max(outputs, 1)
        # print(predicted.shape)
        correct += overall_accuracy
        total_samples += labels.size(0)

    # Print training statistics
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_samples * 100

    print(f"Epoch : {epoch+1}")
    print(f"Training Loss : {avg_loss}")
    print(f"Training Accuracy : {accuracy}")
    print("")

    # Validation loop
    vae_gru_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for val_text, val_image, val_labels in val_loader:
            val_text, val_image, val_labels = val_text.to(device), val_image.to(device), val_labels.to(device)
            val_outputs, val_linear_output, val_concatenated_features, _, _ = vae_gru_model(val_text, val_image)

            val_reconstruction_loss = nn.MSELoss()(val_concatenated_features, val_linear_output)

            # KL Divergence Loss
            val_kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))

            # Total Loss
            val_vae_loss = val_reconstruction_loss + val_kl_loss

            # Binary Cross-Entropy Loss
            val_bce_loss = nn.BCEWithLogitsLoss()(val_outputs, val_labels)

            # Combined Loss
            val_combined_loss = val_bce_loss + 0.5 * val_vae_loss

            val_loss += val_combined_loss.item()
            threshold = 0.5
            predicted_labels = (val_outputs > threshold).float()

            # Calculate accuracy for each sample
            accuracy_per_sample = (predicted_labels == val_labels).float()

            # Calculate overall accuracy
            overall_accuracy = accuracy_per_sample.mean().item()
            # predicted, _ = torch.max(outputs, 1)
            # print(predicted.shape)
            val_correct += overall_accuracy
            val_total_samples += val_labels.size(0)
            # _, val_predicted = torch.max(val_outputs, 1)
            # val_correct += (val_predicted == val_labels).sum().item()
            # val_total_samples += val_labels.size(0)

    # Print validation statistics
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total_samples * 100

    if ((min_val_loss - avg_val_loss) > 1e-4):
        min_val_loss = avg_val_loss
        best_epoch = epoch+1
        torch.save(vae_gru_model.state_dict(), model_name)

    print(f"Epoch : {epoch+1}")
    print(f"Validation Loss : {avg_val_loss}")
    print(f"Validation Accuracy : {val_accuracy}")
    print("")

    vae_gru_model.load_state_dict(torch.load(model_name))

    #print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

import torch

# Example outputs and labels
outputs = torch.tensor([8.9438e-02, 3.2532e-03, 9.6539e-01, 1.6857e-02, 5.0016e-01, 3.4694e-02,
                        5.4462e-04, 9.5172e-01, 6.2582e-03], requires_grad=True)
labels = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.float32)

# Apply threshold
threshold = 0.5
predicted_labels = (outputs > threshold).float()

# Calculate accuracy for each sample
accuracy_per_sample = (predicted_labels == labels).float()

# Calculate overall accuracy
overall_accuracy = accuracy_per_sample.mean().item()


























