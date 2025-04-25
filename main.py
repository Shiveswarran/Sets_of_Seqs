import pandas as pd
import numpy as np
import torch
import ast
from sentence_transformers import SentenceTransformer


#Data Loading
df_train = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Sets_of_Seqs/data/train.csv')
df_valid = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Sets_of_Seqs/data/valid.csv')
df_test = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Sets_of_Seqs/data/test.csv')


#Binary Classifier

#Data Preprocessing
df_train = df_train[['products_before','order_after']]
df_valid = df_valid[['products_before','order_after']]
df_test = df_test[['products_before','order_after']]

#Converts the string representation of list to actual list
def product_splitting(s):
    items = ast.literal_eval(s)
    return [str(item).strip() for item in items]

df_train['products_before'] = df_train['products_before'].apply(product_splitting)
df_valid['products_before'] = df_valid['products_before'].apply(product_splitting)
df_test['products_before'] = df_test['products_before'].apply(product_splitting)


#Encoding the products
#Using Sentence Transformers to encode the products

def encode_products(product_list,encoder_model=SentenceTransformer('all-MiniLM-L6-v2', device='cuda')):
    embeddings = encoder_model.encode(
        product_list,
        convert_to_numpy=True,    # get numpy array
        batch_size=32,            
    )
    return [embeddings[i] for i in range(embeddings.shape[0])]

encoder_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def encode_products(product_list):
    embeddings_list = []
    for product in product_list:
        embeddings = encoder_model.encode(product, convert_to_numpy=True)
        embeddings_list.append(embeddings)
    return embeddings_list

df_train['products_before'] = df_train['products_before'].apply(encode_products)
df_valid['products_before'] = df_valid['products_before'].apply(encode_products)
df_test['products_before'] = df_test['products_before'].apply(encode_products)



#OR OUTPUT ONLY EMBEDDING ARRAY

