import pandas as pd
import numpy as np
import torch
import ast
from sentence_transformers import SentenceTransformer


#Data Loading
df_train = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Sets_of_Seqs/data/train.csv')
df_valid = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Sets_of_Seqs/data/valid.csv')
df_test = pd.read_csv('/scratch/umni5/a/shives/Course_Projects/CS_587_DL/Sets_of_Seqs/data/test.csv')


#Converts the string representation of list to actual list
def product_splitting(s):
    items = ast.literal_eval(s)
    return [str(item).strip() for item in items]

for df in (df_train, df_valid, df_test):
    df['products_before'] = df['products_before'].apply(product_splitting)
    df['post_cutoff'] = df['post_cutoff'].apply(product_splitting)


#ENCODING PRODUCTS
#Using Sentence Transformers to encode the products

#Build the set of all unique products
all_products = set()
for df in (df_train, df_valid, df_test):
    all_products.update(df['products_before'].explode().unique())

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
product_list = list(all_products)
embeddings = model.encode(
    product_list,
    convert_to_numpy=True,
    batch_size=64
)
product_embedding_dict = dict(zip(product_list, embeddings))

#Lookup function that just pulls from the dict
def encode_products_from_dict(product_list, embedding_dict=product_embedding_dict):
    # stack into a (len(product_list), D) array
    return np.vstack([embedding_dict[prod] for prod in product_list])

#Apply lookup to each row
for df in (df_train, df_valid, df_test):
    df['products_before_embeddings'] = df['products_before'].apply(encode_products_from_dict)
    df['post_cutoff_embeddings'] = df['post_cutoff'].apply(encode_products_from_dict)

# now df['{products_before, post_cutoff}_embeddings'] is an array of shape (N_items, embed_dim) per row,



#Janossy Pooling





#DeepSet







#Set Transformer












#BINARY CLASSIFIER *** WILL THE USER BUY OR NOT IN NEXT WEEK ***





#MULTILABEL CLASSIFIER *** WHAT WILL THE USER BUY NEXT WEEK ***
