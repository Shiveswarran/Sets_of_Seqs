import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from typing import List, Dict


class ProductEmbedder:
    def __init__(self, model_name: str = 't5-small', device=None):
        from config import Config
        self.device = device or Config.DEVICE
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def embed_products(self, product_names: List[str]) -> np.ndarray: #
        with torch.no_grad():
            inputs = self.tokenizer(
                product_names, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=64
            ).to(self.device)
            
            outputs = self.model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1) #
            return embeddings.cpu().numpy()

class PurchaseSequenceEncoder:
    def __init__(self, embedder: ProductEmbedder):
        from config import Config
        self.embedder = embedder
        self.embedding_dim = Config.EMBEDDING_DIM
        
    def encode_user_history(self, product_lists: List[List[str]]) -> np.ndarray:
        all_products = list(set(p for sublist in product_lists for p in sublist))
        product_embeddings = self.embedder.embed_products(all_products)
        product_to_embedding = dict(zip(all_products, product_embeddings))
        
        user_embeddings = []
        for products in product_lists:
            if not products:
                user_embedding = np.zeros(self.embedding_dim)
            else:
                embeddings = [product_to_embedding[p] for p in products]
                user_embedding = np.mean(embeddings, axis=0) #
            user_embeddings.append(user_embedding)
            
        return np.array(user_embeddings)