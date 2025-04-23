import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

class PurchaseBinaryClassifier(nn.Module):
    """
    Model that uses T5 encoder to embed product names,
    then aggregates (averages) across all products in 'products_before' to get a single user embedding.
    Finally, uses a linear + sigmoid for binary classification.
    """
    def __init__(self, pretrained_model_name='t5-small', device='cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.t5_encoder.config.d_model
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, 1) 

    def embed_products(self, product_names):
        """
        For each product string, grab the T5 encoder hidden‐state at the last non‐padded token.
        Then average those per‐product embeddings to get a single (1, hidden_size) tensor.
        """
        embeddings = []
        for text in product_names:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # outputs = self.t5_encoder(**inputs)
            # outputs = outputs.last_hidden_state(dim=1)  # shape: (1, hidden_size)
            # embeddings.append(outputs)
            outputs = self.t5_encoder(**inputs)
            last_hidden = outputs.last_hidden_state #(1,seq_len, hidden_size)
            # Get the last non-padded token's hidden state
            mask = inputs['attention_mask'] # (1, seq_len)
            seq_len = mask.sum(dim=1).item()  # e.g. 5
            #grab the hidden state of the last non-padded token
            prod_embed = last_hidden[0, seq_len-1, :].unsqueeze(0)  # (1,hidden_size)
            embeddings.append(prod_embed)
        if not embeddings:
            return torch.zeros((1, self.hidden_size), device=self.device)
        # Concatenate all product embeddings
        all_prods = torch.cat(embeddings, dim=0)  # (num_products, hidden_size)
        # average pool over products
        user_embed = all_prods.mean(dim=0, keepdim=True)  # TO be modified with DeepSets / Set Transformer
        return user_embed

    def forward(self, batch_product_lists):
        """
        batch_product_lists: List of length B, each an inner List[str].
        Returns logits of shape (B,1).
        """
        embeds = []
        for products in batch_product_lists:
            embeds.append(self.embed_products(products))
        
        batch_embeds = torch.cat(embeds, dim=0)  # (B, hidden_size)
        logits = self.classifier(batch_embeds)  # (B, 1)
        return logits

class ProductMultiLabelPredictor(nn.Module):
    """
    Model that uses T5 encoder to embed product names for 'products_before' to get a user embedding.
    Then compares the user embedding with a stored embedding of each candidate product by dot product.
    """
    def __init__(self, candidate_product_list, pretrained_model_name='t5-small', device='cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.t5_encoder.config.d_model
        self.candidate_product_list = candidate_product_list
        
        # Precompute candidate embeddings
        # (num_candidates, hidden_size)
        with torch.no_grad():
            candidate_embeddings = []
            for product in candidate_product_list:
                inputs = self.tokenizer(product, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.t5_encoder(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1)  # shape: (1, hidden_size)
                candidate_embeddings.append(pooled)
            candidate_embeddings_tensor = torch.cat(candidate_embeddings, dim=0)
        # We store these embeddings as a buffer so they're not treated as trainable parameters
        self.register_buffer("candidate_embeddings", candidate_embeddings_tensor)

    def embed_products(self, product_names):
        embeddings = []
        for text in product_names:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.t5_encoder(**inputs)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
            
            mask = inputs['attention_mask']
            seq_len = mask.sum(dim=1).item()
            prod_embed = last_hidden[0, seq_len-1, :].unsqueeze(0)
            embeddings.append(prod_embed)

        if not embeddings:
            return torch.zeros((1, self.hidden_size), device=self.device)

        all_prods = torch.cat(embeddings, dim=0)
        user_embed = all_prods.mean(dim=0, keepdim=True)
        return user_embed
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #     outputs = self.t5_encoder(**inputs)
        #     outputs = outputs.last_hidden_state
        #     embeddings.append(outputs)
        # if len(embeddings) == 0:
        #     # Handle empty product list
        #     return torch.zeros((1, self.hidden_size), device=self.device)
        # embeddings_tensor = torch.cat(embeddings, dim=0)
        # user_embedding = embeddings_tensor.mean(dim=0, keepdim=True)  # (1, hidden_size)
        # return user_embedding

    def forward(self, batch_product_lists):
        """"
        batch_product_lists: List of length B, each an inner List[str].
        Returns logits of shape (B, num_candidates).
        """
        embeds = []
        for products in batch_product_lists:
            embeds.append(self.embed_products(products))  # (1, hidden)
        batch_embeds = torch.cat(embeds, dim=0)          # (B, hidden)
        # dot against precomputed candidates -> (B, num_candidates)
        logits = batch_embeds @ self.candidate_embeddings.T
        return logits