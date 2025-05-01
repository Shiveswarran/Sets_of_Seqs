import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models import inception_v3
from sentence_transformers import SentenceTransformer


class TextModels(nn.Module):

	def __init__(self,
				sbert_name: str = "all-MiniLM-L6-v2",
				freeze_sbert: bool = True,
				rho_hidden_dims: list[int] = [128, 64],
				rho_layers: int = 1,
				f_out_dim: int = 64):

		super(TextModels, self).__init__()

		self.encoder = SentenceTransformer(sbert_name)
		if freeze_sbert:
			for p in self.encoder.parameters():
				p.requires_grad = False
		self.d = self.encoder.get_sentence_embedding_dimension()  # 384 for MiniLM

		self.model = nn.Linear(self.d, f_out_dim)
		self.model_activation = nn.Tanh()
		self.model_out_shape = f_out_dim
		init.xavier_uniform_(self.model.weight)
		self.model.bias.data.fill_(0)	


		layers = []
		in_dim = self.model_out_shape

		for hidden_dim in rho_hidden_dims:
			layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
			in_dim = hidden_dim

		layers.append(nn.Linear(in_dim, 1))
		self.rho = nn.Sequential(*layers)


		for module in self.rho.modules():
			if isinstance(module, nn.Linear):
				init.kaiming_uniform_(module.weight, nonlinearity='relu')
				if module.bias is not None:
					init.zeros_(module.bias)

		self.loss_fn = nn.BCEWithLogitsLoss()
		


	def forward(self, baskets):

		# print(f"\ncalling forward\n")
		# print(f"Baskets length: {len(baskets)}")
	
		set_summaries = []

		for basket in baskets:
			
			# print(f"Basket shape: {basket.shape}")
		
			f_out = self.model_activation(self.model(basket))
			# print(f"F_out shape: {f_out.shape}")

			g = f_out.sum(dim=0)
			# print(f"G shape: {g.shape}")

			set_summaries.append(g)

		set_summaries = torch.stack(set_summaries)
		# print(f"Set summaries shape: {set_summaries.shape}")

		logits = self.rho(set_summaries).squeeze(-1)

		return logits



	def loss(self, baskets, labels):

		logits = self.forward(baskets)
		return self.loss_fn(logits, labels.float())

