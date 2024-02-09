import torch
import torch.nn as nn

# an embedding model using pytorch
class EmbeddingModel(nn.Module):
    def __init__(self, embd_sizes, num_numerical_features):
        super(EmbeddingModel, self).__init__()
        self.embd_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embd_sizes])
        self.num_layers = num_numerical_features

    def forward(self, x_cat, x_num):
        print(x_cat.shape)
        embd_outputs = [embdedding(x_cat[:,i]) for i, embdedding in enumerate(self.embd_layers)]  
        embd_outputs = torch.cat(embd_outputs, axis = 1)
        concatenated_outputs = torch.cat((embd_outputs, x_num), axis = 1)
        return concatenated_outputs