from collections import deque
import torch

class Queue:
    def __init__(self, max_size, embedding_dim, device):
        self.queue = deque(maxlen=max_size)
        self.embedding_dim = embedding_dim
        self.device = device

    def enqueue(self, embeddings):
        embeddings = embeddings.detach().cpu()
        for embedding in embeddings:
            self.queue.append(embedding.unsqueeze(0))

    def get_all_embeddings(self):
        if len(self.queue) == 0:
            return torch.empty(0, self.embedding_dim).to(self.device)
        return torch.cat(list(self.queue), dim=0).to(self.device)