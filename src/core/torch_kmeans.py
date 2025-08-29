import torch

class TorchKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, device=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cluster_centers_ = None

    def fit(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples)[:self.n_clusters]
        centers = X[indices]

        for _ in range(self.max_iter):
            distances = torch.cdist(X, centers)
            labels = torch.argmin(distances, dim=1)
            new_centers = torch.stack([X[labels == i].mean(dim=0) if (labels == i).any() else centers[i]
                                       for i in range(self.n_clusters)])
            shift = torch.norm(centers - new_centers)
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels.cpu().numpy()
        return self