from sklearn.cluster import KMeans

def pretrain_k_means(X, num_clusters):
    model = KMeans(n_clusters=num_clusters, random_state=42, n_init=100)
    model_fit = model.fit(X)
    return model_fit