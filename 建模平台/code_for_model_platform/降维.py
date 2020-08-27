from sklearn.decomposition import PCA
from sklearn import manifold

def decomposit(X,n_component, mode='pca'):
    """

    :param X:
    :param n_component:
    :param mode:
    :return:
    """
    if mode == 'pca':
        x_result = PCA(n_components=n_component).fit_transform(X)
    elif mode =='mds':
        clf = manifold.MDS(n_components=n_component, n_init=1, max_iter=100)
        x_result = clf.fit_transform(X)
    elif mode =='spectral':
        x_result = manifold.SpectralEmbedding(n_components=n_component, random_state=0, eigen_solver='arpack').fit_transform(X)
    elif mode =='tsne':
        x_result = manifold.TSNE(n_components=n_component, init='pca', random_state=0).fit_transform(X)
    return x_result
