from sklearn.decomposition import PCA
from sklearn import manifold

def decomposit(X,n_component, mode='pca'):
    """

    :param X: 需要降维的多维array
    :param n_component: 需要划分成几个部分
    :param mode: 选用的降维方法，默认pca降维
    :return:
    x_result ： 划分后的结果， 是一个降维后的array
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
