from sklearn.decomposition import PCA
from sklearn import manifold
import numpy as np

def decomposit(X,n_component, mode='pca'):
    """
    created by Shaoming, Wang
    集成了目前较为流行的降维算法
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

# 将数据中心化
def centralize(X):
    mean = np.mean(X, axis=0)
    return (X - mean)

def PCA_M(X, n_components=None):
    """
    created by Shaoming, Wang
    :param X: 需要降维的多维array, 要经过缺失值填充，否则会报错
    :param n_components: 需要划分成几个部分
    :return:
    第一个对象：降维之后的结果，一个矩阵
    第二个对象：

    """
    var_s = X.columns
    # step1 数据中心化
    X_ = centralize(X)
    # step2 计算协方差矩阵
    cov_matrix = np.mat(np.cov(X_, rowvar=False))
    # step3 对协方差矩阵做正交分解 选取最大的 n_components 个特征值
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    # todo 这里存在一个bug需要解决：当解特征方程时存在多重根时，
    # 存在一个特征值对应多个特征向量，此时这些特征向量并不是正交的，需要进行正交化
    # 但是因为本方法适用在建模相关性检验未通过的一组变量，因此可以暂不考虑这个问题
    sorted_indice = np.argsort(eigen_values)
    if not n_components:
        indice = sorted_indice[::-1]
    else:
        indice = sorted_indice[:-(n_components+1):-1]
    # 构成一组新基
    base = eigen_vectors[:,indice]
    # 数据降维
    result = np.dot(X, base)
    return result, var_s[indice]

def get_percentage_(X, X_, model='variance'):

    if model == 'variance':
        X_ORI_VARIANCE = np.sum(X.apply(get_col_variance,axis=1))
        X_APP_VARIANCE = get_col_variance(np.array(X_))
        return X_APP_VARIANCE/X_ORI_VARIANCE

def get_col_variance(X):
    row_n = X.shape[0]
    return 1/(row_n -1) * np.sum((X - np.mean(X))**2)

# todo
def var_clus(X,eigenmax=1,maxclus=None):
    pass
