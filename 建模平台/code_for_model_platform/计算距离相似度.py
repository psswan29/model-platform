from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import numpy as np

# 计算向量的模
def mod(vec):
    x = np.sum(vec**2)
    return x ** 0.5

def sim(vect1, vect2):
    s = np.dot(vect1, vect2) / mod(vect1)/ mod(vect2)
    return s

def cal_euclidean_dis(vect1, vect2):
    return  np.sqrt(np.sum((vect2 - vect1) ** 2))


if __name__ == '__main__':
    users = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    rating_matrix = np.array([[4,3,0,0,5,0],
                              [5,0,4,0,4,0],
                              [4,0,5,3,4,0],
                              [0,3,0,0,0,5],
                              [0,4,0,0,0,4],
                              [0,0,2,4,0,5]
                              ])


    distance = euclidean_distances(rating_matrix[0].reshape(-1,1), rating_matrix[1].reshape(-1,1))