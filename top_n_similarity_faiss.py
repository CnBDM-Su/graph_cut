import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

def generate_similarity(in_path, out_path, batch_num, K):
    #gpu
    # ngpu = faiss.get_num_gpus()
    data = np.load(in_path)

    batch_size = data.shape[0] // batch_num
    dimension = data.shape[1]

    index = faiss.IndexFlatIP(dimension)
    # index = faiss.index_cpu_to_all_gpus(index)
    index.add(data)

    edge = []

    for i in range(batch_num):
        st = i * batch_size
        if i == 20:
            ed = data.shape[0]
        else:
            ed = (i+1) * batch_size

        D, I = index.search(data[st:ed,:],K)
        D = D.ravel()
        I = I.ravel()

        lis = []
        for p in range(st, ed):
            a = np.ones((1,K)) * p
            lis.append(a)

        lis = np.concatenate(lis).ravel()
        edge.append(np.concatenate([lis.reshape(1,-1), I.reshape(1,-1), D.reshape(1,-1)],0))

    edges = np.concatenate(edge, 1)

    np.save(out_path, edges)
