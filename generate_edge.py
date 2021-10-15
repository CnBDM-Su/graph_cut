import numpy as np

def generate_graph_edge(threshold, in_path, out_path):

    edge = np.load(in_path)
    lis = np.where(edge[2,:] > threshold)[0]
    edge = edge[:,lis]

    del_col = []
    for i in range(edge.shape[1]):
        if edge[0, i] == edge[1, i]:
            del_col.append(i)

    edge = np.delete(edge, del_col, axis=1)
    np.save(out_path, edge)