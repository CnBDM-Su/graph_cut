from infomap import Infomap
import numpy as np

def infomap(num, in_path, out_path):

    edge = np.load(in_path)
    pred_label = np.zeros(shape=(num,))

    network = Infomap("--two-level")
    for i in range(edge.shape[1]):
        network.addLink(int(edge[0,i]), int(edge[1,i]), edge[2,i])

    del edge

    network.run()
    for node in network.tree:
        if node.is_leaf:
            pred_label[int(node.node_id)] = node.module_id

    np.save(out_path, pred_label)