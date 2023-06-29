import os
import re
import sys
import json
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot
# from weights import *
sys.path.append('./COOT/code')
import cot
import seaborn as sns

# convert mapper graph to hypergraph
def mapper2hypergraph(filename, output_filename):
    with open(filename) as f:
        mapper_graph = json.load(f)
    nodes = mapper_graph['nodes']
    with open(output_filename, 'w') as f:
        for node in nodes:
            line = node + "," + ",".join(str(v) for v in nodes[node]['vertices']) + "\n"
            f.write(line)
            
def write_hypergraph(hgraph, output_filename):
    with open(output_filename, 'w') as f:
        for he in hgraph:
            line = he + "," + ",".join(str(v) for v in hgraph[he]) + "\n"
            f.write(line)
            
def process_hypergraph(filename):
    """
    Returns hgraph
    """
    with open(filename) as f:
        # hyper_data = f.read()
        hyper_data = f.readlines()
    hyper_data.sort()
    
    hgraph = {}
    hlabel2id = {}
    vlabel2id = {}
#     he_id = 1
#     v_id = 1
    he_id = 0
#     v_id = 0
    # for line in hyper_data.split("\n"):
    for line in hyper_data:
        line = line.rstrip().rsplit(',')
        hyperedge, vertices = line[0], line[1:]
        if hyperedge != "":
            if hyperedge not in hlabel2id.keys():
                hyperedge_label = re.sub('[\'\s]+', '', hyperedge)
#                 new_id = 'h'+str(he_id)
                new_id = he_id
                he_id += 1
                hlabel2id[hyperedge_label] = new_id
                hyperedge = new_id
            vertices_new = []
            for v in vertices:
                v_label = re.sub('[\'\s]+', '', v)
                if v_label != "":
                    if v_label not in vlabel2id.keys():
#                         new_id = 'v'+str(v_id)
                        new_id = str(v_label)
#                         new_id = str(v_id)
#                         v_id += 1
                        vlabel2id[v_label] = new_id
                        vertices_new.append(new_id)
                    else:
                        vertices_new.append(vlabel2id[v_label])
            vertices = vertices_new
            if hyperedge not in hgraph.keys():
                hgraph[hyperedge] = vertices
            else:
                hgraph[hyperedge] += vertices
    print(len(hlabel2id), len(vlabel2id))
    label2id = {"h": hlabel2id, "v": vlabel2id}
    return hgraph, label2id

def get_hgraph_dual(hgraph):
    hgraph_dual = {}
    for he in hgraph:
        for v in hgraph[he]:
            if v not in hgraph_dual: 
                hgraph_dual[v] = [he]
            else:
                hgraph_dual[v].append(he)
    return hgraph_dual

def convert_to_line_graph(hgraph_dict):
    # Line-graph is a NetworkX graph
    line_graph = nx.Graph()

    node_list = list(hgraph_dict.keys())
    node_list.sort() # sort the node by id
    # Add nodes
    [line_graph.add_node(edge) for edge in node_list]


    # For all pairs of edges (e1, e2), add edges such that
    # intersection(e1, e2) is not empty
    s = 1
    for node_idx_1, node1 in enumerate(node_list):
        for node_idx_2, node2 in enumerate(node_list[node_idx_1 + 1:]):
            vertices1 = hgraph_dict[node1]
            vertices2 = hgraph_dict[node2]
            if len(vertices1) > 0 or len(vertices2) > 0:
                # Compute the intersection size
                intersection_size = len(set(vertices1) & set(vertices2))
                union_size = len(set(vertices1) | set(vertices2))
                jaccard_index = intersection_size / union_size
                if intersection_size >= s:
                    # line_graph.add_edge(node1, node2, intersection_size=1/intersection_size, jaccard_index=1/jaccard_index)
                    line_graph.add_edge(node1, node2, intersection_size=1/intersection_size, jaccard_index=1-jaccard_index)
    # line_graph = nx.readwrite.json_graph.node_link_data(line_graph)
    return line_graph

def get_u(hgraph_dual_dict, v2weight=None):
    if not v2weight: # if no collapse: all weight = 1
        v2weight = {}
        for v in hgraph_dual_dict:
            v2weight[v] = 1
    v2deg = {}
    for v in hgraph_dual_dict:
        deg_v = len(hgraph_dual_dict[v]) * v2weight[v] # num_degree * how many such vertex in the original hypergraph
        v2deg[v] = deg_v
    # vlist = [i for i in range(50000)]
    vlist = [i for i in range(800000)]
#     vlist = [int(v) for v in list(hgraph_dual_dict.keys())]
#     vlist.sort() # order is important!
    vlist = [str(v) for v in vlist]
    deg_list = []
    for v in vlist:
        if v in v2deg:
            deg_list.append(v2deg[v])
        else:
            deg_list.append(0)
    return np.array(deg_list) / sum(deg_list), v2deg, vlist


def get_v(hgraph_dict, v2deg):
#     vdict = collections.defaultdict(list)
    hedict = {}
    edge_list = []
#     for i in range(len(hgraph_dict.keys())):
#         h = "h" + str(i+1)
    for h in hgraph_dict:
        total_vdeg = 0
        for v in hgraph_dict[h]:
            total_vdeg += v2deg[v]
        hedict[h] = total_vdeg
#         edge_list.append(h)
    edge_list = list(hgraph_dict.keys())
    edge_list.sort()
    # print("get_v", edge_list)
    deg_list = [] # sort by hyperedge id
    for h in edge_list:
        deg_list.append(hedict[h])
    return np.array(deg_list) / sum(deg_list), edge_list

def get_omega_no_weight(h_dict, h_edges, h_nodes):
    # w_ij = 1 if node in the hyperedge, otherwise w_ij = 0
    num_edges = len(h_edges)
    num_nodes = len(h_nodes)

    edges_id2idx = {}
    nodes_id2idx = {}
    for i in range(len(h_edges)):
        edges_id2idx[h_edges[i]] = i
    for i in range(len(h_nodes)):
        nodes_id2idx[h_nodes[i]] = i
        
#     print("edges_id2idx", edges_id2idx)
#     print("nodes_id2idx",nodes_id2idx)

    # w = np.zeros((50000, num_edges))
    w = np.zeros((800000, num_edges))
    for edge in h_dict:
        edge_idx = edges_id2idx[edge]
        nodes_list = h_dict[edge]
        for node in nodes_list:
            node_idx = nodes_id2idx[node]
            w[node_idx, edge_idx] = 1
    return w

def collapse_vertex(hgraph_dual):
    vlist = list(hgraph_dual.keys())
    hkey2v = collections.defaultdict(list)
    for v in vlist:
        hset = ''.join(['h'+str(hkey) for hkey in hgraph_dual[v]])
        hkey2v[hset].append(v)
    vnew2weight = {}
    hgraph_dual_new = collections.defaultdict(list)
#     idx = 1
    idx = 0
    for hkey in hkey2v:
#         vnew = 'v'+str(idx)
        vnew = str(idx)
        vnew2weight[vnew] = len(hkey2v[hkey])
        for v in hkey2v[hkey]:
            for he in hgraph_dual[v]:
                if he not in hgraph_dual_new[vnew]:
                    hgraph_dual_new[vnew].append(he)
        idx += 1 
    print("num of vertices after collapse:", len(hgraph_dual_new))
    return hgraph_dual_new, vnew2weight

def get_omega(hgraph, hgraph_dual, lgraph,h_edges, h_nodes, weight_type=None):
    """    
    Node-hyperedge distances. Uses single Floyd-Warshall call
    on line graph
    
    Parameter:
    hgraph      : hnx.Hypergraph, nodes in form [str(v) for v in range(numNodes)] 
    hgraph_dual : hnx.Hypergraph
    lgraph      : nx.Graph
    weight_type : str, optional
        Options: None, 'intersection_size', 'jaccard_index'. The default is None.
                 If None, then each edge has weight 1.
                 
    Returns:
    w : np.ndarray
    
    """
    num_nodes, num_edges = len(hgraph_dual), len(hgraph)
    
    try:
        ldist = nx.floyd_warshall_numpy(lgraph,weight=weight_type) # May have inf 
    except:
        return "weight type doesn't exist!"
    
    edges_id2idx = {}
    nodes_id2idx = {}
    for i in range(len(h_edges)):
        edges_id2idx[h_edges[i]] = i
    for i in range(len(h_nodes)):
        nodes_id2idx[h_nodes[i]] = i
        
    w = np.zeros((50000, num_edges))
    for i in range(50000):
        node = str(i)
        if node in hgraph_dual:
            edge_list = list(hgraph_dual[node])
            edge_idx_list = [edges_id2idx[edge] for edge in edge_list]
            shortest_target_dists = ldist[edge_idx_list,:].min(axis=0)
            w[i, :] = shortest_target_dists
        else:
            w[i, :] = np.inf
            
    # for node in hgraph_dual:
    #     node = str(node)
    #     node_idx = nodes_id2idx[node]
    #     edge_list = list(hgraph_dual[node])
    #     edge_idx_list = [edges_id2idx[edge] for edge in edge_list]
    #     shortest_target_dists = ldist[edge_idx_list,:].min(axis=0)
    #     w[node_idx, :] = shortest_target_dists
#     for i in range(num_nodes):
#         node = "v"+str(i+1) # Important that hypergraph nodes were formatted this way
#         # all hyperedges containing the node
#         idxs = list(hgraph_dual[node])
#         # print(idxs)
#         idxs = [int(idx[1:])-1 for idx in idxs]
#         shortest_target_dists = ldist[idxs,:].min(axis=0)
#         w[i,:] = shortest_target_dists
        
    # replace the inf value with max * 1.1
    w[np.isinf(w)] = np.nan
    w[np.isnan(w)] = np.nanmax(w * 1.1)
    return w

def cot_wass(X1, X2, w1 = None, w2 = None, v1 = None, v2 = None,
              niter=10, algo='emd', reg=0,algo2='emd',
              reg2=0, verbose=True, log=False, random_init=False, C_lin=None): 
    print("from cot")
    if v1 is None:
        v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
    if v2 is None:
        v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
    if w1 is None:
        w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
    if w2 is None:
        w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

    # Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0]) 
    if X1.shape[0] != X2.shape[0]:
        print("node size doesn't match!")
        return
    Ts = np.eye(X1.shape[0])/X1.shape[0]
    if not random_init:
        # Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0])  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (X1.shape[1] * X2.shape[1])  # is (d,d')
    else:
        # Ts=cot.random_gamma_init(w1,w2) 
        Tv=cot.random_gamma_init(v1,v2)


    constC_s, hC1_s, hC2_s = cot.init_matrix_np(X1, X2, v1, v2)

    constC_v, hC1_v, hC2_v = cot.init_matrix_np(X1.T, X2.T, w1, w2)
    cost = np.inf

    log_out ={}
    log_out['cost'] = []
    
#     if len(w1) == len(w2):
#         C_lin = np.identity(len(w1))

#     M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T) 
    
    M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)
    Tv = ot.emd(v1, v2, M, numItermax=1e7)
    cost = np.sum(M * Tv)
#     for i in range(niter):
#         print("cot_wass iter", i)
#         Tsold = Ts
#         Tvold = Tv
#         costold = cost

#         # M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
# #         if C_lin is not None:
# #             M=M+C_lin 
                  
# #         if algo == 'emd':
# #             Ts = ot.emd(w1, w2, M, numItermax=1e7)
# #         elif algo == 'sinkhorn':
# #             Ts = ot.sinkhorn(w1, w2, M, reg)

#         M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)    
#         if algo2 == 'emd':
#             Tv = ot.emd(v1, v2, M, numItermax=1e7)
#         elif algo2 == 'sinkhorn':
#             Tv = ot.sinkhorn(v1,v2, M, reg2)

# #         delta = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
#         delta = np.linalg.norm(Tv - Tvold)
#         cost = np.sum(M * Tv)
#         if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
#             if verbose:
#                 print('converged at iter ', i)
#             break
# #     if log:
# #         return Ts, Tv, cost, log_out
# #     else:
# #         return Ts, Tv, cost
    return cost

def get_distances(hypergraphs, weight_type, num_iter=10):
    distances = []

    for i in range(0, len(filenames)):
        h1_filename = filenames[0]
        h2_filename = filenames[i]
#         print(h1_filename, h2_filename)

        h1 = hypergraphs[0]
        h2 = hypergraphs[i]

        h1_dual = get_hgraph_dual(h1)
        h2_dual = get_hgraph_dual(h2)

        u1, v2deg1, h1_nodes = get_u(h1_dual)
        u2, v2deg2, h2_nodes = get_u(h2_dual)
        
        v1, h1_edges = get_v(h1, v2deg1)
        v2, h2_edges = get_v(h2, v2deg2)
        
        l1 = convert_to_line_graph(h1)
        l2 = convert_to_line_graph(h2)

        if weight_type == "no_weight":
            w1 = get_omega_no_weight(h1, h1_edges, h1_nodes)
            w2 = get_omega_no_weight(h2, h2_edges, h2_nodes)
        else:
            w1 = get_omega(h1, h1_dual, l1, h1_edges, h1_nodes, weight_type=weight_type)
            w2 = get_omega(h2, h2_dual, l2, h2_edges, h2_nodes, weight_type=weight_type)        

        curr_dist = []

        for i in range(num_iter):
            print("iter", i)
            # cost = cot_wass(w1, w2, w1=u1, w2=u2, v1=v1, v2=v2, niter=100, log=True, verbose = False, random_init=True)
            cost = cot_wass(w1, w2, niter=100, log=True, verbose = False, random_init=True)
            print(cost)
            curr_dist.append(cost)

        distances.append(curr_dist)


    min_distances = [np.min(dist) for dist in distances]

    return distances, min_distances

if __name__ == '__main__':
    weight_type = sys.argv[1]

    # path = "../../mapper_graphs/single_batches/"
    # output_path = "../../hypergraphs/single_batches/"
    # path = "../../mapper_graphs/full_fg_1/"
    # output_path = "../../hypergraphs/full_fg_1/"
    # path = "../../mapper_graphs/full_bg_1/"
    # output_path = "../../hypergraphs/full_bg_1/"
    path = "../../mapper_graphs/full_batches/"
    output_path = "../../hypergraphs/full_batches/"
    fig_path = "../../figs/gw-distances/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # filenames = ['mapper_single_batch_layer4.1.bn2_40_25_8.71.json', 'mapper_single_batch_layer4.1.bn1_40_25_4.2.json', 'mapper_single_batch_layer4.0.bn2_40_25_5.5.json', 'mapper_single_batch_layer4.0.bn1_40_25_5.004.json', 'mapper_single_batch_layer3.1.bn2_40_25_7.69.json', 'mapper_single_batch_layer2.1.bn2_40_25_6.8.json', 'mapper_single_batch_layer1.1.bn2_40_25_4.5.json']
    # filenames = ['mapper_single_batch_layer4.1.bn2-enhanced_40_25_8.71_enhanced.json', 'mapper_single_batch_layer4.1.bn1-enhanced_40_25_4.22_enhanced.json', 'mapper_single_batch_layer4.0.bn2-enhanced_40_25_5.5_enhanced.json', 
    # 'mapper_single_batch_layer4.0.bn1-enhanced_40_25_5.004_enhanced.json', 'mapper_single_batch_layer3.1.bn2-enhanced_40_25_7.69_enhanced.json', 'mapper_single_batch_layer2.1.bn2-enhanced_40_25_6.8_enhanced.json',
    # 'mapper_single_batch_layer1.1.bn2-enhanced_40_25_4.5_enhanced.json']
    # filenames = ['mapper_full_fg_1_layer4.1.bn2_40_25_10.65.json', 'mapper_full_fg_1_layer4.1.bn1_40_25_8.5.json', 'mapper_full_fg_1_layer4.0.bn2_40_25_13.0.json', 'mapper_full_fg_1_layer4.0.bn1_40_25_9.286.json', 'mapper_full_fg_1_layer3.1.bn2_40_25_11.865.json', 'mapper_full_fg_1_layer2.1.bn2_40_25_8.511.json', 'mapper_full_fg_1_layer1.1.bn2_40_25_4.993.json']
    # filenames = ['mapper_full_fg_1_layer4.1.bn2-enhanced_40_25_10.65_enhanced.json',
    # 'mapper_full_fg_1_layer4.1.bn1-enhanced_40_25_8.5_enhanced.json',
    # 'mapper_full_fg_1_layer4.0.bn2-enhanced_40_25_13.0_enhanced.json',
    # 'mapper_full_fg_1_layer4.0.bn1-enhanced_40_25_9.286_enhanced.json',
    # 'mapper_full_fg_1_layer3.1.bn2-enhanced_40_25_11.865_enhanced.json',
    # 'mapper_full_fg_1_layer1.1.bn2-enhanced_40_25_4.993_enhanced.json',
    # 'mapper_full_fg_1_layer1.1.bn2-enhanced_40_25_4.993_enhanced.json']
    # filenames = ['mapper_single_batch_layer4.1.bn2-balanced_40_25_8.71_balanced-cover.json',
    # 'mapper_single_batch_layer4.1.bn1-balanced_40_25_4.22_balanced-cover.json',
    # 'mapper_single_batch_layer4.0.bn2-balanced_40_25_5.5_balanced-cover.json',
    # 'mapper_single_batch_layer4.0.bn1-balanced_40_25_5.004_balanced-cover.json',
    # 'mapper_single_batch_layer3.1.bn2-balanced_40_25_7.69_balanced-cover.json',
    # 'mapper_single_batch_layer2.1.bn2-balanced_40_25_6.8_balanced-cover.json',
    # 'mapper_single_batch_layer1.1.bn2-balanced_40_25_4.5_balanced-cover.json']
    # filenames = ['mapper_full_fg_1_layer4.1.bn2-balanced_40_25_10.65_balanced-cover.json',
    # 'mapper_full_fg_1_layer4.1.bn1-balanced_40_25_8.5_balanced-cover.json',
    # 'mapper_full_fg_1_layer4.0.bn2-balanced_40_25_13.0_balanced-cover.json',
    # 'mapper_full_fg_1_layer4.0.bn1-balanced_40_25_9.286_balanced-cover.json',
    # 'mapper_full_fg_1_layer3.1.bn2-balanced_40_25_11.865_balanced-cover.json',
    # 'mapper_full_fg_1_layer2.1.bn2-balanced_40_25_8.511_balanced-cover.json',
    # 'mapper_full_fg_1_layer1.1.bn2-balanced_40_25_4.993_balanced-cover.json']
    # filenames = [
    #     'mapper_full_bg_1_layer4.1.bn2_40_25_12.091.json',
    #     'mapper_full_bg_1_layer4.1.bn1_40_25_8.188.json',
    #     'mapper_full_bg_1_layer4.0.bn2_40_25_13.984.json',
    #     'mapper_full_bg_1_layer4.0.bn1_40_25_9.2.json',
    #     'mapper_full_bg_1_layer3.1.bn2_40_25_12.409.json',
    #     'mapper_full_bg_1_layer2.1.bn2_40_25_8.198.json',
    #     'mapper_full_bg_1_layer1.1.bn2_40_25_4.847.json'
    # ]
    filenames = [
        'mapper_full_batches_layer4.1.bn2_40_25_8.5.json',
        'mapper_full_batches_layer4.1.bn1_40_25_2.5.json',
        'mapper_full_batches_layer4.0.bn2_40_25_5.0.json',
        'mapper_full_batches_layer4.0.bn1_40_25_3.5.json',
        'mapper_full_batches_layer3.1.bn2_40_30_5.41.json',
        'mapper_full_batches_layer2.1.bn2_40_30_4.5.json',
        'mapper_full_batches_layer1.1.bn2_40_30_3.5.json'
    ]

    layers = ['layer16', 'layer15', 'layer14', 'layer13', 'layer12', 'layer8', 'layer4']
    hypergraphs = []
    for i in range(len(filenames)):
        filename = filenames[i]
        output_filename = os.path.join(output_path, "hypergraph_" + filename)
        if not os.path.exists(output_filename):
            print("mapper2hypergraph")
            mapper2hypergraph(os.path.join(path, filename), output_filename)
        hg, _ = process_hypergraph(output_filename)
        hypergraphs.append(hg)

    distances, min_distances = get_distances(hypergraphs, weight_type, num_iter=1)

    print(distances)

    print(min_distances)

    # plot the distances
    fig_filename = os.path.join(fig_path, 'wasserstein-distances-full-' + weight_type)
    # fig_filename = os.path.join(fig_path, 'wasserstein-distances-fg_1-' + weight_type)
    plt.figure(figsize=(15, 6), dpi=200)
    plt.boxplot(distances[::-1])
    plt.plot(np.arange(1,len(layers)+1), min_distances[::-1])
    plt.xticks(np.arange(1,len(layers)+1), layers[::-1])
    # plt.title("Random activations, "+weight_type)
    plt.title("Full activations, "+weight_type)
    # plt.title("Top 1 activations, "+weight_type)
    plt.savefig(fig_filename)
    