import community as community_louvain
import networkx as nx
import numpy as np
from collections import defaultdict
import argparse

from utils import extract_domain_network_nodes

def generate_domain_knowledge__network(node_sets):
    '''
    This function generates hetergenous network to represent domain knowledge
    '''
    node2id = defaultdict(lambda: -1)
    edge_weights = defaultdict(lambda: 0)
    G = nx.Graph()

    for node_set in node_sets:
        tokens = node_set.strip().split(' ')

        for i, x in enumerate(tokens[:-1]):
            for y in tokens[i+1:]:
                if x == y:
                    continue

                if node2id[x] == -1:
                    node2id[x] = len(node2id)
                    G.add_node(node2id[x])
                if node2id[y] == -1:
                    node2id[y] = len(node2id)
                    G.add_node(node2id[y])
                edge_weights[(x,y)] += 1
                G.add_edge(node2id[x], node2id[y], weight=edge_weights[(x, y)])
    return G, edge_weights, node2id

def generate_domain_embeddings(args, node_sets, partitions, node2id, G):
    '''
    This function produces domain embeddings using the hetergenous network
    '''
    domain2vecs = []
    for node_set in node_sets:
        # compute overlapping with each partition
        temp_scores = defaultdict(lambda: 0)
        for item in partitions:
            temp_scores[partitions[item]] = 0

        for token in node_set.strip().split(' '):
            k = 0
            id = node2id[token]
            try:
                cluster_id = partitions[id]
                for e in G.edges(id):
                    k += G.edges[e]['weight']
                temp_scores[cluster_id] += k
            except:
                print('undefined node (node {}) is found'.format(node2id[token]))

        # generate the domain embedding
        max = 0
        max_id = -1
        for item in temp_scores:
            if max < temp_scores[item]:
                max = temp_scores[item]
                max_id = item

        temp_vec = []
        for item in temp_scores:
            temp_vec.append(temp_scores[item])
        temp_vec = np.array(temp_vec)

        if np.sum(temp_vec) != 0:
            temp_vec = temp_vec/np.sum(temp_vec)
        domain2vecs.append(temp_vec)
    print('Domain Embedding Processing (in shape {}) Complete'. format(np.array(domain2vecs).shape))
    np.save(open(args.path + args.dataset + '/' + args.dataset + '_domain_emb.npy', 'wb'), np.array(domain2vecs))
    return domain2vecs

def discover_domains(args):
    '''
    This function runs the pipeline for the proposed domain discovery algorithm
    '''
    # generate nodes in the domain-network
    global_node_sets = []
    local_node_sets = defaultdict(lambda: None)
    for dataset in args.datasets:
        args.dataset = dataset
        node_sets = extract_domain_network_nodes(args)

        global_node_sets = global_node_sets + node_sets
        local_node_sets[dataset] = node_sets

    # generate domain network
    G, edge_weights, node2id = generate_domain_knowledge__network(global_node_sets)
    # compute louvain communities
    partitions = community_louvain.best_partition(G)

    # generate domain embeddings
    for dataset in args.datasets:
        args.dataset = dataset
        domain2vecs = generate_domain_embeddings(args, local_node_sets[dataset], partitions, node2id, G)
        print('Domain Embeddings for {} dataset are stored at {}'.format(dataset, args.path+dataset+'/'))


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="data directory", type=str, default='./DATA/')
    parser.add_argument("--datasets", help='datasets_name', type=str, default=['TOY1', 'TOY2', 'TOY3'])
    args = parser.parse_args()

    discover_domains(args)
