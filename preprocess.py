import json

import nltk
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler

from utils import preprocess_text, load_graphs, generate_graph_level_features, generate_node_level_features


def generate_text_representation(args):
    '''
    This function generates text embeddings of the news records
    '''
    text_file = open(args.path + args.dataset + '/' + args.dataset + '_text.txt', 'r')
    text_contents = []
    text_embs = []

    roberta_model = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta_model.eval()

    for line in text_file:
        text_contents.append(line.strip())
    text_contents = preprocess_text(text_contents)
    print(text_contents)

    for text in text_contents:
        tokens = roberta_model.encode(text)
        final_layer = roberta.extract_features(tokens, return_all_hiddens=True)[-1]
        final_layer = final_layer.detach().numpy()
        text_emb = np.mean(final_layer[0], axis = 0)
        text_embs.append(text_emb)
    print('Text Feature Processing (in shape {}) Complete'. format(np.array(text_embs).shape))
    np.save(open(args.path + args.dataset + '/' + args.dataset + '_txt_emb.npy', 'wb'), np.array(text_embs))

def node_aggregation(childs, features, sources, iterations = 10):
    '''
    This function performs the node-level aggregation proposed in (Silva et al. 2020)
    to summarize node-level features at the source node
    '''
    gamma = 0.5
    no_features = features.shape[1]
    embeddings = defaultdict(lambda: defaultdict(lambda: np.zeros((no_features,))))

    # initialization
    for node, fea in enumerate(features):
        embeddings[0][node] = features[node]

    # aggregations
    for i in range(iterations):
        for node, fea in enumerate(features):
            neigb_values = np.zeros((no_features,))
            for child in childs[node]:
                neigb_values += embeddings[i][child]
            if len(childs[node]) > 0:
                embeddings[i+1][node] = gamma*embeddings[i][node] + (1-gamma)*neigb_values/len(childs[node])
            else:
                embeddings[i+1][node] = embeddings[i][node]

    aggregated_features = []
    for g_id in sources:
        aggregated_features.append(embeddings[iterations][sources[g_id]])
    return np.array(aggregated_features)

def generate_network_representation(args, local_features, global_features):
    '''
    This function generates network embeddings of the news records, which also performs
    standard scaling for network features
    '''
    network_features = np.concatenate((local_features, global_features), axis = 1)

    # scale network features using a standard scaler
    for col in range(network_features.shape[1]):
        network_features[:, col] = StandardScaler().fit_transform(network_features[:, col].reshape(-1,1)).reshape(-1,)
    print('Network Feature Processing (in shape {}) Complete'. format(network_features.shape))
    np.save(open(args.path + args.dataset + '/' + args.dataset + '_net_emb.npy', 'wb'), network_features)

def generate_labels(args):
    labels = []
    f = open(args.path + args.dataset + '/' + args.dataset + '_graph_labels.txt', 'r')
    for line in f:
        labels.append(int(line.strip().split(',')[0]))
    labels = np.array(labels)
    print('labels Processing (in shape {}) Complete'. format(labels.shape))
    np.save(open(args.path + args.dataset + '/' + args.dataset + '_labels.npy', 'wb'), labels)

def preprocess(args):
    '''
    This function runs the pipeline for preprocessing
    '''

    # generate node-level network features
    node_level_features = generate_node_level_features(args)
    graphs, sources, childs = load_graphs(args)
    # perform node aggregation
    aggregated_features = node_aggregation(childs, node_level_features, sources)

    # generate graph-level network features
    graph_level_features = generate_graph_level_features(graphs, sources, childs)

    # generate network representation
    generate_network_representation(args, aggregated_features, graph_level_features)

    # generate text representation
    # generate_text_representation(args)

    # generate labels
    generate_labels(args)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="data directory", type=str, default='./DATA/')
    parser.add_argument("--dataset", help='dataset_name', type=str, default='TOY1')
    args = parser.parse_args()

    preprocess(args)
