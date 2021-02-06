from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from datetime import datetime
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from copy import deepcopy
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

stop_words = stopwords.words('english')
punctuations = ['(',')',';',':','[',']',',','!','=','==','<','>','@','#','$','%','^','&','*','.','//','{','}','...','``','+',"''","?"]

def preprocess_text(sentences):
    '''
    This is an utility function to preprocess text content, which accepts list of
    sentences and output the preprocessed sentences (lematized, without stop words, and without punctuations)
    '''
    preprocessed_sentences = []
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    punctuations = ['(',')',';',':','[',']',',','!','=','==','<','>','@','#','$','%','^','&','*','.','//','{','}','...','``','+',"''","?", "‘", '’', "'", '"']

    for i, sentence in enumerate(sentences):
        preprocessed_sentence = sentence
        preprocessed_sentence = preprocessed_sentence.lower()
        preprocessed_tokens = word_tokenize(preprocessed_sentence)
        keywords = [lemmatizer.lemmatize(word) for word in preprocessed_tokens if not word in stop_words and not word in punctuations]
        preprocessed_sentence = ' '.join(keywords)
        preprocessed_sentences.append(preprocessed_sentence)
    return preprocessed_sentences

def extract_domain_network_nodes(args):
    '''
    This function combines users and words related to each record
    '''
    tweet_folder = args.path + args.dataset + '/' + args.dataset + '_tweets/'
    users_dict = defaultdict(lambda: [])

    # read graph indicators
    indicator = defaultdict(lambda: -1)
    f = open(args.path + args.dataset + '/' + args.dataset + '_graph_indicator.txt', 'r')
    no_graphs = 0
    for i, line in enumerate(f):
        g_id = int(line.strip())-1
        indicator[i] = g_id
        if g_id+1 > no_graphs:
            no_graphs = g_id+1
    f.close()

    # read user
    f = open(args.path + args.dataset + '/' + args.dataset + '_node_labels.txt', 'r')
    for i, line in enumerate(f):
        splits = line.strip().split(',')
        if int(splits[-1]) == -1:
            continue
        tweet_name = str(splits[-1].strip())
        try:
            tweet_file = json.load(open(tweet_folder + tweet_name + '.json', 'r'))
            user = str(tweet_file["user"]["id"])
            users_dict[indicator[i]].append(user)
        except:
            continue
    f.close()

    # compile words and users
    texts = []
    users = []
    f = open(args.path + args.dataset + '/' + args.dataset + '_text.txt', 'r')
    for i, line in enumerate(f):
        users.append(' '.join(users_dict[i]))
        texts.append(line.strip())
    texts = preprocess_text(texts)

    node_sets = []
    for i, item in enumerate(texts):
        node_sets.append(item + ' ' + users[i])
    return node_sets

def load_graphs(args):
    '''
    This function generates propagation network for each news record
    '''

    # read graph indicators
    indicator = defaultdict(lambda: -1)
    f = open(args.path + args.dataset + '/' + args.dataset + '_graph_indicator.txt', 'r')
    for i, line in enumerate(f):
        indicator[i] = int(line.strip())-1
    f.close()

    # read graph labels
    labels = []
    f = open(args.path + args.dataset + '/' + args.dataset + '_graph_labels.txt', 'r')
    for i, line in enumerate(f):
        splits = line.strip().split(',')
        labels.append(int(splits[0]))
    f.close()

    # read source nodes
    source_nodes = defaultdict(lambda: None)
    f = open(args.path + args.dataset + '/' + args.dataset + '_node_labels.txt', 'r')
    for i, line in enumerate(f):
        splits = line.strip().split(',')
        if int(splits[-1]) == -1:
            source_nodes[indicator[i]] = i
    f.close()

    # read edges
    edges = defaultdict(lambda: [])
    childs = defaultdict(lambda: [])
    f = open(args.path + args.dataset + '/' + args.dataset + '_A.txt', 'r')
    for i, line in enumerate(f):
        items = [int(item) for item in line.strip().split(',')[:2]]

        assert indicator[items[0]-1] == indicator[items[1]-1]
        edges[indicator[items[0]-1]].append((items[0]-1, items[1]-1))
        childs[items[0]-1].append(items[1]-1)
    f.close()

    # construct graphs
    graphs = list()
    for g_id in range(len(labels)):
        G = nx.Graph()
        for e in edges[g_id]:
            if e[0] not in G.nodes():
                G.add_node(e[0])
            if e[1] not in G.nodes():
                G.add_node(e[1])
            G.add_edge(e[0], e[1])
        graphs.append(G)
    return graphs, source_nodes, childs

def compute_depth(g, source):
    '''
    This utility function computes maximum depth of a given
    propagation network from the source node
    '''
    T = nx.dfs_edges(g, source)
    visited_nodes = defaultdict(lambda:-1)
    max_depth = 0
    traverse_order = [-2]
    try:
        for e in list(T):
            if visited_nodes[e[0]] == -1:
                traverse_order.append(e[0])
                visited_nodes[e[0]] = len(traverse_order)-1
            else:
                traverse_order = traverse_order[:visited_nodes[e[0]]]

            if len(traverse_order) > max_depth:
                max_depth = len(traverse_order)
        return max_depth-1
    except:
        return 0

def compute_max_outdegree(g, childs):
    '''
    This utility function computes maximum outdegree of a given
    propagation network
    '''
    max_degree = 0
    for n in g.nodes:
        if len(childs[n]) > max_degree:
            max_degree = len(childs[n])
    return max_degree

def compute_graph_level_features(graphs, childs, sources, levels = 5):
    hop_nbrs = []
    branch_factors = []
    for i, g in enumerate(graphs):
        # compute number of neigbours
        temp_y = [0]*levels
        prev_parents = [sources[i]]
        for k in range(levels):
            current_parents = []
            for n in prev_parents:
                current_parents = current_parents + childs[n]
                temp_y[k] += len(childs[n])
            prev_parents = deepcopy(current_parents)
        hop_nbrs.append(np.array(temp_y))

        # compute branching factors
        prev = 1
        temp = []
        for j in temp_y:
            if prev == 0:
                temp.append(0)
            else:
                temp.append(j/prev)
            prev = j
        branch_factors.append(np.array(temp))
    return hop_nbrs, branch_factors

def generate_graph_level_features(graphs, sources, childs):
    features = []
    for i, g in enumerate(graphs):
        temp_fea = []

        temp_fea.append(nx.wiener_index(g)) # wiener index
        temp_fea.append(len(g.nodes())) # network size
        temp_fea.append(compute_depth(g, sources[i])) # maximum depth
        temp_fea.append(compute_max_outdegree(g, childs)) # maximum outdegree
        features.append(np.array(temp_fea))
    hop_nbrs, branch_factors = compute_graph_level_features(graphs, childs, sources)

    graph_level_features = np.array(features)
    graph_level_features = np.concatenate((graph_level_features, hop_nbrs), axis = 1)
    graph_level_features = np.concatenate((graph_level_features, branch_factors), axis = 1)
    return graph_level_features

def generate_node_level_features(args):
    '''
    This function reads [DATASET]_A.txt, [DATASET]_node_attributes.txt files
    and tweets in [DATASET]_tweets/ directory and compile [DATASET]_node_attribues.txt file,
    which consists of node level features for each node in propagation networks
    '''
    node_file = open(args.path + args.dataset + '/' + args.dataset + '_node_labels.txt')
    tweet_folder = args.path + args.dataset + '/' + args.dataset + '_tweets/'

    node2features = defaultdict(lambda: [])
    analyzer = SentimentIntensityAnalyzer()

    for i, line in enumerate(node_file):
        tweet_name = line.strip().split(',')[-1].strip()

        if tweet_name == '-1':
            node2features[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            try:
                tweet_file = json.load(open(tweet_folder + tweet_name + '.json', 'r'))
                node2features[i].append(int(tweet_file["user"]['verified'])) # user verification

                # timestamp of user acccount
                date_time = tweet_file["user"]["created_at"]
                date_object = datetime.strptime(date_time, "%a %b %d %H:%M:%S %z %Y")
                timestamp = datetime.timestamp(date_object)
                node2features[i].append(timestamp)

                node2features[i].append(tweet_file["user"]["followers_count"]) # follower count
                node2features[i].append(tweet_file["user"]["friends_count"]) # friend count
                node2features[i].append(tweet_file["user"]["listed_count"]) # lists count
                node2features[i].append(tweet_file["user"]["favourites_count"]) # favourites count
                node2features[i].append(tweet_file["user"]["statuses_count"]) # status count

                # timestamp of tweet
                date_time = tweet_file["created_at"]
                date_object = datetime.strptime(date_time, "%a %b %d %H:%M:%S %z %Y")
                timestamp = datetime.timestamp(date_object)
                node2features[i].append(timestamp)

                tweet_text = tweet_file["text"]
                vs = analyzer.polarity_scores(tweet_text)
                # sentiment score
                if vs['compound'] >= 0.05:
                    node2features[i].append(1)
                elif vs['compound'] <= -0.05:
                    node2features[i].append(-1)
                else:
                    node2features[i].append(0)
                node2features[i].append(vs['pos']) # propotion of positive words
                node2features[i].append(vs['neg']) # propotion of negative words

                node2features[i].append(int(tweet_text.count('#'))) # number hash tags
                node2features[i].append(int(tweet_text.count('@'))) # number of mentions
            except:
                print('tweet file for node {} is not available or incomplete'.format(i))
                node2features[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    adj_file = open(args.path + args.dataset + '/' + args.dataset + '_A.txt')
    parent_node = defaultdict(lambda: None)
    child_nodes = defaultdict(lambda: [])
    for line in adj_file:
        splits = line.strip().split(',')
        parent_node[int(splits[1])-1] = int(splits[0])-1
        child_nodes[int(splits[0])-1].append(int(splits[1])-1)

    keys = list(node2features.keys())
    for item in keys:
        try:
            if np.sum(np.array(node2features[item])) == 0:
                node2features[item] = node2features[item] + [0, 0]
            else:
                parent = parent_node[item]
                childs = child_nodes[item]

                # weak label for news timestamp
                min_timestamp = node2features[parent][7]
                if min_timestamp == 0:
                    min_timestamp = node2features[item][7]

                    for c in child_nodes[parent]:
                        if node2features[c][7] < min_timestamp:
                            min_timestamp = node2features[c][7]
                node2features[item].append(node2features[item][7]-min_timestamp)

                child_score = 0
                for c in child_nodes[item]:
                    child_score += (node2features[c][7] - node2features[item][7])
                if child_score:
                    child_score = child_score/len(child_nodes[item])
                node2features[item].append(child_score)
        except:
            print('isolate node ({}) is found in the graph'.format(str(item)))

    f = open(args.path + args.dataset + '/' + args.dataset + '_node_attributes.txt', 'w')
    node_level_features = []
    for key in node2features:
        f.write(','.join([str(item) for item in node2features[key]]) + '\n')
        node_level_features.append(np.array([float(item) for item in node2features[key]]))
    return np.array(node_level_features)
