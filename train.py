import numpy as np
import random
import argparse
from collections import defaultdict

from sklearn.model_selection import train_test_split

from model import FAKE_NEWS_DETECTOR

# LSH-based instance selection
def lsh_data_selection(domain_embeddings, labelling_budget=100, num_h = 10):
    size = domain_embeddings.shape[1]
    final_selected_ids = []
    is_final_selected = defaultdict(lambda: False)

    random_distribution = [np.sqrt(3), 0, 0, 0, 0, -np.sqrt(3)]

    while len(final_selected_ids) < labelling_budget:
        # generate random vectors
        random_vectors =[]
        for hash_run in range(num_h):
            vec = random.choices(random_distribution, k=size)
            random_vectors.append(np.array(vec))

        # hash vectors
        code_dict = defaultdict(lambda: [])
        for i, item in enumerate(domain_embeddings):
            code = ''
            # skip if the item is already selected
            if is_final_selected[i]:
                continue

            for code_vec in random_vectors:
                code = code + str(int(np.dot(item, code_vec)>0))
            code_dict[code].append(i)

        selected_ids = []
        is_selected = defaultdict(lambda: False)
        for item in code_dict:
            # pick item from each item bin
            selected_item = random.choice(code_dict[item])
            selected_ids.append(selected_item)
            is_selected[selected_item] = True

        if len(final_selected_ids + selected_ids) > labelling_budget:
            # remove a set of instances randomly to meet the labelling budget
            random_pick_size = labelling_budget - len(final_selected_ids)
            mod_selected_ids = []
            for z in range(random_pick_size):
                select_item = random.choice(selected_ids)
                mod_selected_ids.append(select_item)
                selected_ids.remove(select_item)
            final_selected_ids = final_selected_ids + mod_selected_ids
            return final_selected_ids

        for item in selected_ids:
            is_final_selected[item] = True
        final_selected_ids = final_selected_ids + selected_ids
    return final_selected_ids

def dataset_generator(args):
    '''
    This functions combines cross-domain datasets and returns:
        (1) train_dataset, where each instance is tuple <a,b,c,d>, where a is the input
        multimodal representation, b is the label, c is the domain embedding, d is
        the actual domain label (d is not used for training only for visualization)

        (2) test_datasets, which is a dict that stores testing instances for each dataset,
        and under key 'TOTAL', the whole testing dataset is stored for the evaluation of
        overall performance. Each data instance under the test dataset follows the same
        format as in train_dataset
    '''
    train_dataset = None
    test_datasets = defaultdict(lambda: None)

    for i, dataset in enumerate(args.datasets):
        data_dir = args.path + '/' + dataset + '/' + dataset
        temp_dataset = None
        # construct multimodal input representation (temp_dataset) by concatenating text and network embeddings
        if 'txt' in args.features:
            temp_dataset = np.load(data_dir + '_txt_emb.npy')
        if 'net' in args.features:
            temp_net_dataset = np.load(data_dir  + '_net_emb.npy')
            if temp_dataset is None:
                temp_dataset = temp_net_dataset
            else:
                temp_dataset = np.concatenate((temp_dataset, temp_net_dataset), axis = 1)
        labels = np.load(data_dir + '_labels.npy')
        domain_embeddings = np.load(data_dir + '_domain_emb.npy')
        actual_domain_labels = labels*0 + i

        X_train, X_test, y_train, y_test, dy_train, dy_test, ady_train, ady_test = train_test_split(temp_dataset, labels, domain_embeddings, actual_domain_labels, test_size = 0.25, random_state=args.random_state)
        if train_dataset is None:
            train_dataset = (X_train, y_train, dy_train, ady_train)
        else:
            train_dataset = (np.concatenate((train_dataset[0], X_train), axis = 0), np.concatenate((train_dataset[1], y_train), axis = 0), np.concatenate((train_dataset[2], dy_train), axis = 0), np.concatenate((train_dataset[3], ady_train), axis = 0))
        test_datasets[dataset] = (X_test, y_test, dy_test, ady_test)

    test_datasets['TOTAL'] = test_datasets[args.datasets[0]]
    for i, dataset in enumerate(args.datasets[1:]):
        test_datasets['TOTAL'] = (np.concatenate((test_datasets['TOTAL'][0], test_datasets[dataset][0]), axis = 0), np.concatenate((test_datasets['TOTAL'][1], test_datasets[dataset][1]), axis = 0),
                                 np.concatenate((test_datasets['TOTAL'][2], test_datasets[dataset][2]), axis = 0), np.concatenate((test_datasets['TOTAL'][3], test_datasets[dataset][3]), axis = 0))
    return train_dataset, test_datasets

def train(args):
    train_dataset_pool, test_datasets = dataset_generator(args)

    # select training dataset using lsh-based instance selection
    train_datapool_size = train_dataset_pool[0].shape[0]
    labelling_budget = int(train_datapool_size*args.B)
    selected_ids = lsh_data_selection(train_dataset_pool[2], labelling_budget=labelling_budget, num_h = args.h)

    train_dataset = (train_dataset_pool[0][selected_ids], train_dataset_pool[1][selected_ids],
                    train_dataset_pool[2][selected_ids], train_dataset_pool[3][selected_ids])

    classifier = FAKE_NEWS_DETECTOR(train_dataset[0].shape[1], train_dataset[2].shape[1], 512, lambda1=args.lambda1, lambda2=args.lambda2, lambda3=-args.lambda3)
    classifier.train(train_dataset[0], train_dataset[0], train_dataset[1], train_dataset[1], train_dataset[2], train_dataset[2], epochs = args.epochs, batch_size = args.batch_size)

    for dataset in args.datasets:
        classifier.evaluate(test_datasets[dataset][0], test_datasets[dataset][1], test_datasets[dataset][2])

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="data directory", type=str, default='./DATA/')
    parser.add_argument("--datasets", help='dataset name', type=list, default=['TOY1','TOY2', 'TOY3'])
    parser.add_argument("--features", help='feature types', type=list, default=['txt','net'])
    parser.add_argument("--random_state", help='random state', type=int, default=0)
    parser.add_argument("--B", help='labelling budget as % of training datapool', type=float, default=0.8)
    parser.add_argument("--h", help='number of hashing functions', type=int, default=10)
    parser.add_argument("--lambda1", help='lambda1 hyperparameter', type=float, default=1)
    parser.add_argument("--lambda2", help='lambda2 hyperparameter', type=float, default=10)
    parser.add_argument("--lambda3", help='lambda3 hyperparameter', type=float, default=5)
    parser.add_argument("--d", help='latent dimension size', type=int, default=512)
    parser.add_argument("--epochs", help='number of epochs', type=int, default=300)
    parser.add_argument("--batch_size", help='batch size', type=int, default=64)
    args = parser.parse_args()

    train(args)
