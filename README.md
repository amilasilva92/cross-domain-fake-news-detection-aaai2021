# Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data
[![GitHub](https://img.shields.io/github/license/amilasilva92/multilingual-communities-by-code-switching?style=plastic)](https://opensource.org/licenses/MIT)

This repository provides the source and the formats of the data files to reproduce the results in the following paper.

```
Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data
Amila Silva, Ling Luo, Shanika Karunasekera, Christopher Leckie
In Proceedings of the AAAI Conference on Artificial Intelligence, 2021 (AAAI2021)
```

Abstract: With the rapid evolution of social media, fake news has become a significant social problem, which cannot be addressed in a timely manner using manual investigation. This has motivated numerous studies on automating fake news detection. Most studies explore supervised training models with different modalities (e.g., text, images, and propagation networks) of news records to identify fake news. However, the performance of such techniques generally drops if news records are coming from different domains (e.g., politics, entertainment), especially for domains that are unseen or rarely-seen during training. As motivation, we empirically show that news records from different domains have significantly different word usage and propagation patterns. Furthermore, due to the sheer volume of unlabelled news records, it is challenging to select news records for manual labelling so that the domain-coverage of the labelled dataset is maximized. Hence, this work: (1) proposes a novel framework that jointly preserves domain-specific and cross-domain knowledge in news records to detect fake news from different domains; and (2) introduces an unsupervised technique to select a set of unlabelled informative news records for manual labelling, which can be ultimately used to train a fake news detection model that performs well for many domains while minimizing the labelling cost. Our experiments show that the integration of the proposed fake news model and the selective annotation approach achieves state-of-the-art performance for cross-domain news datasets, while yielding notable improvements for rarely-appearing domains in news datasets.

##### Datasets 
This paper uses three publicly availabe fake news datasets from three different domains. Due to the restrictions in Twitter, we cannot share the crawled tweet messages. Please use the following links to download the datasets.

PolitiFact :
[https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

GossipCop :
[https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

CoAID :
[https://github.com/cuilimeng/CoAID](https://github.com/cuilimeng/CoAID)

To show the inputted formats for these three dataset, the /DATA/ directory consists of three toy datasets: TOY1, TOY2, and TOY3. ***These toy datasets are not representative subsets of the original datasets. We use them only to show the input format of the datasets. Thus, they are not suitable to run the experiments.*** The structure of the /DATA/ directory should be as follows.


![](https://i.imgur.com/iHj3PIj.png)

For each dataset XXX, there should be a directory in DATA/ under its name (XXX). This directory should include the following files:
* XXX_graph_labels.txt - i^th^ line in this file gives the fake news label of the i^th^ instance in the dataset XXX 
* XXX_text.txt - i^th^ line in this file gives the news title of the i^th^ instance in the dataset XXX
* XXX_node_labels.txt - Each line in this file represent a node in the propagation networks of the dataset XXX. Each line has two entries separated by a comma and the first entry is the index of the node and the second entry shows the type of the node: if the node represent a source node (i.e., news record) in a propagation network, then the value is -1; otherwise the node is a tweet in a propagation network,  then the value gives the tweet id of the corresponding tweet. 
* XXX_A.txt - Each line in this file represent an edge in a propagation network of the XXX dataset. The first two entries (separated by) in each line represents the source node id and the destination node id of an edge.
* XXX_graph_indicator.txt - i^th^ line in this file shows the graph (i.e., news record) that i+1^th^ node belongs to 
* XXX_tweets/ - This directory includes all the tweets files in XXX_node_labels.txt, stored as json files and named under the twitter ids.
 
Please see the toy datasets in the DATA/ directory for an example.

After the datasets are stored in the DATA/ directory, please follow the below steps to run the model. 

##### Instructions to Run 
1. To install required libraries (Note: The code is written in python 3 and requires GPU to run RoBERTa pretrained model).
```shell=
pip install -r requirements.txt
```

2. To run the whole pipline (preprocessing, domain discovery, training and evaluation) of the model:
```shell=
python main.py --path [PATH TO DATA DIRECTORY] --datasets [LIST OF DATASETS]
```

3. To run each step in the pipline separatly:
    3.1 To preprocess and construct the multi-modal input representations
    ```shell=
    python preprocess.py --path [PATH TO DATA DIRECTORY] --dataset [DATASET NAME]
    ```
    This script produces three files in the corresponding data directory ([PATH TO DATA DIRECTORY] + [DATASET NAME]): (1) [DATASET NAME]_txt_emb.npy - text-based rerpesentations of the news recrods; (2) [DATASET NAME]_net_emb.npy - network-based representations of the news recrods; (3) [DATASET NAME]_labels.npy - labels of the news records.
    3.2 To generate the domain embeddings of the news records:
    ```shell
    python domain_discovery.py --path [PATH TO DATA DIRECTORY] --datasets [LIST OF DATASETS]
    ```
    This script produces another additional file under each dataset's directory: [DATASET NAME]_domain_emb.npy - domain embeddings of the news.
    3.3 To train and evaluate the model:
    ```shell=
    python train.py --path [PATH TO DATA DIRECTORY] --datasets [LIST OF DATASETS]
    ```
    
##### Hyperparameters
Here we describe the hyperparameters of the model, with some advice in how to set them.

[features] 
What is it: Defines the list of modalities (e.g., text, network) that should be considered to construct the input representation. The possible items for this list are 'txt' (for the text modality) and 'net' (for the network modality).
Default: ['txt', 'net']

[random_state]
What is it: Defines the random state of the training and testing dataset splitting process. To reproduce the results in the paper, please run the experiments using 0, 1, 2 random_state values and average the results.
Default: 0 

[B]
What is it: Defines the labelling budget as a percentage of the training dataset pool size. This value can take any float number in the range of (0, 1].
Default: 1.0


[h]
What is it: Defines the number of hash functions used for the LSH-based instance selection approach. As shown in our parameter sensitivity study, the integer values between [10, 15] work well for this parameter. 
Default: 10

[lambda1]
What is it: Defines the importance assigned to the input reconstruction-based loss term. Please see our parameter sensitivity study for more details about the optimal range.
Default: 1.0

[lambda2]
What is it: Defines the importance assigned to the domain specific loss term. Please see our parameter sensitivity study for more details about the optimal range.
Default: 10.0

[lambda3]
What is it: Defines the importance assigned to the domain shared loss term. Please see our parameter sensitivity study for more details about the optimal range.
Default: 5.0

[d]
What is it: Defines the size of the latent space. Please see our parameter sensitivity study for more details about the optimal range.
Default: 512

[epochs]
What is it: Defines the number of epochs. Please see our parameter sensitivity study for more details about the optimal range.
Default: 300

[batch_size]
What is it: Defines the size of a batch. Please see our parameter sensitivity study for more details about the optimal range.
Default: 64
    
NOTE: The default values of the hyperparameters can be changed by passing as a conventional argument as shown below.
```shell=
python main.py --path [PATH TO DATA DIRECTORY] --datasets [LIST OF DATASETS] --random_state 1 --lambda3 10
```


