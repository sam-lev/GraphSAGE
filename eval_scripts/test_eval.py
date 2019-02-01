from __future__ import print_function
import json
from networkx.readwrite import json_graph
import numpy as np

import networkx as nx
from argparse import ArgumentParser

''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
python  eval_scripts/ppi_eval.py (dataset_dir) ../example_data (embed_dir) ../unsup-example_data/graphsage_mean_small_0.000010 (setting) test
python  eval_scripts/ppi_eval.py example_data unsup-example_data/graphsage_mean_small_0.000010  test

python  eval_scripts/ppi_eval.py (dataset_dir) ../json_graphs (embed_dir) /Users/multivax/Documents/PhD/Research/topologicalComputing-Pascucci/TopoML/graphSage/GraphSAGE/unsup-json_graphs/graphsage_mean_small_0.000010 (setting) test

python  eval_scripts/ppi_eval.py ../json_graphs /Users/multivax/Documents/PhD/Research/topologicalComputing-Pascucci/TopoML/graphSage/GraphSAGE/unsup-json_graphs/graphsage_mean_small_0.000010 test
'''
import os

def run_regression(train_embeds, train_labels, test_embeds, test_labels, test_node_ids = None, test_graph = None):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier
    dummy = MultiOutputClassifier(DummyClassifier())
    dummy.fit(train_embeds, train_labels)
    log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)

    log.fit(train_embeds, train_labels)
    prediction = log.predict(test_embeds)
    
    f1 = 0
    for i in range(test_labels.shape[1]):
        print("F1 score", f1_score(test_labels[:,i], prediction[:,i], average="binary"))
    for i in range(test_labels.shape[1]):
        print("Random baseline F1 score", f1_score(test_labels[:,i], dummy.predict(test_embeds)[:,i], average="micro"))

    if test_graph:
        prediction_prob={}
        for id, pred in zip(range(len(test_node_ids)), prediction):
            prediction_prob[id] = {'prediction':pred}#, 'prob_predict':prob_predict[id]}
        nx.set_node_attributes(test_graph, prediction_prob)
        print(test_graph.nodes[2])
        if not os.path.exists(  'predicted_graph-G.json'):
            open(  'predicted_graph-G.json', 'w').close()
        with open(  'predicted_graph-G.json', 'w') as graph_file:
            write_form = json_graph.node_link_data(test_graph)
            json.dump(write_form, graph_file)
            

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on TEST data.")
    parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    parser.add_argument("dataset_dir_test", help="Path to directory containing the dataset.")
    parser.add_argument("embed_dir", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("graph_set", help="Either val or test.")
    parser.add_argument("graph_set_test", help="Either val or test.")
    parser.add_argument("label_node_predictions", default=False,  help="Either val or test.")
    parser.add_argument("setting", help="Either val or test.")
    args = parser.parse_args()
    dataset_dir = os.path.join(os.getcwd(),args.dataset_dir)
    dataset_dir_test = os.path.join(os.getcwd(),args.dataset_dir_test)
    data_dir = os.path.join(os.getcwd(),args.embed_dir)
    graph_set = args.graph_set
    graph_set_test =  args.graph_set_test
    label_node_predictions =  args.label_node_predictions
    setting = args.setting
    

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/"+graph_set+"-G.json")))
    G_test = json_graph.node_link_graph(json.load(open(dataset_dir_test + "/"+graph_set_test+"-G.json")))
    labels = json.load(open(dataset_dir +  "/"+graph_set+"-class_map.json"))
    labels = {int(i):l for i, l in labels.items()}#iteritems()} #for python3
    labels_test = json.load(open(dataset_dir_test +  "/"+graph_set_test+"-class_map.json"))
    labels_test = {int(i):l for i, l in labels_test.items()}#iteritems()} #for python3
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G_test.nodes() if G_test.node[n][setting]] #changed here!
    train_labels = np.array([labels[i] for i in train_ids])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels_test[i] for i in test_ids])
    print("running\n", data_dir)

    if data_dir == "feat":
        print("\n", "Using only features..","\n")
        feats = np.load(dataset_dir +  "/"+graph_set+"-feats.npy")
        feats_test = np.load(dataset_dir_test +  "/"+graph_set_test+"-feats.npy")
        
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:,0] = np.log(feats[:,0]+1.0)
        feats[:,1] = np.log(feats[:,1]-min(np.min(feats[:,1]), -1))
        feat_id_map = json.load(open(dataset_dir +  "/"+graph_set+"-id_map.json"))
        feat_id_map = {int(id):val for id,val in feat_id_map.items()}#iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]

        feats_test[:,0] = np.log(feats_test[:,0]+1.0)
        feats_test[:,1] = np.log(feats_test[:,1]-min(np.min(feats_test[:,1]), -1))
        feat_id_map_test = json.load(open(dataset_dir_test +  "/"+graph_set_test+"-id_map.json"))
        feat_id_map_test = {int(id):val for id,val in feat_id_map_test.items()}#iteritems()}
        test_feats_test = feats_test[[feat_id_map_test[id] for id in test_ids_test]]
        
        print("\n","Running regression..","\n")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats_test)
        if not label_node_predictions:
            run_regression(train_feats, train_labels, test_feats, test_labels)
        else:
            run_regression(train_feats, train_labels, test_feats, test_labels, test_ids, G_test)
    else:
        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]] 
        test_embeds = embeds[[id_map[id] for id in test_ids]] 

        print("Running regression..")
        if not label_node_predictions:
            run_regression(train_embeds, train_labels, test_embeds, test_labels)
        else:
            run_regression(train_embeds, train_labels, test_embeds, test_labels, test_ids, G_test)
