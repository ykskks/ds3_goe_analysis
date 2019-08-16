import argparse 
import collections
import itertools
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import PyPDF2


def clean_texts(input_string):
    for unwanted in unwanteds:
        input_string = input_string.replace(unwanted, "")
        
    return porter.stem(input_string.lower())


def pdf2texts(base_path, pdf_paths):
    # list that will store list of words in each slide
    all_pages = []
    for pdf_path in pdf_paths:
        pdfFileObj = open(base_path + pdf_path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        NUM_PAGES = pdfReader.numPages
        
        pages = []

        for page in range(NUM_PAGES):
            pageObj = pdfReader.getPage(page)
            # clean raw texts
            cleaned_word_list = [clean_texts(s) for s in pageObj.extractText().split() if (clean_texts(s) not in stop_words) & (not re.match("^.$", s)) & (not re.match("^[0-9]{2}$", s)) & (not any(unwanted in s for unwanted in unwanteds))]
            cleaned_word_list = [s for s in cleaned_word_list if s]
            pages.append(cleaned_word_list)
            
        all_pages.extend(pages)

    return all_pages


def get_jacaad_coef(all_pages):
    # get combinations of words in each page
    pages_combi = [list(itertools.combinations(page, 2)) for page in all_pages]
    all_combi = []
    for page_combi in pages_combi:
        all_combi.extend(list(set(page_combi)))

    combi_count = collections.Counter(all_combi)

    # make dataframe that holds pair of words and number of pages they appeared together
    word_associates = []
    for key, value in combi_count.items():
        word_associates.append([key[0], key[1], value])

    word_associates = pd.DataFrame(word_associates, columns=['word1', 'word2', 'intersection_count'])

    # number of pages each word appeared 
    target_words = []
    for page in all_pages:
        target_words.extend(page)

    word_count = collections.Counter(target_words)
    word_count = [[key, value] for key, value in word_count.items()]
    word_count = pd.DataFrame(word_count, columns=['word', 'count'])

    # combine them together
    word_associates = pd.merge(word_associates, word_count, left_on='word1', right_on='word', how='left')
    word_associates.drop(columns=['word'], inplace=True)
    word_associates.rename(columns={'count': 'count1'}, inplace=True)
    word_associates = pd.merge(word_associates, word_count, left_on='word2', right_on='word', how='left')
    word_associates.drop(columns=['word'], inplace=True)
    word_associates.rename(columns={'count': 'count2'}, inplace=True)

    word_associates['union_count'] = word_associates['count1'] + word_associates['count2'] - word_associates['intersection_count']
    word_associates['jaccard_coefficient'] = word_associates['intersection_count'] / word_associates['union_count'] 

    return word_associates  


def plot_network(data, edge_threshold=0., fig_size=(15, 15), file_name=None, dir_path=None):

    nodes = list(set(data['node1'].tolist()+data['node2'].tolist()))

    G = nx.Graph()
    # each word as a node
    G.add_nodes_from(nodes)

    # add edges that have bigger coef than threshold
    for i in range(len(data)):
        row_data = data.iloc[i]
        if row_data['value'] > edge_threshold:
            G.add_edge(row_data['node1'], row_data['node2'], weight=row_data['value'])

    # remove node with no edges
    isolated = [n for n in G.nodes if len([i for i in nx.all_neighbors(G, n)]) == 0]
    for n in isolated:
        G.remove_node(n)

    plt.figure(figsize=fig_size)
    pos = nx.spring_layout(G, k=2)  

    pr = nx.pagerank(G)

    nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()),
                           cmap=plt.cm.Reds,
                           alpha=0.7,
                           node_size=[60000*v for v in pr.values()])

    nx.draw_networkx_labels(G, pos, fontsize=14, font_family='IPAexGothic', font_weight="bold")

    # edge width
    edge_width = [d["weight"] * 5 for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="orange", width=edge_width)

    plt.axis('off')
    plt.show()

    # save image
    if file_name is not None:
        if dir_path is None:
            dir_path = Path('.').joinpath('image')
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        plt.savefig(dir_path.joinpath(file_name), bbox_inches="tight")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--n_word_lower", help="minimum number of pages that a word appears in")    
    parser.add_argument("--edge_threshold", help="minimum jacarrd coefficient that word paors should have to appear in network")
    parser.add_argument("--file_name", help="file name to be saved")
    args = parser.parse_args()   
    n_word_lower = int(args.n_word_lower)
    edge_threshold = float(args.edge_threshold)
    file_name = args.file_name

    # paths to pdfs to be used in analysis
    BASE_PATH = "./data/input/raw/"
    pdf_paths = [path for path in os.listdir(BASE_PATH) if "ipynb" not in path]


    # stemmer for standardizing and define stop words
    porter = PorterStemmer()
    stop_words = stopwords.words('english')
    stop_words.extend(["", " ", "dr.", "marcus", "baum", "prof.", "ing.", "seit"])

    # unwanted symbols 
    unwanteds = ["(", ")", "!", "?", "¥", ">", "&", "#", " ", ",", ":", "...", "/", "$", "%", '""', "˛", "ˆˆ" ]

    # plot
    all_pages = pdf2texts(BASE_PATH, pdf_paths)
    word_associates = get_jacaad_coef(all_pages)
    word_associates.query('count1 >= @n_word_lower & count2 >= @n_word_lower', inplace=True)
    word_associates.rename(columns={'word1':'node1', 'word2':'node2', 'jaccard_coefficient':'value'}, inplace=True)
    plot_network(data=word_associates, edge_threshold=edge_threshold, file_name=file_name)







