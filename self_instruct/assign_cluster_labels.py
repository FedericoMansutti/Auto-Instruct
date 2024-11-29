from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import argparse
from collections import Counter
import json
import re
import string
import nltk
import umap
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import StandardScaler
import numpy as np
from wordcloud import WordCloud
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from openai import OpenAI
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from sklearn.metrics import silhouette_score


def process(text, lemmatizer=nltk.WordNetLemmatizer()):

    """ 
    Normalizes case and handles punctuation

    Parameters
    ----------
    text: str: 
        raw text
    lemmatizer: nltk.WordNetLemmatizer() 
        an instance of a class implementing the lemmatize() method
        (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    
    Returns
    -------
    list(str)
        tokenized text
    """

    posMapping = {
        "N":'n',
        "V":'v',
        "J":'a',
        "R":'r'
    }


    text = text.lower()
    text = re.sub(r"http:/+[^ ]*", '', text)
    text = re.sub(r"https:/+[^ ]*", '', text)
    text = re.sub(r"www\.+[^ ]*", '', text)

    #remove punctuation
    text = re.sub(r"'s\b", '', text)
    text = re.sub(r"'", '', text)
    
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')

    split_text = text.split()
    pos_text = nltk.pos_tag(split_text)
    lemmatized_text = []
    for word, pos in pos_text:
        if pos[0] in posMapping:
            lemmatized_text.append(lemmatizer.lemmatize(word, posMapping[pos[0]]))
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word, 'n'))

    return lemmatized_text

def get_common_words():
    """
    Returns a set of common English connectives, articles, and prepositions
    that should be filtered out from text analysis.

    Returns
    -------
    set(str)
        Set of common words to filter
    """
    return {
        'the', 'and', 'or', 'but', 'nor', 'for', 'yet', 'so',
        'a', 'an', 'in', 'on', 'at', 'to', 'of', 'with', 'by',
        'as', 'from', 'into', 'during', 'including', 'until',
        'against', 'among', 'throughout', 'despite', 'towards',
        'upon', 'concerning', 'about', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'shall', 'should', 'may', 'might',
        'must', 'can', 'could', 'i', 'you', 'he', 'she', 'it', 
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'that'
    }


def filter_words(tokenized_text:list, words_to_filter:set):
    """
    Returns a tokens list with the words in `words_to_filter`
    filtered out

    Parameters
    ----------
    tokenized_text : list(str)
        List of text tokens
    words_to_filter : set(str)
        Set of words to filter out

    Returns
    -------
    list(str)
        List of text tokens with words in
        `words_to_filter` filtered out
    """

    return [token for token in tokenized_text if token not in words_to_filter]


def create_features_tfidf(instructions:pd.DataFrame):
    """
    Compute TF-IDF features for the instructions dataset

    Parameters
    ----------
    instructions : pd.DataFrame
        Dataframe with a column named 'instruction'
        containing list of instructions
    
    Returns
    -------
    TfidfVectorizer()
        Instance of the class TfidfVectorizer
    scipy.sparse._csr_matrix
        TF-IDF feature matrix
    """

    vectorizer = TfidfVectorizer(
        min_df=2, 
        tokenizer=lambda x: x,
        lowercase=False,
    )

    tfidf_matrix = vectorizer.fit_transform(instructions['instruction'])
    
    return tfidf_matrix, vectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key for GPT analysis"
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="GPT engine"
    )
    return parser.parse_args()

def plot_silhouette_scores(silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(list(n_clusters_range), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    # Add a vertical line at the optimal number of clusters
    plt.axvline(x=optimal_n_clusters, color='r', linestyle='--', 
                label=f'Optimal clusters: {optimal_n_clusters}')
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(args.batch_dir, 'silhouette_scores.png'))
    plt.close()








if __name__ == '__main__':
    args = parse_args()

    # load the instructions
    with open(os.path.join(args.batch_dir, "is_clf_or_not_gpt-4o_template_1.jsonl"), encoding="utf-8") as fin:
        loaded = [json.loads(line) for line in fin]
        lines = [line["instruction"] for line in loaded if line["is_classification"]=="No"]

    '''
    output_path = os.path.join(args.batch_dir, f"cluster_assignments.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")
    '''

    # lemmatize the instructions
    lemmatized_lines = [process(line) for line in lines]

    # filter out common words
    common_words = get_common_words()
    filtered_lines = [filter_words(line, common_words) for line in lemmatized_lines]

    
    dmeasure = 'euclidean' # distance metric
    rdims    = 6 # r-dims == Reduced dimensionality
    print(f"UMAP dimensionality reduction to {rdims} dimensions with '{dmeasure}' distance measure.")


    instructions_df = pd.DataFrame({'instruction': filtered_lines})

    tfidf_matrix, vectorizer = create_features_tfidf(instructions_df)

    # Apply UMAP
    reducer = umap.UMAP(
        n_components=rdims,
        metric=dmeasure,
        random_state=42  # for reproducibility
    )
    embedding = reducer.fit_transform(tfidf_matrix)

    embedding_df = pd.DataFrame(
        embedding,
        columns=[f'UMAP_{i+1}' for i in range(rdims)]
    )


    scaler = StandardScaler()
    scaled_embedding = scaler.fit_transform(embedding)

    # Ensure data is in the correct format
    scaled_embedding = np.array(scaled_embedding, dtype=np.float64)

     # Try different numbers of clusters and compute silhouette scores
    n_clusters_range = range(2, 51)  # Test from 2 to 50 clusters
    silhouette_scores = []
    
    print("Computing silhouette scores for different cluster counts...")
    for n_clusters in tqdm(n_clusters_range):
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            covariance_type='full',
            n_init=5
        )
        clusters = gmm.fit_predict(scaled_embedding)
        score = silhouette_score(scaled_embedding, clusters)
        silhouette_scores.append(score)

    # Find optimal number of clusters
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")

    plot_silhouette_scores(silhouette_scores)

    # Apply GMM with optimal number of clusters
    gmm = GaussianMixture(
        n_components=optimal_n_clusters,
        random_state=42,
        covariance_type='full',
        n_init=5
    )
    clusters = gmm.fit_predict(scaled_embedding)

    # Create a figure with subplots for each cluster
    unique_clusters = sorted(np.unique(clusters))

    output_path = os.path.join(args.batch_dir, f"cluster_assignments.jsonl")
    with open(output_path, 'w', encoding='utf-8') as fout:
        for cluster_id in unique_clusters:
            cluster_mask = (clusters == cluster_id)
            label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            # Get original instructions (before lemmatization)
            cluster_examples = [line for i, line in enumerate(lines) if cluster_mask[i]]


            numbered_examples = "\n".join(f"{i+1}. {example.strip()}" for i, example in enumerate(cluster_examples[:100]))
            prompts = [numbered_examples + "\nFind the common topic of the previous instructions. Answer with no preamble."]
            results = make_gpt3_requests(
                        engine=args.engine,
                        prompts=prompts,
                        max_tokens=100,
                        temperature=0,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop_sequences=["\n", "Task"],
                        n=1,
                        api_key=args.api_key)
            
            
            for example in cluster_examples:
                data = {
                    "instruction": example,
                    "topic": results[0]["response"]
                }
                fout.write(json.dumps(data, ensure_ascii=True) + "\n")
                fout.flush()