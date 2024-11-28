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
import seaborn as sns
import matplotlib.pyplot as plt

posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

# 5% Credits
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

    # [Your Code Starts Here]

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
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    return parser.parse_args()


def top_25_hf_words(instructions_ser:pd.Series):
    """
    Returns the top 25 most commonly occurring words
    across all instructions in a series object containing
    a list of instruction texts

    Parameters
    ----------
    instructions_ser : pd.Series
        Series objects containing a list of instructions

    Returns
    -------
    set(str)
        Set of top 25 high frequency words
    """

    word_counts = Counter()

    for instruction in instructions_ser:
        word_counts.update(instruction)

    most_common = word_counts.most_common(25)

    return set(word for word, count in most_common)



args = parse_args()

with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
    lines = [json.loads(line)['instruction'] for line in fin]
    if args.num_instructions is not None:
        lines = lines[:args.num_instructions]

lemmatized_lines = [process(line) for line in lines]
top_words = top_25_hf_words(pd.Series(lemmatized_lines))
print("Top 25 most frequent words:", top_words)


dmeasure = 'euclidean' # distance metric
rdims    = 4 # r-dims == Reduced dimensionality
print(f"UMAP dimensionality reduction to {rdims} dimensions with '{dmeasure}' distance measure.")


instructions_df = pd.DataFrame({'instruction': lemmatized_lines})

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

projected = instructions_df.join(embedding_df).sort_values(by=['UMAP_1','UMAP_2'])

print(projected.head())

def tune_figure(ax, title:str='Title'):
    ax.axis('off')
    ax.set_title(title)
    ax.get_legend().set_title("")
    ax.get_legend().prop.set_family('Times New Roman')
    ax.get_legend().prop.set_size(12)
    ax.get_legend().get_frame().set_linewidth(0.0)
    
f, axs = plt.subplots(1, 1, figsize=(10, 6))
sns.scatterplot(data=projected, x='UMAP_1', y='UMAP_2', s=5, alpha=0.1)
plt.title('UMAP Projection of Instructions')
plt.show()