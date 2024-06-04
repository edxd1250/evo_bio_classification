
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


def compute_tfidf(corpus, vocab):
    '''
    Compute TF-IDF scores for given corpus and dictionary of words.

    Parameters:
    corpus (list of str): List of text documents/abstracts.
    vocab (list of str): Dictionary of words for desired tf-idf scores.

    Returns:
    sparse_matrix (scipy.sparse.csr.csr_matrix): Sparse matrix containing corresponding corpus TF-IDF scores for the given vocab.
    feature_names (list of str): List of words corresponding to columns of the sparse matrix.
    
    '''
    vocab = [x.lower() for x in vocab]
    # Initialize the TfidfVectorizer with the given dictionary
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    
    # Fit and transform the corpus using the vectorizer
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Get the feature names (words in the dictionary)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names

def main():
    parser = argparse.ArgumentParser(description= "Generate TF-IDF matrix for given corpus and matrix")

    #Add arguments for input files and output file
    parser.add_argument("-c", "--corpus", help="Path to cleaned text corpus")
    parser.add_argument("-v", "--vocab", help="Path to vocabulary")
    parser.add_argument("-o", "--output", help="Path to the output dir")
    parser.add_argument("-n", "--name", help="<Optional> naming scheme", default='')

    args = parser.parse_args()
    corpus_path = Path(args.corpus)
    vocab_path = Path(args.vocab)
    output_dir = Path(args.output)
    name = str(args.name)
    matrix_output = output_dir/f'{name}_tf_idf_matrix.pickle'
    terms_output = output_dir/f'{name}_output_terms.pickle'

    with open(vocab_path, 'r') as file:
        vocab = [line.strip() for line in file.readlines()]

    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    matrix, terms = compute_tfidf(corpus, vocab)

    with open(matrix_output, "wb") as f:
        pickle.dump(matrix,f)
    print(f"Matrix saved at: {matrix_output}")

    with open(terms_output, "wb") as f:
        pickle.dump(terms,f)
    print(f'Matrix terms saved at: {terms_output}')
    

        
    

if __name__ == '__main__':
    main()