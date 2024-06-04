
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pd


def create_corpus(dataset, journals, cleans):
    '''
    Create subset of given dataset using list of given journal names

    Parameters:
    dataset (pandas df): Dataframe of pubmed articles. Dataframe must be in the format of previously parsed 
                        pubmed articles.
    journals (list of str): List of journals to be filtered. Names must specifically match Pubmed journal titles. 
                            For more information see classification.ipynb
    cleans (list of str): List of text preprocessed articles corresponding with given dataset.

    Returns:
    filtered_data (pandas df): Subset of given dataset containing only articles in given journal list
    indices_of_filtered_data (list of int): List of indicies of selected articles. Useful for matching with cleans or embeddings
    new_clean: Subset of cleans corresponding with filtered_data

    
    '''
    filtered_data = dataset[dataset['Journal'].isin(journals)]
    indices_of_filtered_data = filtered_data.index.tolist()
    new_clean = [cleans[x] for x in indices_of_filtered_data]
    return filtered_data, indices_of_filtered_data, new_clean

def main():
    parser = argparse.ArgumentParser(description= "Generate TF-IDF matrix for given corpus and matrix")

    #Add arguments for input files and output file
    parser.add_argument("-d", "--dataset", help="Path to dataset")
    parser.add_argument("-c", "--corpus", help="Path to cleaned text corpus")
    parser.add_argument("-o", "--output", help="Path to the output dir")
    parser.add_argument("-n", "--name", help="<Optional> naming scheme", default='')

    args = parser.parse_args()
    corpus_path = Path(args.corpus)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    name = str(args.name)

    #Hardcoded journal list from classification.ipynb
    journals_dir = '/mnt/home/ande2472/data/full_clean_data/journalswISSN.csv'
    journals = pd.read_csv(journals_dir, sep='\t')
    issnlist = ['15452069','14712148','20419139','00143820','1525142X','20563744','17524571','15738477','2296701X','14209101','14321432','2041210X','15371719','1365294X','10557903','2397334X','1076836X','18728383','17596653','15375323','10960325']
    journalslist = list(journals[journals['issn'].isin(issnlist)]['Journal'])

    filtered_output = output_dir/f'{name}_filtered_data.tsv'
    indicies_output = output_dir/f'{name}_filtered_indicies.pickle'
    clean_output = output_dir/f'{name}_filtered_cleans.pickle'

    dataset = pd.read_csv(dataset_path, sep='\t')

    with open(corpus_path, "rb") as f:
        cleans = pickle.load(f)

    new_data, indicies, new_cleans = create_corpus(dataset, journalslist, cleans)

    with open(indicies_output, "wb") as f:
        pickle.dump(indicies,f)
    print(f"Indicies saved at: {indicies_output}")

    with open(clean_output, "wb") as f:
        pickle.dump(new_cleans,f)
    print(f'Cleans saved at: {clean_output}')

    new_data.to_csv(filtered_output, sep='\t')
    print(f'Dataset saved at: {filtered_output}')
    

        
    

if __name__ == '__main__':
    main()