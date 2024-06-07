
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pd
import yaml


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

def create_neg_set(dataset, journals, cleans, n):
    filtered_data = dataset[-dataset['Journal'].isin(journals)]
    sampled_data = filtered_data.sample(n=n)
    indices_of_sampled_data = sampled_data.index.tolist()
    new_clean = [cleans[x] for x in indices_of_sampled_data]
    return filtered_data, indices_of_sampled_data, new_clean
    

def get_args():
  """Get command-line arguments"""
  parser = argparse.ArgumentParser(
    description='Retrain BERT using the plant science history project corpus',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-c', '--config_file', 
                      type=str,
                      help='Config file path',
                      default='./config.yaml')

  args = parser.parse_args()
  
  return args

def main():
    parser = argparse.ArgumentParser(description= "Generate TF-IDF matrix for given corpus and matrix")

    #Add arguments for input files and output file
    # parser.add_argument("-d", "--dataset", help="Path to dataset")
    # parser.add_argument("-c", "--corpus", help="Path to cleaned text corpus")
    # parser.add_argument("-o", "--output", help="Path to the output dir")
    # parser.add_argument("-n", "--name", help="<Optional> naming scheme", default='')
    #args = parser.parse_args()
    corpus_path = Path(config['env']['clean_path'])
    dataset_path = Path(config['env']['dataset_path'])
    output_dir = Path(config['env']['output_dir'])
    name = config['create_corpus']['run_name']

    print("Get config")
    args        = get_args()
    config_file = Path(args.config_file)  # config file path
    print(f"  config_file: {config_file}\n")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    corpus_path = Path(config['env']['clean_path'])
    dataset_path = Path(config['env']['dataset_path'])
    output_dir = Path(config['env']['output_dir'])
    journals_dir = Path(config['env']['journals_dir'])
  

    issnlist = config['create_corpus']['issn_list']
    journals = pd.read_csv(journals_dir, sep='\t')
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