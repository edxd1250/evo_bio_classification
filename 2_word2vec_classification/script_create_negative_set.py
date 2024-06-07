from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pd

def create_neg_set(dataset, journals, cleans, n):
    filtered_data = dataset[-dataset['Journal'].isin(journals)]
    sampled_data = filtered_data.sample(n=n)
    indices_of_sampled_data = sampled_data.index.tolist()
    new_clean = [cleans[x] for x in indices_of_sampled_data]
    return filtered_data, indices_of_sampled_data, new_clean