from pinecone import Pinecone
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import argparse
from tqdm import tqdm


load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")

def fetch_metadata(output_path, namespace, num_entries=None):
    """ 
    Fetch metadata from Pinecone index and save to a file.
    
    Parameters
    ----------
    output_path : str
        The path to save the metadata.
    namespace : str
        The namespace of the index.
    num_entries : int
        Number of entries to fetch.
    """
    pc = Pinecone(PINECONE_API_KEY)
    index = pc.Index("qa")
    metadata_texts = []
    vectors = []
    ids = []
    for id in index.list(namespace=namespace):
        ids.extend(id)
    # take first num_entries ids
    if num_entries is not None:
        ids = ids[:num_entries]
    # Step 2: Fetch metadata in batches using ID ranges
    for id in tqdm(ids, desc="Fetching metadata"):
        # Fetch metadata for the batch of IDs
        response = index.fetch(
            ids=[id], 
            namespace=namespace,
        )
        metadata_texts.append(response["vectors"][str(id)].metadata['text'])
        vectors.append(np.array(response["vectors"][str(id)]['values']))
    # save the metadata and ids as csv
    metadata_df = pd.DataFrame()
    metadata_df['text'] = metadata_texts
    # add ids to the dataframe
    metadata_df['id'] = ids
    metadata_df.to_csv(os.path.join(output_path, 'raw_text.csv'), index=False)
    # save the vectors
    vectors = np.array(vectors)
    np.save(os.path.join(output_path, 'vectors.npy'), vectors)
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fetch metadata from Pinecone index.')
    parser.add_argument('--output_name', type=str, help='Output directory name', required=True)
    parser.add_argument('--num_entries', type=int, help='Number of entries to fetch', default=None)
    parser.add_argument('--namespace', type=str, help='Namespace of the index', default='8d65f922-0a51-4cdb-8a6c-c40ab8b8603d')
    args = parser.parse_args()

    load_dotenv()
    namespace = args.namespace
    output_name = args.output_name
    num_entries = args.num_entries

    output_path = os.path.join('../data', output_name)
    os.makedirs(output_path, exist_ok=True)
    fetch_metadata(output_path, namespace, num_entries)
    print(f"Metadata saved to {output_path}")