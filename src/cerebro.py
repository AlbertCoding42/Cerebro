import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import umap
import scipy
from src.plotting import *
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import multiprocessing as mp
import logging
from tqdm import tqdm

class Cerebro(object):
    def __init__(self, data_dir, affinity_thresh=0.5):
        """ 
        Initialize the Cerebro object.
        Parameters
        ----------
        data_dir : str
            The path to the data.
        """
        # load the raw data
        self.data_dir = data_dir
        raw_data_path = os.path.join(data_dir, 'raw_text.csv')
        self.raw_data = pd.read_csv(raw_data_path)
        # load the vectors
        vectors_path = os.path.join(data_dir, 'vectors.npy')
        self.vectors = np.load(vectors_path)
        # load the adjacency matrix
        aff_path = os.path.join(data_dir, 'affinity.npy')
        if os.path.exists(aff_path):
            self.affinity_matrix = np.load(aff_path)
        else:
            # create the affinity matrix
            self._create_affinity_matrix(affinity_thresh)
        print('Successfully loaded data')
        print('Formatting conversations')
        self._format_conversations()
        # initialize the affinity matrix and shortest path distance matrix
        # self.sp_dist_matrix = None
        # create relevant matrices
        # self._create_sp_dist_matrix()
        self.graph = nx.from_numpy_array(self.affinity_matrix)
        # initialize embeddings
        self.spectral()
        self.umap()

    def _create_affinity_matrix(self, threshold):
        """
        Create the adjacency matrix from the vectors.
        """
        # calculate the cosine similarity between vectors
        self.affinity_matrix = cosine_similarity(self.vectors)
        # set entries below threshold to 0
        self.affinity_matrix[self.affinity_matrix < threshold] = 0
        # save the adjacency matrix
        np.save(os.path.join(self.data_dir, 'affinity.npy'), self.affinity_matrix)

    def _format_conversations(self):
        """
        Format the conversations for display.
        """
        conversations = self.raw_data['text'].to_numpy()
        # for each conversation, insert <br> before InsightAC, and every 90 characters
        for i, conversation in enumerate(tqdm(conversations, desc='Formatting conversations')):
            conversations[i] = conversation.replace('**InsightAC**', '<br>**InsightAC**')
            conversations[i] = '<br>'.join([conversation[j:j+90] for j in range(0, len(conversation), 90)])
        self.raw_conversations = conversations

    def _create_sp_dist_matrix(self):
        """
        Create the shortest path distance matrix from the affinity matrix.
        """
        distance_matrix = np.exp(-self.affinity_matrix)
        self.sp_dist_matrix = scipy.sparse.csgraph.shortest_path(distance_matrix, directed=False, unweighted=False)

    def spectral(self, dim=3):
        """
        Perform spectral embedding on the affinity matrix.
        Parameters
        ----------
        dim : int
            The dimension of the embedding.
        """
        se = SpectralEmbedding(n_components=dim, affinity='precomputed')
        self.emb_spectral = se.fit_transform(self.affinity_matrix)
        return self.emb_spectral
    
    def umap(self, dim=3):
        """
        Perform UMAP on the affinity matrix.
        Parameters
        ----------
        dim : int
            The dimension of the embedding.
        """
        umap_obj = umap.UMAP(n_components=dim)
        self.emb_umap = umap_obj.fit_transform(self.vectors)
        return self.emb_umap
    
    def ricci_flow(self, iterations=200):
        """
        Perform Ricci flow on the affinity matrix.
        Parameters
        ----------
        dim : int
            The dimension of the embedding.
        """
        # catch logger output and redirect to stdout
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        # instantiate ricciflow object
        self.ricci = OllivierRicci(self.graph, alpha=0.5, verbose="INFO", proc=mp.cpu_count()//2)
        print("\nBeginning ricci curvature computation\n")
        self.ricci.compute_ricci_curvature()
        print("\nRicci curvature computation done\n")
        G_rf = self.ricci.compute_ricci_flow(iterations=iterations)
        return G_rf
    
    def spring_layout(self, G, dim=3):
        """
        Perform spring layout on the graph.
        Parameters
        ----------
        G : networkx.Graph
            The graph to layout.
        """
        if dim == 2:
            return nx.spring_layout(G)
        val = nx.spring_layout(G, dim=dim)
        return np.array([val[key] for key in range(len(val))])
    
    def visualize(self, emb="umap"):
        """
        Visualize the graph embeddings.
        Parameters
        ----------
        emb : str
            The embedding method to use.
        """
        emb_map = {
            "umap": self.emb_umap,
            "spectral": self.emb_spectral,
        }

        emb = emb_map[emb]
        fig = plot_data_3D(
            emb, 
            title='Cerebro', 
            color=None, 
            text=self.raw_conversations
        )
        return fig