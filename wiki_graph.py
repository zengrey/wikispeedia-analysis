import pandas as pd
import networkx as nx
import numpy as np
from urllib.parse import unquote
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from datetime import datetime

class WikiGraph:
    def __init__(self):
        self.articles_df = None
        self.categories_df = None
        self.links_df = None
        self.distances_matrix = None
        self.G = nx.DiGraph()  # Directed graph since links have direction
        self.article_to_index = {}  # Mapping from article name to matrix index
        
    def read_articles(self):
        """Read articles.tsv file"""
        self.articles_df = pd.read_csv('articles.tsv', comment='#', names=['article'])
        self.articles_df['article'] = self.articles_df['article'].apply(unquote)
        
        # Create mapping from article name to index
        self.article_to_index = {article: idx for idx, article in enumerate(self.articles_df['article'])}
        
        return self.articles_df
    
    def read_categories(self):
        """Read categories.tsv file"""
        self.categories_df = pd.read_csv('categories.tsv', sep='\t', comment='#', 
                                       names=['article', 'category'])
        self.categories_df['article'] = self.categories_df['article'].apply(unquote)
        return self.categories_df
    
    def read_links(self):
        """Read links.tsv file"""
        self.links_df = pd.read_csv('links.tsv', sep='\t', comment='#',
                                   names=['linkSource', 'linkTarget'])
        self.links_df['linkSource'] = self.links_df['linkSource'].apply(unquote)
        self.links_df['linkTarget'] = self.links_df['linkTarget'].apply(unquote)
        return self.links_df
    
    def read_distances(self):
        """Read shortest-path-distance-matrix.txt file"""
        # Read the raw text file
        with open('shortest-path-distance-matrix.txt', 'r') as f:
            # Skip comments
            while True:
                line = f.readline()
                if not line.startswith('#'):
                    break
            
            # Read the distance matrix
            distances = []
            for line in f:
                # Convert string of digits to list of integers
                # Replace '_' with -1 to indicate unreachable
                row = [int(x) if x != '_' else -1 for x in line.strip()]
                distances.append(row)
                
        self.distances_matrix = np.array(distances)
        return self.distances_matrix
    
    def read_paths_data(self):
        """Read and process paths data from both finished and unfinished games"""
        # Read finished paths
        finished_paths = pd.read_csv('paths_finished.tsv', 
                                   comment='#',
                                   sep='\t',
                                   names=['ip', 'timestamp', 'duration', 'path', 'rating'])
        
        # Read unfinished paths
        unfinished_paths = pd.read_csv('paths_unfinished.tsv',
                                     comment='#',
                                     sep='\t',
                                     names=['ip', 'timestamp', 'duration', 'path', 'target', 'type'])
        
        return finished_paths, unfinished_paths
    
    def process_path(self, path_str):
        """Process a path string into a list of articles, handling back clicks"""
        path = path_str.split(';')
        processed_path = []
        
        for step in path:
            if step == '<':
                if processed_path:  # If there's a path to go back from
                    processed_path.pop()
            else:
                processed_path.append(unquote(step))
                
        return processed_path
    
    def get_shortest_path_length(self, source, target):
        """Get shortest path length from the distance matrix"""
        if self.distances_matrix is None:
            self.read_distances()
            
        try:
            source_idx = self.article_to_index[source]
            target_idx = self.article_to_index[target]
            distance = self.distances_matrix[source_idx][target_idx]
            return distance if distance != -1 else None
        except KeyError:
            return None
    
    def create_graph(self):
        """Create a networkx graph from the data"""
        # First ensure we have read all necessary data
        if self.articles_df is None:
            self.read_articles()
        if self.categories_df is None:
            self.read_categories()
        if self.links_df is None:
            self.read_links()
            
        # Add nodes (articles)
        for article in self.articles_df['article']:
            self.G.add_node(article)
            
        # Add edges (links)
        for _, row in self.links_df.iterrows():
            self.G.add_edge(row['linkSource'], row['linkTarget'])
            
        # Add category information as node attributes
        categories_dict = {}
        for _, row in self.categories_df.iterrows():
            if row['article'] not in categories_dict:
                categories_dict[row['article']] = []
            categories_dict[row['article']].append(row['category'])
            
        nx.set_node_attributes(self.G, categories_dict, 'categories')
        
        return self.G
    
    def plot_graph(self, max_nodes=100):
        """Plot a subset of the graph for visualization"""
        if len(self.G) == 0:
            self.create_graph()
            
        # Take a subset of nodes for visualization
        if len(self.G) > max_nodes:
            nodes = list(self.G.nodes())[:max_nodes]
            subgraph = self.G.subgraph(nodes)
        else:
            subgraph = self.G
            
        # Create the plot
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, 
                node_color='lightblue',
                node_size=500,
                with_labels=True,
                font_size=8,
                arrows=True,
                edge_color='gray',
                alpha=0.6)
        plt.title(f"Wiki Graph Visualization (showing {len(subgraph)} nodes)")
        plt.savefig('wiki_graph.png')
        plt.close()
        
    def get_graph_statistics(self):
        """Return basic statistics about the graph"""
        stats = {
            'Number of nodes': len(self.G),
            'Number of edges': self.G.size(),
            'Is directed': nx.is_directed(self.G),
            'Is connected': nx.is_strongly_connected(self.G) if nx.is_directed(self.G) else nx.is_connected(self.G),
            'Average degree': sum(dict(self.G.degree()).values()) / len(self.G),
            'Number of strongly connected components': nx.number_strongly_connected_components(self.G),
        }
        return stats
    
    def analyze_degree_distribution(self):
        """Analyze and plot in/out degree distributions"""
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot in-degree distribution
        in_degree_count = Counter(in_degrees)
        ax1.hist(in_degrees, bins=50, density=True, alpha=0.75, color='blue')
        ax1.set_xlabel('In-degree')
        ax1.set_ylabel('Frequency')
        ax1.set_title('In-degree Distribution')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # Plot out-degree distribution
        out_degree_count = Counter(out_degrees)
        ax2.hist(out_degrees, bins=50, density=True, alpha=0.75, color='red')
        ax2.set_xlabel('Out-degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Out-degree Distribution')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('degree_distribution.png')
        plt.close()
        
        return {
            'average_in_degree': np.mean(in_degrees),
            'max_in_degree': max(in_degrees),
            'average_out_degree': np.mean(out_degrees),
            'max_out_degree': max(out_degrees)
        }
    
    def analyze_path_lengths(self):
        """Analyze and plot path length distribution using the distance matrix"""
        if self.distances_matrix is None:
            self.read_distances()
        
        # Convert distances to a flat list, excluding unreachable paths (-1)
        path_lengths = self.distances_matrix[self.distances_matrix != -1].flatten()
        
        plt.figure(figsize=(10, 6))
        plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths) + 2, 1),
                density=True, alpha=0.75, color='green')
        plt.xlabel('Path Length')
        plt.ylabel('Frequency')
        plt.title('Path Length Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig('path_length_distribution.png')
        plt.close()
        
        return {
            'average_path_length': np.mean(path_lengths),
            'median_path_length': np.median(path_lengths),
            'max_path_length': np.max(path_lengths)
        }
    
    def analyze_clustering(self):
        """Analyze clustering coefficients and centrality measures"""
        # Calculate clustering coefficients
        clustering_coeffs = nx.clustering(self.G)
        avg_clustering = nx.average_clustering(self.G)
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.G)
        betweenness_centrality = nx.betweenness_centrality(self.G)
        
        try:
            # Eigenvector centrality might not converge for some graphs
            eigenvector_centrality = nx.eigenvector_centrality(self.G)
        except:
            eigenvector_centrality = {node: 0 for node in self.G.nodes()}
        
        # Create figure with multiple subplots for distributions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot clustering coefficient distribution
        sns.histplot(data=list(clustering_coeffs.values()), ax=ax1, bins=50)
        ax1.set_title('Clustering Coefficient Distribution')
        ax1.set_xlabel('Clustering Coefficient')
        ax1.set_ylabel('Count')
        
        # Plot degree centrality distribution
        sns.histplot(data=list(degree_centrality.values()), ax=ax2, bins=50)
        ax2.set_title('Degree Centrality Distribution')
        ax2.set_xlabel('Degree Centrality')
        ax2.set_ylabel('Count')
        
        # Plot betweenness centrality distribution
        sns.histplot(data=list(betweenness_centrality.values()), ax=ax3, bins=50)
        ax3.set_title('Betweenness Centrality Distribution')
        ax3.set_xlabel('Betweenness Centrality')
        ax3.set_ylabel('Count')
        
        # Plot eigenvector centrality distribution
        sns.histplot(data=list(eigenvector_centrality.values()), ax=ax4, bins=50)
        ax4.set_title('Eigenvector Centrality Distribution')
        ax4.set_xlabel('Eigenvector Centrality')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('centrality_measures.png')
        plt.close()
        
        return {
            'average_clustering': avg_clustering,
            'max_clustering': max(clustering_coeffs.values()),
            'max_degree_centrality': max(degree_centrality.values()),
            'max_betweenness_centrality': max(betweenness_centrality.values()),
            'max_eigenvector_centrality': max(eigenvector_centrality.values())
        }
    
    def analyze_path_differences(self):
        """Analyze differences between user paths and shortest paths"""
        finished_paths, unfinished_paths = self.read_paths_data()
        
        # Process finished paths
        path_differences = []
        ratings = []
        
        for _, row in finished_paths.iterrows():
            path = self.process_path(row['path'])
            if len(path) >= 2:  # Need at least source and target
                source = path[0]
                target = path[-1]
                user_path_length = len(path) - 1  # Number of steps (edges) in path
                shortest_length = self.get_shortest_path_length(source, target)
                
                if shortest_length is not None:
                    difference = user_path_length - shortest_length
                    path_differences.append(difference)
                    ratings.append(row['rating'] if row['rating'] != 'NULL' else None)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Create main plot for all path differences
        plt.subplot(2, 1, 1)
        plt.hist(path_differences, bins=range(min(path_differences), max(path_differences) + 2, 1),
                alpha=0.75, color='blue')
        plt.xlabel('User Path Length - Shortest Path Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Path Length Differences')
        plt.grid(True, alpha=0.3)
        
        # Create subplot for path differences by rating
        plt.subplot(2, 1, 2)
        
        # Convert to numpy arrays for easier manipulation
        path_differences = np.array(path_differences)
        ratings = np.array(ratings)
        
        # Create box plot for each rating
        rating_differences = []
        rating_labels = []
        
        for rating in range(1, 6):  # Ratings 1-5
            mask = ratings == rating
            if np.any(mask):
                rating_differences.append(path_differences[mask])
                rating_labels.append(f'Rating {rating}')
        
        plt.boxplot(rating_differences, labels=rating_labels)
        plt.xlabel('User Rating')
        plt.ylabel('Path Length Difference')
        plt.title('Path Length Differences by User Rating')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('path_analysis.png')
        plt.close()
        
        # Calculate statistics
        stats = {
            'total_games': len(path_differences),
            'average_difference': np.mean(path_differences),
            'median_difference': np.median(path_differences),
            'min_difference': np.min(path_differences),
            'max_difference': np.max(path_differences),
            'std_difference': np.std(path_differences)
        }
        
        # Calculate statistics by rating
        rating_stats = {}
        for rating in range(1, 6):
            mask = ratings == rating
            if np.any(mask):
                rating_stats[f'rating_{rating}'] = {
                    'count': np.sum(mask),
                    'average_difference': np.mean(path_differences[mask]),
                    'median_difference': np.median(path_differences[mask])
                }
        
        return stats, rating_stats
    
    def analyze_network_properties(self):
        """Perform all network analyses and print results"""
        print("\nAnalyzing degree distribution...")
        degree_stats = self.analyze_degree_distribution()
        print("Degree distribution analysis saved as 'degree_distribution.png'")
        print(f"Average in-degree: {degree_stats['average_in_degree']:.2f}")
        print(f"Average out-degree: {degree_stats['average_out_degree']:.2f}")
        
        print("\nAnalyzing path lengths...")
        path_stats = self.analyze_path_lengths()
        print("Path length distribution saved as 'path_length_distribution.png'")
        print(f"Average path length: {path_stats['average_path_length']:.2f}")
        print(f"Maximum path length: {path_stats['max_path_length']}")
        
        print("\nAnalyzing clustering and centrality measures...")
        clustering_stats = self.analyze_clustering()
        print("Centrality measures distributions saved as 'centrality_measures.png'")
        print(f"Average clustering coefficient: {clustering_stats['average_clustering']:.4f}")
        print(f"Maximum clustering coefficient: {clustering_stats['max_clustering']:.4f}")
        
        print("\nAnalyzing user paths...")
        path_stats, rating_stats = self.analyze_path_differences()
        print("\nPath Analysis Statistics:")
        print(f"Total games analyzed: {path_stats['total_games']}")
        print(f"Average path difference: {path_stats['average_difference']:.2f}")
        print(f"Median path difference: {path_stats['median_difference']:.2f}")
        print(f"Path difference range: [{path_stats['min_difference']}, {path_stats['max_difference']}]")
        print("\nStatistics by rating:")
        for rating, stats in rating_stats.items():
            print(f"{rating.replace('_', ' ').title()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Average difference: {stats['average_difference']:.2f}")
            print(f"  Median difference: {stats['median_difference']:.2f}")
        print("\nPath analysis visualization saved as 'path_analysis.png'")

def main():
    # Create WikiGraph instance
    wiki = WikiGraph()
    
    # Read all data and create graph
    print("Reading data files...")
    wiki.read_articles()
    wiki.read_categories()
    wiki.read_links()
    wiki.read_distances()
    
    print("\nCreating graph...")
    wiki.create_graph()
    
    # Print graph statistics
    print("\nGraph Statistics:")
    stats = wiki.get_graph_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Perform network analysis
    wiki.analyze_network_properties()
    
    # Plot a subset of the graph
    print("\nCreating visualization...")
    wiki.plot_graph(max_nodes=100)
    print("Visualization saved as 'wiki_graph.png'")

if __name__ == "__main__":
    main() 