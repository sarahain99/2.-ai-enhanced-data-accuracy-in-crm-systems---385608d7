import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cluster_similar_profiles(df, columns_to_cluster, n_clusters=None, similarity_threshold=0.8):
    """Clusters similar customer profiles based on text features."""
    combined_text = df[columns_to_cluster].astype(str).agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    cosine_similarities = cosine_similarity(tfidf_matrix)

    # Determine number of clusters automatically based on similarity
    if n_clusters is None:
        n_clusters = int((1 - similarity_threshold) * df.shape[0]) + 1
        n_clusters = max(2, min(n_clusters, df.shape[0] // 2)) # Ensure a reasonable number

    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward') # 'ward' minimizes within-cluster variance
    clusters = clustering_model.fit_predict(tfidf_matrix.toarray())

    df['cluster_id'] = clusters
    return df.groupby('cluster_id').filter(lambda x: len(x) > 1) # Return clusters with more than one member

def analyze_clusters(clustered_df):
    """Analyzes the identified clusters of potential duplicates."""
    for cluster_id, group in clustered_df.groupby('cluster_id'):
        print(f"\nPotential Duplicates in Cluster {cluster_id}:")
        print(group[['Name', 'Email', 'Phone']].to_string()) # Display relevant columns

if __name__ == "__main__":
    # Sample data
    data = {'Name': ['John Doe', 'John D.', 'Jane Smith', 'Jane S.', 'Peter Jones', 'Peter Jone'],
            'Email': ['john.doe@example.com', 'johndoe@example.com', 'jane.smith@example.com', 'jane.smith@example.com', 'peter.j@example.com', 'peterj@example.com'],
            'Phone': ['123-456-7890', '1234567890', '987-654-3210', '9876543210', '555-123-4567', '5551234567'],
            'City': ['New York', 'NY', 'Los Angeles', 'LA', 'London', 'London']}
    sample_df = pd.DataFrame(data)

    clustered_df = cluster_similar_profiles(sample_df, ['Name', 'Email', 'Phone', 'City'], similarity_threshold=0.85)
    analyze_clusters(clustered_df)