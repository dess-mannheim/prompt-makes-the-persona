from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
import argparse

MODELS = [
    "google/gemma-3-27b-it",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-0325-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]


def get_task_details(task_name):
    if task_name == 'SD':
        return {'name': 'SD', 'full': 'self-description', 'column': 'self_description'}
    elif task_name == 'Bio':
         return {'name': 'Bio', 'full': 'social-media biography', 'column': 'bio'}


def dunn_index(embeddings, labels):
    unique_labels = np.unique(labels)
    clusters = [embeddings[labels == lbl] for lbl in unique_labels]
    
    inter_cluster_dists = [
        np.min(cdist(c1, c2))
        for i, c1 in enumerate(clusters)
        for j, c2 in enumerate(clusters)
        if i < j
    ]
    intra_cluster_dists = [np.max(pdist(c)) if len(c) > 1 else 0 for c in clusters]
    
    if not intra_cluster_dists or max(intra_cluster_dists) == 0:
        return np.nan
    
    return min(inter_cluster_dists) / max(intra_cluster_dists)

def average_intra_cluster_distance(embeddings, labels):
    unique_labels = np.unique(labels)
    intra_dists = []
    
    for lbl in unique_labels:
        cluster_points = embeddings[labels == lbl]
        if len(cluster_points) > 1:
            dists = pdist(cluster_points)
            intra_dists.append(np.mean(dists))
    
    return np.mean(intra_dists) if intra_dists else np.nan

def evaluate_clusters(df, cluster_col='cluster'):
    embeddings = df.drop(columns=[cluster_col]).values
    labels = df[cluster_col].values

    if len(set(labels)) < 2:
        return {"error": "Need at least 2 clusters to compute these metrics."}

    results = {}
    metrics = [
        ("silhouette_score", lambda: silhouette_score(embeddings, labels)),
        ("davies_bouldin_index", lambda: davies_bouldin_score(embeddings, labels)),
        ("calinski_harabasz_index", lambda: calinski_harabasz_score(embeddings, labels)),
        #("dunn_index", lambda: dunn_index(embeddings, labels)),
        ("avg_intra_cluster_distance", lambda: average_intra_cluster_distance(embeddings, labels)),
    ]

    for name, func in tqdm(metrics, desc="Cluster metrics", position=1, leave=False):
        try:
            results[name] = func()
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    
    return results

def per_cluster_evaluations(df, cluster_col='cluster'):
    embeddings = df.drop(columns=[cluster_col]).values
    labels = df[cluster_col].values
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {"error": "Need at least 2 clusters."}

    silhouette_vals = silhouette_samples(embeddings, labels)

    results = []

    for cluster_id in tqdm(unique_labels, desc="Per-cluster metrics", position=1, leave=False):
        cluster_points = embeddings[labels == cluster_id]
        other_points = embeddings[labels != cluster_id]

        # Intra-cluster pairwise distance
        if len(cluster_points) > 1:
            intra_dists = pdist(cluster_points)
            avg_intra_pdist = np.mean(intra_dists)
        else:
            avg_intra_pdist = np.nan
        
        # Intra-cluster mean distance to centroid (=variance)
        if len(cluster_points) > 1:
            centroid = np.mean(cluster_points, axis=0)
            avg_intra_var = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
        else:
            avg_intra_var = np.nan

        # Mean distance of centroid to other cluster's centroids
        if len(other_points) > 0:
            cluster_centroid = np.mean(cluster_points, axis=0)
            other_centroids = [
                np.mean(embeddings[labels == other_id], axis=0)
                for other_id in unique_labels if other_id != cluster_id
            ]
            inter_dists = [np.linalg.norm(cluster_centroid - oc) for oc in other_centroids]
            avg_inter_centroid_dist = np.mean(inter_dists)
        else:
            avg_inter_centroid_dist = np.nan

        # Cluster size and average silhouette
        cluster_size = len(cluster_points)
        cluster_silhouette = silhouette_vals[labels == cluster_id]
        avg_silhouette = np.mean(cluster_silhouette)

        results.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'avg_intra_cluster_pairwise_distance': avg_intra_pdist,
            'avg_intra_cluster_variance': avg_intra_var,
            'avg_silhouette_score': avg_silhouette,
            'avg_inter_centroid_distance': avg_inter_centroid_dist
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, help='huggingface checkpoint id of the model')
    parser.add_argument('task', type=str, help="which open task to run ('SD' or 'Bio')")
    args = parser.parse_args()

    model_short = args.model_id.split('/')[1]

    task = get_task_details(args.task)

    if task is None:
        raise ValueError(f'Cannot annotate task {task['name']}, please select one of "SD" or "Bio"')

    # path to results
    res_path = f"../../data/results/{task['name']}"

    # path to embeddings
    embeddings_df = pd.read_csv(f'{task['name']}/embeddings_anonymized_{model_short}.csv', index_col=0)
    main_df = pd.read_csv(f'{res_path}/merged_results_anonymized_{model_short}.csv', index_col=0)
    combined_df = pd.merge(
        left=main_df, right=embeddings_df,
        left_index=True, right_index=True
    )

    combined_df['persona_string'] = combined_df['persona_string'].replace({'2nd': 'Direct', '3rd': 'Third Person', 'interview': 'Interview'})
    combined_df['persona_type'] = combined_df['persona_type'].replace({'dem_descr': 'Explicit', 'dem_cat+descr': 'Structured', 'name': 'Name'})
    combined_df = combined_df.rename(columns={'persona_string': 'Role Adoption', 'persona_type': 'Demographic Priming'})

    # create cluster ids
    combined_df['cluster'] = combined_df.gender + '.' + combined_df.race + '.' \
        + combined_df['Demographic Priming'] + '.' + combined_df['Role Adoption']

    # remove AI output
    combined_df = combined_df[combined_df['LLM_AI_annotation'].astype(str) == 'False']

    metrics_df = per_cluster_evaluations(combined_df[list(embeddings_df.columns) + ['cluster']], cluster_col='cluster')
    metrics_df['model'] = model_short
    metrics_df['task'] = task['column']

    metrics_df.to_csv(f'{task['name']}/cluster_metrics_anonymized_{model_short}.csv')
    