""" file containing modified CHIP for WSDM-challenge datasets
    and other helper functions
"""
import warnings
import numpy as np
from scipy.optimize import minimize_scalar
import sys
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
sys.path.append("./CHIP-Network-Model")
from generative_model_utils import event_dict_to_block_pair_events
from parameter_estimation import neg_log_likelihood_beta

def unbiased_mean_var_sparse(a):
    """ Unbiased mean and variance of sparse matrix a
    """
    a_mean = a.mean()
    n = a.shape[0] * a.shape[1]
    a_squared = a.copy()
    a_squared.data **= 2
    a_var = n/(n-1) * (a_squared.mean() - np.square(a_mean))
    # a_sum = a.sum()
    return a_mean, a_var

def compute_sample_mean_and_variance_sparse(agg_adj, user_item_mem, K, n_users):
    """
    :param agg_adj: sparse weighted adjacency of the network
    :param class_vec: (list) membership of every node to one of K classes.

    :return: N, S^2
    """
    sample_mean = np.zeros((K, K))
    sample_var = np.zeros((K, K))
    bp_size = np.zeros((K, K))

    # used for adj slicing
    users_per_class = []
    items_per_class = [] #scaled from 0
    for i in range(K):
        users_per_class.append(np.where(user_item_mem[:n_users] == i)[0])
        items_per_class.append(np.where(user_item_mem[n_users:] == i)[0])


    for a in range(K):
        for b in range(K):
            users_in_a = users_per_class[a]
            items_in_b = items_per_class[b]
            bp_size[a, b] = users_in_a.size * items_in_b.size

            # if both block sizes = 1, no need to compute params of that block pair, set it to the default.
            if users_in_a.size <= 1 and items_in_b.size <= 1:
                sample_mean[a, b] = 0
                sample_var[a, b] = 0
                continue

            agg_adj_block = agg_adj.tocsr()[users_in_a[:, np.newaxis], items_in_b]
            sample_mean[a, b], sample_var[a, b] = unbiased_mean_var_sparse(agg_adj_block)

    return sample_mean, sample_var, bp_size

def estimate_hawkes_from_counts_sparse(agg_adj, nodes_mem, duration, K, n_users, default_mu=None):
    sample_mean, sample_var, bp_size = compute_sample_mean_and_variance_sparse(agg_adj, nodes_mem, K, n_users)

    # Variance can be zero, resulting in division by zero warnings. Ignore and set a default mu.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.sqrt(sample_mean**3 / sample_var) / duration
        alpha_beta_ratio = 1 - np.sqrt(sample_mean / sample_var)

    # If sample_var is 0, depending on sample mean, mu and the ratio can be nan or inf. Set them to default values.
    if default_mu is not None:
        mu[np.isnan(mu)] = default_mu
        mu[np.isinf(mu)] = default_mu
        alpha_beta_ratio[np.isnan(alpha_beta_ratio)] = 0
        alpha_beta_ratio[np.isinf(alpha_beta_ratio)] = 0

        # If ratio is negative, set it to 0
        alpha_beta_ratio[alpha_beta_ratio < 0] = 0

    return mu, alpha_beta_ratio, bp_size

def estimate_beta_from_events_sparse(bp_events, mu, alpha_beta_ratio, end_time, block_pair_size=None, tol=1e-3):
    res = minimize_scalar(neg_log_likelihood_beta, method='bounded', bounds=(0, 10),
                          args=(bp_events, mu, alpha_beta_ratio, end_time, block_pair_size))
    return res.x, res

def estimate_bp_hawkes_params_sparse(event_dict, agg_adj, nodes_mem, duration, K, n_users,
                                     return_block_pair_events=False):

    bp_mu, bp_alpha_beta_ratio, bp_size = estimate_hawkes_from_counts_sparse(agg_adj, nodes_mem, duration, K, n_users, 1e-10 / duration)

    bp_beta = np.zeros((K,K), dtype=np.float)
    block_pair_events = event_dict_to_block_pair_events(event_dict, nodes_mem, K)

    for b_i in range(K):
        for b_j in range(K):
            bp_beta[b_i, b_j], _ = estimate_beta_from_events_sparse(block_pair_events[b_i][b_j],
                                                                    bp_mu[b_i, b_j],
                                                                    bp_alpha_beta_ratio[b_i, b_j],
                                                                    duration, bp_size[b_i, b_j])

    bp_alpha = bp_alpha_beta_ratio * bp_beta

    if return_block_pair_events:
        return bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio, block_pair_events

    return bp_mu, bp_alpha, bp_beta, bp_alpha_beta_ratio

def event_dict_to_aggregated_adjacency_sparse(n_rows, n_cols, event_dicts):
    l = len(event_dicts)
    data = np.zeros(l, dtype=int)
    row = np.zeros(l, dtype=int)
    col = np.zeros(l, dtype=int)
    for i, ((u, v), events) in enumerate(event_dicts.items()):
        data[i] = len(events)
        row[i] = u
        col[i] = v - n_rows
    sparse_adj = coo_matrix((data, (row, col)), shape=(n_rows, n_cols), dtype=float)
    # for all node pairs (u, v) in events dict -> u:[0, n_rows-1] and v:[n_rows, n_rows+n_cols-1]
    return sparse_adj

def spectral_cluster_sparse(adj, num_classes=2, n_kmeans_init=50):
    """
    Runs spectral clustering on sparse weighted adjacency matrix

    :param adj: weighted, unweighted or regularized adjacency matrix
    :param num_classes: number of classes for spectral clustering
    :param n_kmeans_init: number of initializations for k-means

    :return: predicted clustering membership
    """
    # # compute normalized adjacency
    # D_items = diags(np.ravel(adj.sum(axis=0))).sqrt()
    # D_users = diags(np.ravel(adj.sum(axis=1))).sqrt()
    # adj = D_users.dot(adj.dot(D_items))

    # Compute largest num_classes singular values and vectors of normalized adjacency matrix
    u, s, v = svds(adj, k=num_classes)
    v = v.T

    # Sort in decreasing order of magnitude
    sorted_ind = np.argsort(-s)
    u = u[:, sorted_ind]
    v = v[:, sorted_ind]

    z = np.r_[u, v]
    z = normalize(z, norm='l2', axis=1)

    km = KMeans(n_clusters=num_classes, n_init=n_kmeans_init)
    cluster_pred = km.fit_predict(z)

    return cluster_pred

def hawkes_intensity_modified(mu, alpha, beta, s, timestamps):
    """ Returns Hawkes intensity """
    return mu + alpha * np.sum(np.exp(-beta * (s - timestamps)))

def get_node_map_modified(node_set, start=0):
    nodes = list(node_set)
    nodes.sort()
    node_id_map = {}
    for i, n in enumerate(nodes, start=start):
        node_id_map[n] = i
    return node_id_map

def cal_expected(mu, alpha, beta, t, lambda_0, taw):
    first = - mu * beta * taw / (alpha - beta)
    second_nominator = - mu * beta + np.exp((alpha - beta) * taw) * (mu * beta + lambda_0 * alpha - lambda_0 * beta) - alpha * lambda_0
    second_denominator = (alpha - beta) ** 2
    return first + np.exp((alpha-beta)*t)*second_nominator/second_denominator