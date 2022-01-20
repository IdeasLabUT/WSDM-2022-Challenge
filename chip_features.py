import os
import sys
import numpy as np
import pickle
sys.path.append(os.path.join(os.getcwd(), "CHIP-Network-Model"))
from dataset_utils import get_node_map
from model_fitting_utils import fit_community_model
import chip_helper


second_hour_scale = 60 * 60


# Convert train dataset A to dictionary of node pairs for CHIP
def create_dictionary_chip_A(datapath, savepath=None):
    # load sender_node, receiver_node, timestamp
    data = np.loadtxt(datapath, np.float, delimiter=',', usecols=(0, 1, 3))
    train_nodes = data[:, 0:2]
    train_nodes_set = set(train_nodes.reshape(train_nodes.shape[0] * 2))
    N = len(train_nodes_set) # number of nodes
    # map each node to unique id starting from 0 to N-1
    train_node_id_map = get_node_map(train_nodes_set)
    # Sort by timestampe
    data = data[data[:, 2].argsort()]
    # scaled (seconds to hours) end time of train dataset
    T_train = (data[-1, 2] - data[0, 2]) / second_hour_scale
    dataset_start_t = data[0, 2]
    # Create dictionary of node pairs, each is list of links timestamps between the node pair
    events_dict_train = {}
    for i in range(data.shape[0]):
        sender_id = train_node_id_map[np.int(data[i, 0])]
        receiver_id = train_node_id_map[np.int(data[i, 1])]
        if (sender_id, receiver_id) not in events_dict_train:
            events_dict_train[(sender_id, receiver_id)] = []
        # scale timestamp from sec to hour & start from 0
        timestamp = (data[i, 2] - data[0, 2]) / second_hour_scale
        events_dict_train[(sender_id, receiver_id)].append(timestamp)

    if savepath is not None:
        datasetA_tuple = (events_dict_train, T_train, dataset_start_t, N, train_node_id_map)
        with open(savepath, 'wb') as file:
            pickle.dump(datasetA_tuple, file)

    return events_dict_train, T_train, dataset_start_t, N, train_node_id_map

def read_dictionary_chip_A(savepath):
    with open(savepath, 'rb') as file:
        datasetA_tuple = pickle.load(file)
    return datasetA_tuple


# fit CHIP or load saved fit
def fit_chip_A(datasetA_tuple, K, savepath=None):
    # Get train dataset A in dictionary formate for CHIP model
    events_dict_train, T_train, dataset_start_t, N, train_node_id_map = datasetA_tuple

    # Fitting the model to the train dictionary
    node_mem_train, bp_mu, bp_alpha, bp_beta, events_dict_train_bp = \
        fit_community_model(events_dict_train, N, T_train, K, local_search_max_iter=0, local_search_n_cores=-1)
    fit_param_dict = {"parameters": (bp_mu, bp_alpha, bp_beta), "nodes_mem": node_mem_train}

    # save fit results
    if savepath is not None:
        with open(savepath, 'wb') as file:
            pickle.dump(fit_param_dict, file)

    return fit_param_dict

def load_fit_chip_A(savepath):
    with open(savepath, 'rb') as file:
        fit_param_dict = pickle.load(file)
    return fit_param_dict


def create_feature_chip_A(datasetA_tuple, data_test, fit_param_dict, savepath=None):
    events_dict_train, T_train, dataset_start_t, N, train_node_id_map = datasetA_tuple

    # CHIP fit parameters
    bp_mu, bp_alpha, bp_beta = fit_param_dict["parameters"]
    node_mem_train = fit_param_dict["nodes_mem"]

    expected_result = np.zeros((data_test.shape[0]))
    for r in range(data_test.shape[0]):
        # row (sender, receiver, edge_type, start_edge_t, end_edge_t)
        row = data_test[r, :]
        src_id, dst_id = train_node_id_map[row[0]], train_node_id_map[row[1]]
        from_blk, to_blk = node_mem_train[src_id], node_mem_train[dst_id]
        # scale start and end link time from seconds to hours
        start_edge_t = (row[3] - dataset_start_t) / second_hour_scale
        end_edge_t = (row[4] - dataset_start_t) / second_hour_scale
        # get parameters and timestamps history of block pair
        mu, alpha, beta = bp_mu[from_blk, to_blk], bp_alpha[from_blk, to_blk], bp_beta[from_blk, to_blk]
        # node pair train timestamps
        if events_dict_train.get((src_id, dst_id)):
            timestamps_data = np.array(events_dict_train[(src_id, dst_id)])
        else:
            timestamps_data = np.array([])

        # compute expected num of links between 2 nodes within [start, end edge time]
        lambda_0 = chip_helper.hawkes_intensity_modified(mu, alpha, beta, T_train, timestamps_data)  # lambda at end train time
        # 1) expected num of links within [end train time, end edge time]
        taw = end_edge_t - T_train
        expected_T_train_end = chip_helper.cal_expected(mu, alpha, beta, T_train, lambda_0, taw)
        # 2) expected num of links within [end train time, start edge time]
        taw = start_edge_t - T_train
        expected_T_train_start = chip_helper.cal_expected(mu, alpha, beta, T_train, lambda_0, taw)
        # 3) expected num of links within [start edge time, end edge time]
        expected_start_end = expected_T_train_end - expected_T_train_start
        # scale expectation using 1-exp(-x)
        expected_result[r] = 1 - np.exp(-expected_start_end)

    if savepath is not None:
        np.savetxt(savepath, expected_result, delimiter=',')

    return expected_result

# Convert train dataset B to dictionary of node pairs for CHIP
def create_dictionary_chip_B(datapath, savepath=None):
    # read (user, iterm, edge type, timestamp) columns
    dataB = np.loadtxt(datapath, np.int, delimiter=',', usecols=(0, 1, 2, 3))
    users_set = set(dataB[:,0])
    items_set = set(dataB[:,1])
    n_users = len(users_set)
    n_items = len(items_set)
    user_id_map = chip_helper.get_node_map_modified(users_set) # ids values between [0, n_users-1]
    item_id_map = chip_helper.get_node_map_modified(items_set, start=n_users) # ids values between [n_users, n_users + n_items - 1]

    # edges types
    edges_set = set(dataB[:,2])
    print(f"#users={n_users}, #items={n_items}, #edges={len(edges_set)}")

    # Sort by timestamp - forth column
    dataB = dataB[dataB[:, 3].argsort()]
    T_train = (dataB[-1, 3] - dataB[0, 3]) / second_hour_scale
    dataset_start_t = dataB[0, 3]

    # Create dictionary of events
    events_dict_train = {}
    for i in range(dataB.shape[0]):
        user_id = user_id_map[np.int(dataB[i, 0])]
        item_id = item_id_map[np.int(dataB[i, 1])]
        if (user_id, item_id) not in events_dict_train:
            events_dict_train[(user_id, item_id)] = []
        # scale timestamp from sec to hour & start from 0
        timestamp = (dataB[i, 3] - dataset_start_t) / second_hour_scale
        events_dict_train[(user_id, item_id)].append(timestamp)

    # save events_dictionary for CHIP
    if savepath is not None:
        datasetB_tuple = (events_dict_train, T_train, dataset_start_t, user_id_map, item_id_map)
        with open(savepath, 'wb') as file:
            pickle.dump(datasetB_tuple, file)
    return events_dict_train, T_train, dataset_start_t, user_id_map, item_id_map

# load saved event dictionary
def read_dictionary_chip_B(savepath):
    with open(savepath, 'rb') as file:
        datasetB_tuple = pickle.load(file)
    return datasetB_tuple



# fit CHIP or load saved fit
def fit_chip_B(datasetB_tuple, K, savepath=None):
    events_dict_train, T_train, dataset_start_t, user_id_map, item_id_map = datasetB_tuple
    n_users, n_items = len(user_id_map), len(item_id_map)
    # 1) events_dict to sparse aggregated adjacency
    agg_adj = chip_helper.event_dict_to_aggregated_adjacency_sparse(n_users, n_items, events_dict_train)
    user_item_mem = chip_helper.spectral_cluster_sparse(agg_adj, num_classes=K)
    print(np.unique(user_item_mem[:n_users], return_counts=True))
    print(np.unique(user_item_mem[n_users:], return_counts=True))
    bp_mu, bp_alpha, bp_beta, _ = chip_helper.estimate_bp_hawkes_params_sparse(events_dict_train, agg_adj, user_item_mem, T_train, K, n_users)
    fit_param_dict = {"parameters": (bp_mu, bp_alpha, bp_beta), "nodes_mem": user_item_mem}
    # save fit results
    if savepath is not None:
        with open(savepath, 'wb') as file:
            pickle.dump(fit_param_dict, file)
    return fit_param_dict

def load_fit_chip_B(savepath):
    with open(savepath, 'rb') as file:
        fit_param_dict = pickle.load(file)
    return fit_param_dict

def create_feature_chip_B(datasetB_tuple, data_test, fit_param_dict, savepath):
    # read dictionary of events datasetB
    events_dict_train, T_train, dataset_start_t, user_id_map, item_id_map = datasetB_tuple
    # read CHIP fit parameters
    bp_mu, bp_alpha, bp_beta = fit_param_dict["parameters"]
    user_item_mem = fit_param_dict["nodes_mem"]

    expected_result = np.zeros((data_test.shape[0]))
    for r in range(data_test.shape[0]):
        # row (sender, receiver, edge type, start_edge_t, end_edge_t, bool_edg)
        row = data_test[r, :]
        if user_id_map.get(row[0])== None or item_id_map.get(row[1])==None:
            print("new item/user")
        else:
            user_id, item_id = user_id_map[row[0]], item_id_map[row[1]]
            from_block, to_block = user_item_mem[user_id], user_item_mem[item_id]
            start_edge_t = (row[3] - dataset_start_t) / second_hour_scale
            end_edge_t = (row[4] - dataset_start_t) / second_hour_scale
            mu, alpha, beta = bp_mu[from_block, to_block], bp_alpha[from_block, to_block], bp_beta[from_block, to_block]
            # node pair train timestamps
            if events_dict_train.get((user_id, item_id)):
                timestamps_train_np = np.array(events_dict_train[(user_id, item_id)])
            else:
                timestamps_train_np = np.array([])

            # compute expected num of links between 2 nodes within [start, end edge time]
            lambda_0 = chip_helper.hawkes_intensity_modified(mu, alpha, beta, T_train, timestamps_train_np) #lambda(end train time)
            # 1) expected num of links within [end train time, end edge time]
            taw = end_edge_t - T_train
            expected_T_train_end = chip_helper.cal_expected(mu, alpha, beta, T_train, lambda_0, taw)
            # 2) expected num of links within [end train time, start edge time]
            taw = start_edge_t -T_train
            expected_T_train_start = chip_helper.cal_expected(mu, alpha, beta, T_train, lambda_0, taw)
            # 3) expected num of links within [start edge time, end edge time]
            expected_start_end = expected_T_train_end - expected_T_train_start
            # scale expectation using 1-exp(-x)
            expected_result[r] = 1 - np.exp(- expected_start_end)

    if savepath is not None:
        np.savetxt(savepath, expected_result, delimiter=',')

    return expected_result
