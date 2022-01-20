def load_dataset(filePath):
    data = []
    with open(filePath) as file:
        for line in file:
            line = [int(x) for x in line.rstrip().split(',')]
            d = tuple(line[:3])
            data.append(d)
    return data


# Dataset A
def load_node_features(filePath):
    nodes = {}
    with open(filePath, 'r') as file:
        for line in file:
            line = [int(x) for x in line.split(',')]
            nodes[line[0]] = set((f for f in line[1:] if f != -1))
    return nodes


def load_edge_type_features(filePath):
    edges = {}
    with open(filePath, 'r') as file:
        for line in file:
            line = [int(x) for x in line.split(',')]
            edges[line[0]] = set((f for f in line[1:] if f != -1))
    return edges


def compute_edge_frequencies(file_path):
    edge_freq = {}
    with open(file_path, 'r') as file:
        for line in file:
            u, v, e = [int(x) for x in line.split(',')[:3]]
            edge_freq.setdefault(u, {}).setdefault(v, {}).setdefault(e, 0)
            edge_freq[u][v][e] += 1
            edge_freq.setdefault(u, {}).setdefault(v, {}).setdefault('total', 0)
            edge_freq[u][v]['total'] += 1
    return edge_freq


def create_node_similarities(node_features):
    node_similarities = {}
    for i in node_features:
        for j in node_features:
            node_similarities.setdefault(i, {})
            node_similarities[i][j] = len(node_features[i].intersection(node_features[j])) / 3
    return node_similarities


def create_edge_type_similarities(edge_features):
    edge_similarities = {}
    for i in edge_features:
        for j in edge_features:
            edge_similarities.setdefault(i, {})
            edge_similarities[i][j] = len(edge_features[i].intersection(edge_features[j])) / 3

    return edge_similarities


def create_node_sim_features(node_similarities, data, missing=0):
    feature = []
    for i in range(len(data)):
        u, v, e = data[i]
        if u not in node_similarities or v not in node_similarities[u]:
            feature.append(missing)
        else:
            feature.append(node_similarities[u][v])
    return feature


def create_edge_type_sim_features(edge_similarities, edge_freq, data, missing=-1):
    feature = []
    for u, v, e in data:
        if u in edge_freq and v in edge_freq[u]:
            d = edge_freq[u][v]
            edgeSimilarity = 0
            for etype in d:
                if etype != 'total':
                    edgeSimilarity += d[etype] / d['total'] * edge_similarities[e][etype]
            feature.append(edgeSimilarity)
        else:
            feature.append(missing)

    return feature

