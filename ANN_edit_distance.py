
def ANN_edit_distance(seq, n_neighbors=30, radius=None,
                   MAX_SHEARCH=200, verbose=False):
    if verbose:
        print("1/3 Count all elements...")
    import pandas as pd
    import numpy as np
    from collections import Counter
    n = len(seq)
    count_pool = list(map(Counter, seq))
    data_counter = pd.DataFrame.from_dict(count_pool)
    data_counter = data_counter.fillna(0).values

    if radius is None:
        if verbose:
            print("2/3 Find possible neighbors...")
        # if MAX_SHEARCH > n/n_neighbors:
        #     raise warnings("MAX_SHEARCH is too large!")
        from pynndescent import NNDescent
        n_candidate = MAX_SHEARCH
        index = NNDescent(data_counter, random_state=42,
                          n_neighbors=n_candidate, metric="manhattan")
        knnIdx, knnDist = index.neighbor_graph
        if verbose:
            print("3/3 Compute edit distances...")
        data = []
        row = []
        col = []
        if verbose:
            from tqdm import tqdm
            range_n = tqdm(range(n))
        else:
            range_n = range(n)
        from Levenshtein import distance
        import heapq

        indicator = {}
        jump_count = 0
        MAX_DISTANCE = 100
        knnIdx_real, knnDist_real = np.zeros(
            (n, n_neighbors), dtype=int), np.zeros((n, n_neighbors), dtype=int)
        knnDist_candidate = np.zeros_like(knnDist)+MAX_DISTANCE
        for i in range_n:
            # the max distance of top n_neighbors
            current_max_distance_of_seq_i = 0
            # compute first n_neighbors
            for j in range(n_neighbors):
                if i == knnIdx[i, j]:
                    dist = 0
                else:
                    dist = distance(seq[i], seq[knnIdx[i, j]])
                knnDist_candidate[i, j] = dist
            # record the max distance of top n_neighbors
            current_max_distance_of_seq_i = knnDist_candidate[i, :n_neighbors].max(
            )
            # record the top n_neighbors
            # set negitive value to Max Heap
            top_n_neighbors = (-knnIdx[i, :n_neighbors]).tolist()

            for j in range(n_neighbors, n_candidate):

                if knnDist[i, j] > current_max_distance_of_seq_i * 2:
                    # if the distance is too large, we can jump
                    continue
                else:
                    mark = (min(knnIdx[i, j], i), max(knnIdx[i, j], i))
                    if mark in indicator:
                        jump_count += 1
                        dist = indicator[mark]
                    else:
                        dist = distance(seq[i], seq[knnIdx[i, j]])
                        indicator[mark] = dist
                    # update max distance
                    if dist < current_max_distance_of_seq_i:
                        knnDist_candidate[i, j] = dist
                        heapq.heappushpop(top_n_neighbors, -dist)
                        current_max_distance_of_seq_i = -top_n_neighbors[0]

            ind = knnDist_candidate[i].argsort(kind="quicksort")
            ind = ind[:n_neighbors]
            knnIdx_real[i] = knnIdx[i, ind]
            knnDist_real[i] = knnDist_candidate[i, ind]
        if verbose:
            filtered_ratio = 1-(knnDist_candidate < MAX_DISTANCE).sum()/(n**2)
            print("filtered_ratio: ", filtered_ratio)
            print("jump_count: ", jump_count)
        return knnIdx_real, knnDist_real
    else:
        if verbose:
            print("2/3 Find possible neighbors...")

        # from sklearn.neighbors import NearestNeighbors
        # neighbors_model = NearestNeighbors(
        #     radius=radius * 2,
        #     metric="manhattan",
        #     algorithm="ball_tree",
        #     leaf_size=n//2,
        #     n_jobs=-1,
        # )

        # neighbors_model.fit(data_counter)
        # # This has worst case O(n^2) memory complexity
        # knnDist, knnIdx = neighbors_model.radius_neighbors(data_counter,
        #                                                    return_distance=True, sort_results=True)
        from pynndescent import NNDescent
        index = NNDescent(data_counter, random_state=42,
                          n_neighbors=n_candidate, metric="manhattan")
        knnIdx, knnDist = index.neighbor_graph

        if verbose:
            print("3/3 Compute edit distances...")
        from Levenshtein import distance
        from scipy.sparse import csr_matrix
        data = []
        row = []
        col = []
        if verbose:
            range_n = tqdm(range(n))
        else:
            range_n = range(n)

        for i in range_n:
            for j in range(len(knnDist[i])):
                if j > MAX_SHEARCH:
                    break
                if i == knnIdx[i][j]:
                    dist = 0
                else:
                    dist = distance(seq[i], seq[knnIdx[i][j]])
                if dist <= radius:
                    row.append(i)
                    col.append(knnIdx[i][j])
                    data.append(dist)
        data1 = data + data
        row1 = row + col
        col1 = col + row
        A = csr_matrix((data1, (row1, col1)),
                       shape=(n, n))

        return A