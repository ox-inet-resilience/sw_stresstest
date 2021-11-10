import random
import numpy as np
import networkx as nx
# from numba import jit


random.seed(1337)
np.random.seed(1337)

size = 51


# Halaj-Kok 2013
def generate_halaj_kok_2013_network():
    """
    See https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1506.pdf page 8-9
    """
    def generate_output():
        percentages = np.ones((size, size))
        out = np.zeros((size, size))

        for n in range(size * size):
            i, j = np.random.randint(0, size, size=2)
            if i != j and np.random.random() > 0.5:
                if out[i][j] == 0:  # don't double-assign on a matrix element
                    # the drawn link is accepted
                    epsilon = np.random.uniform()
                    # epsilon on average is 0.5, so the first allocation will be about 50%
                    # of the total liability, the next will be about 25%, and so on ...
                    out[i][j] = epsilon * percentages[i][j]
                    percentages[i][j] -= out[i][j]
        return out

    # Repeat the matrix generation many times, then average out the result, to avoid
    # order-dependence of the result, see Halaj-Kok section 2.2
    output = sum([generate_output() for i in range(100)]) / 100
    print(output)


# Montagna-Kok 2016
#@jit(nopython=True, nogil=True)
def generate_montagna_l1_network(interbanks, interbanks_liability):
    """
    :param interbanks list: List of total interbank exposure of each banks
    See the Montagna paper "Multi-layered interbank model for assessing
    systemic risk" P. 19-20
    """
    size = len(interbanks)
    iterations = 10000
    out = np.zeros((size, size))
    remainders_asset = np.array(interbanks, dtype=np.float64)  # create a copy of the array
    remainders_liability = np.array(interbanks_liability, dtype=np.float64)  # create a copy of the array

    # generate the random numbers once for all
    ij = np.random.randint(0, size, size=(iterations, 2))

    count = 0
    total_assets = sum(interbanks)
    total_liabilities = sum(interbanks_liability)
    # Set remainders that converge to 0 to be either on the asset or liability
    # side depending on which side is larger
    if int(total_assets) <= int(total_liabilities):
        sum_remainders = total_assets
        mode = "A"
    else:
        # total_assets > total_liabilities:
        sum_remainders = total_liabilities
        mode = "L"
    for n in range(iterations):
        count += 1
        i, j = ij[n]
        # NOTE: the 0.5 should have depended on geographical probability
        # map.
        fiftypercent = np.random.random() <= 0.5

        still_unlinked = out[i, j] == 0  # to make sure the link is still non-existent
        # WARN deviation from original version but this is used in Halaj 2018
        # (Agent-Based Model of System-Wide Implications of Funding Risk)
        # if this is enabled
        #still_unlinked = True

        if (i != j and fiftypercent and still_unlinked and (interbanks[i] != 0 and interbanks_liability[j] != 0 and
                                                            remainders_asset[i] != 0 and remainders_liability[j] != 0)):
            # The drawn link is accepted.
            epsilon = np.random.uniform()
            # Truncate the amount to not exceed 20% of total interbank asset
            # of the lender, and at the same time not exceed the remainder
            # of the quota for the borrower's interbank liability.
            #amount = min(min(0.2, epsilon) * interbanks[i], remainders_liability[j])
            # currently this algorithm is reversed since total interbank asset < total interbank liability
            # WARN deviation from original version: adding both remainders
            amount = min(
                min(0.2, epsilon) * interbanks_liability[j],
                remainders_asset[i],
                remainders_liability[j]
            )
            ###

            out[i, j] = amount
            remainders_asset[i] -= amount
            remainders_liability[j] -= amount
            sum_remainders -= amount

            # stop when the remainders have all been exhausted
            if sum_remainders <= 0:
                break

    # if there are still remainders asset left, link them to remainders liability
    # or vice versa
    # WARN this is deviation from original version
    if mode == 'A':
        # L == A
        sum_remainders = sum(remainders_asset)
        A_isequal_L = False
        if int(sum_remainders) == int(sum(remainders_liability)):
            A_isequal_L = True
        # L > A
        if sum_remainders > 0:
            js = list(range(size))
            for i in range(size):
                if remainders_asset[i] <= 0:
                    # already exhausted
                    continue
                np.random.shuffle(js)
                for j in js:
                    if i != j and remainders_liability[j] > 0:
                        epsilon = np.random.uniform()
                        amount = min(
                            # Deviation from Montagna algorithm: the 20% limit
                            # is no longer enforced
                            remainders_asset[i],
                            remainders_liability[j]
                        )
                        if amount > 0:
                            out[i, j] += amount
                            remainders_asset[i] -= amount
                            remainders_liability[j] -= amount

        if A_isequal_L:
            # HACK in the return structure TODO
            return out, (remainders_asset, remainders_liability)
        sum_remainders = sum(remainders_asset)
        sum_remainders_l = abs(sum(remainders_liability))
        error = (sum_remainders_l - abs(total_assets - total_liabilities)) / sum_remainders_l
        assert sum_remainders == 0, (sum_remainders, sum_remainders_l, count)
        assert abs(error) < 1e-5, error

        return out, remainders_liability
    else:
        # A > L
        sum_remainders_l = abs(sum(remainders_liability))
        assert sum_remainders_l == 0, sum_remainders_l
        return out, remainders_asset


#@jit(nopython=True, nogil=True, cache=True)
#@jit('Tuple(f8[:,:], f8[:,:])(i8, b1)', nopython=True, nogil=True, cache=True)
def generate_montagna_l3_network(size, asset_size, seed, generate_overlap=False):
    # Alternative version of the common asset network
    n_hf = size
    n_am = 5
    # ((financial vehicle corps) + (insurance companies & pension funds) +
    #  (financial corporations engaging lending))
    n_other = 5
    size_l3 = size + n_hf + n_am + n_other  # Number of nodes

    # Generate l3 network
    S = np.zeros((size_l3, asset_size))
    p = 0.5
    # must be reseeded because numba jit apparently doesn't take in the previously seeded random state
    np.random.seed(29979 + seed)
    for i in range(size_l3):
        S[i] = np.where(
            # generate a linkage between node i and the security at probability p
            np.random.random(asset_size) <= p,
            1,  # True value
            0)  # False value

    # Calculate weighted l3 network
    W = np.zeros((size_l3, asset_size))
    for i in range(size_l3):
        norm = np.sum(S[i])
        for mu in range(asset_size):
            if norm != 0:
                W[i][mu] = S[i][mu] / norm

    if not generate_overlap:
        return W, np.zeros((2, 2))
    else:
        # Calculate network of overlapping portofolio
        W_overlap = np.zeros((size_l3, size_l3))
        for i in range(size_l3):
            for j in range(size_l3):
                for mu in range(asset_size):
                    sum_Sj = np.sum(S[j])
                    if S[j][mu] != 0:
                        # equation 17
                        W_overlap[i][j] += S[j][mu] / sum_Sj * max(1, S[i][mu] / S[j][mu])
        return W, W_overlap


def generate_poisson_l1_network(interbanks, interbanks_liability, prob=0.5):
    size = len(interbanks)
    remainders = np.array(interbanks, dtype=np.float64)  # create a copy of the array
    remainders_liability = np.array(interbanks_liability, dtype=np.float64)  # create a copy of the array

    def _generate_net():
        out = np.zeros((size, size))
        for i in range(size):
            out[i] = np.where(
                np.random.random(size) <= prob,
                1, 0
            )
        np.fill_diagonal(out, 0)
        return out

    def _has_small_row_col(mat):
        for row in mat:
            if sum(row) <= 3:
                return True
        for col in mat.T:
            if sum(col) <= 3:
                return True
        return False
    out = _generate_net()
    _iteration = 0
    while _has_small_row_col(out):
        out = _generate_net()
        _iteration += 1
        if _iteration > 5000:
            print(prob)
            raise Exception('does not converge')
    for i in range(size):
        loan_size = interbanks[i]
        _sum_rem_lia = sum(remainders_liability[ii] for ii in range(size) if (ii != i) and out[i, ii] > 0 and remainders_liability[ii] > 0)
        if _sum_rem_lia == 0:
            print(out[i])
            raise Exception('ugh')
        for j in range(size):
            if j == i:
                continue
            if out[i, j] == 0:
                continue
            if _sum_rem_lia == 0:
                continue
            if remainders_liability[j] <= 0:
                continue
            _l = loan_size * remainders_liability[j] / _sum_rem_lia
            out[i, j] = _l
            remainders_liability[j] -= _l
            remainders[i] -= _l
    _g = nx.from_numpy_matrix(out, create_using=nx.DiGraph())
    _nnodes = _g.number_of_nodes()
    _in_degree = sum(d for n, d in _g.in_degree()) / _nnodes
    print('%.2f' % _in_degree)
    return out, remainders_liability

if __name__ == '__main__':
    import time
    import pandas as pd
    import matplotlib.pyplot as plt

    snl_data = pd.read_csv('data/snl_data_Q4_2015.csv')

    def get_snl(x, idx):
        return snl_data[x][idx] / 1000
    bank_names = [i.split()[0] for i in open('data/EBA_2016.csv', 'r').read().strip().split('\n')[1:]]
    banned_banks = ['AT02', 'NL33'] + ['DE21'] + ['FR09', 'DE22', 'NL35']
    interbanks = [get_snl('224934', i) for i in range(51) if bank_names[i] not in banned_banks]
    interbank_liabilities = [get_snl('224953', i) for i in range(51) if bank_names[i] not in banned_banks]
    #print(interbanks)
    #print(interbank_liabilities)

    #interbank_matrix = generate_montagna_l1_network_alternative_implementation(interbanks, interbanks)
    tic = time.time()
    interbank_matrix = generate_montagna_l1_network(interbanks, interbank_liabilities)
    tic2 = time.time()
    print('first', tic2 - tic)
    interbank_matrix = generate_montagna_l1_network(interbanks, interbank_liabilities)
    tic3 = time.time()
    print('second', tic3 - tic2)
    exit()
    W_govbonds, W_overlap_govbonds = generate_montagna_l3_network(20)
    W_corpbonds, W_overlap_corpbonds = generate_montagna_l3_network(200)

    #W_equities, W_overlap_equities = generate_montagna_l3_network(200)

    # topology analysis
    degrees = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if i != j:
                # include both in-degree and out-degree
                degrees[i] += 1 if interbank_matrix[i][j] > 0 else 0
                degrees[i] += 1 if interbank_matrix[j][i] > 0 else 0

    # average degree of nearest neighbors
    knn = [[] for i in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                # include both in-degree and out-degree
                if interbank_matrix[i][j] > 0:
                    knn[i].append(degrees[j])
                if interbank_matrix[j][i] > 0:
                    knn[i].append(degrees[j])
    # average out the degrees of the neighbors
    knn = [np.mean(i) for i in knn]

    fig9 = plt.figure(9)
    plt.hist(degrees, bins=list(range(0, 50, 2)), normed=True)
    plt.title("Heterogeneity of Degree Distribution of Interbank (short-term) Network")
    plt.xlabel("total degree")
    plt.ylabel("density")
    plt.savefig('figure9_degree_distribution.png')

    fig10 = plt.figure(10)
    fig10.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.scatter(degrees, knn)
    plt.title("Dissociative behaviour in interbank network")
    plt.xlabel("total degree")
    plt.ylabel("$K_{nn}$")
    plt.savefig("figure10_dissociative_behaviour.png")

    plt.figure(11)
    m = np.delete(interbank_matrix, 11, 0)
    m = np.delete(m, 11, 1)
    m = np.delete(m, 12, 0)
    m = np.delete(m, 12, 1)
    m = np.sqrt(np.sqrt(np.sqrt(m)))
    G = nx.from_numpy_matrix(m)
    #edges = G.edges()
    #weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw(G, with_labels=True)  # , width=weights)
    plt.savefig('plots/graph.png')

    plt.figure(12)
    G_govbonds = nx.from_numpy_matrix(np.power(W_overlap_govbonds, 1 / 16))
    nx.draw(G_govbonds, with_labels=True)
    plt.savefig('plots/graph_govbonds.png')

    plt.figure(13)
    G_corpbonds = nx.from_numpy_matrix(np.power(W_overlap_corpbonds, 1 / 16))
    nx.draw(G_corpbonds, with_labels=True)
    plt.savefig('plots/graph_corpbonds.png')
