import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from scipy.special import gamma, gammaln
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from robust_parallel import robust_map

POPUL_INIT_SIZE = 100
PERCENT_FLIP = 0.05
PERCENT_CROSS = 0.05
EDGE_COST = -5
MAX_PARENTS = 2
NGEN = 1000

def set_edge(graph, G, i, j, val):
    G[i][j] = val
    try:
        if G[i][j]:
            graph.add_edge(j,i)
        else:
            graph.remove_edge(j,i)
    except:
        pass

def flip_edge(graph, G, i, j):
    val = int(not G[i][j])
    set_edge(graph, G, i, j, val)

def contains_cycle(graph):
    try:
        cycle = nx.find_cycle(graph)
        return True
    except:
        return False

def init_population(pcls, D, num_vars):
    population = []
    corr = D.corr()
    names = D.columns
    for i in range(POPUL_INIT_SIZE):
        G = np.zeros([num_vars,num_vars])
        graph = nx.DiGraph(G)
        for v in range(num_vars):
            #init at least one edge for each var
            name = names[v]
            num_parents = min(np.random.randint(1, num_vars-2),MAX_PARENTS)
            parent_options = list(set(range(num_vars))-{v})
            p = 0
            tries = 0
            while p < num_parents and parent_options:
                # Correlation is not a causation, but it's a good start!
                probs = np.array([abs(corr[name][option])/100.0 for option in parent_options])
                probs = probs/sum(probs)
                parent = np.random.choice(parent_options, p=probs)
                set_edge(graph, G, v, parent, 1)
                if contains_cycle(graph):
                    set_edge(graph, G, v, parent, 0)
                    parent_options = list(set(parent_options)-{parent})
                else:
                    p+=1
        ind = creator.Individual((graph,G))
        population.append(ind)

    return pcls(population)

def bayes_net_mutate(ind):
    graph, G = ind
    G_org = np.copy(G)
    num_vars = len(G)
    num_flip = int(PERCENT_FLIP*(num_vars**2))
    # Flip equal amounts of bits to not get overly complex nets
    indeces_zero = np.where(G == 0)
    indeces_one = np.where(G == 1)
    f = 0
    while f < int(num_flip/2) and len(indeces_zero[0])>0:
        idx = np.random.choice(range(len(indeces_zero[0])))
        i,j = indeces_zero[0][idx],indeces_zero[1][idx]
        flip_edge(graph, G, i, j)
        if contains_cycle(graph):
            flip_edge(graph, G, i, j)
        else:
            f+=1
    f = 0
    while f < int(num_flip/2) and len(indeces_one[0])>0:
        idx = np.random.choice(range(len(indeces_one[0])))
        i,j = indeces_one[0][idx],indeces_one[1][idx]
        flip_edge(graph, G, i, j)
        if contains_cycle(graph):
            flip_edge(graph, G, i, j)
        else:
            f+=1
    return ind,

def bayes_net_crossover(ind1,ind2):
    graph1, G1 = ind1
    graph2, G2 = ind2
    num_vars = len(G1)
    num_cross = int(PERCENT_CROSS*(num_vars**2))
    indeces = np.where(G1 != G2)
    c = 0
    while c < int(num_cross/2) and len(indeces[0])>0:
        idx = np.random.choice(range(len(indeces[0])))
        i,j = indeces[0][idx],indeces[1][idx]
        val1 = G1[i][j]
        val2 = G2[i][j]
        set_edge(graph1, G1, i, j, val2)
        set_edge(graph2, G2, i, j, val1)
        if contains_cycle(graph1):
            set_edge(graph1, G1, i, j, val1)
        else:
            c+=1
        if contains_cycle(graph2):
            set_edge(graph2, G2, i, j, val2)
        else:
            c+=1
    return ind1, ind2

def score(ind, D, cardinalities, names):
    """
    ind is a networkx graph and a numpy adjacency matrix
    D is a pandas df
    cardinalities is a numpy array
    names is a list
    """
    #can ignore ln(G) since always 0
    graph, G = ind
    num_edges = len(np.nonzero(G)[0])
    score = EDGE_COST*num_edges
    for i in xrange(len(G)):
        parent_idxs = np.nonzero(G[i])[0]
        def score_for_all_j(D_subset, current_parent):
            "Implement as function to more efficiently select in D"
            score = 0
            if current_parent == len(parent_idxs):
                m_ij0 = len(D_subset)
                a_ij0 = cardinalities[i]
                prior_term = gammaln(a_ij0) - gammaln(a_ij0+m_ij0)

                posterior_term = 0
                for k in range(1,cardinalities[i]+1):
                    relevant = D_subset[D_subset[names[i]]==k]
                    m_ijk = len(relevant)
                    #can ignore gammaln(aijk) since always 0
                    posterior_term+=gammaln(1+m_ijk)

                score = prior_term + posterior_term
            else:
                parent_idx = parent_idxs[current_parent]
                for assignment in range(1,cardinalities[parent_idx]+1):
                    D_subset = D_subset[D_subset[names[parent_idx]]==assignment]
                    score+=score_for_all_j(D_subset, current_parent+1)
            return score
        score+= score_for_all_j(D, 0)
    return score

def write_drawing(ind, idx2names, filename): 
    graph, G = ind
    d = {}
    for idx in range(len(idx2names)):
        d[idx] = idx2names[idx]
    graph = nx.relabel_nodes(graph, d, copy=False)
    #nx.draw(graph,with_labels=True)
    #plt.show()
    nx.draw(graph,with_labels=True)
    plt.savefig(filename.replace('gph','png'))

    with open(filename, 'w') as f:
        for i in xrange(len(G)):
            for j in xrange(len(G)):
                if G[i][j]!=0:
                    f.write("{}, {}\n".format(idx2names[j], idx2names[i]))

def compute(infile, outfile):
    D = pd.read_csv(infile)
    names = D.columns
    num_vars = len(names)
    cardinalities = D.max().tolist()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", tuple, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("population_guess", init_population, list, D, num_vars)
    population = toolbox.population_guess()

    toolbox.register("mate", bayes_net_crossover)
    toolbox.register("mutate", bayes_net_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda G: [score(G, D, cardinalities, names)])

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    toolbox.register("map", robust_map)
    hof = tools.HallOfFame(1, np.array_equal)
    population,logbook = algorithms.eaSimple(population, toolbox, 
                                 cxpb=0.5, mutpb=0.2, ngen=NGEN,
                                 stats=stats, halloffame=hof, verbose=True)
    winner = hof[0]
    write_drawing(winner, names, outfile)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()

