import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False





class TS(object):
    def __init__(self, num_city, data):
        self.taboo_size = 5
        self.iteration = 500
        self.num_city = num_city
        self.location = data
        self.taboo = []

        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.path = self.greedy_init(self.dis_mat,100,num_city)
        self.best_path = self.path
        print("Best path is: ", self.best_path)
        self.best_lengthNN = self.compute_pathlen(self.path, self.dis_mat)
        print("Best length from NN: ", self.best_lengthNN)

        #Using christofides to compute the approximation
        self.chris_path = self.christofides(self.dis_mat, num_city)
        print("Christofides' path: ", self.chris_path)
        self.best_lengthCH = self.compute_pathlen(self.chris_path[:-1], self.dis_mat)
        print("the best length using christofides: ",self.best_lengthCH)
        self.cur_path = self.path
        self.best_length = self.compute_pathlen(self.path, self.dis_mat)

        init_pathlen = 1. / self.compute_pathlen(self.path, self.dis_mat)
        self.iter_x = [0]
        self.iter_y = [1. / init_pathlen]
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        
        return result[index]

    def christofides(self, dis_mat, num_city):
        # Step 1: Compute the Minimum Spanning Tree (MST)
        graph = nx.Graph()

        for i in range(0, num_city):
            for j in range(i+1, num_city):
                graph.add_edge(i, j, weight=dis_mat[i][j])

        mst = nx.minimum_spanning_tree(graph)

        # Step 2: Find nodes with odd degree in the MST
        odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]

        # Step 3: Find Minimum Weight Perfect Matching among odd degree nodes
        subgraph = graph.subgraph(odd_degree_nodes)
        matching = nx.algorithms.matching.min_weight_matching(subgraph)

        # Step 4: Combine MST and Matching to form an Eulerian Graph
        eulerian_graph = nx.MultiGraph(mst)
        eulerian_graph.add_edges_from(matching)

        # Step 5: Find an Eulerian circuit
        eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))

        # Step 6: Shortcutting to form the final TSP tour
        tsp_tour = []
        visited = set()
        for u, v in eulerian_circuit:
            if u not in visited:
                tsp_tour.append(u)
                visited.add(u)
        tsp_tour.append(tsp_tour[0])  # Return to starting point

        return tsp_tour

    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        random.shuffle(tmp)
        return tmp

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    def ts_search(self, x):
        moves = []
        new_paths = []
        while len(new_paths)<400:
            i = np.random.randint(len(x))
            j = np.random.randint(len(x))
            tmp = x.copy()
            tmp[i:j] = tmp[i:j][::-1]
            new_paths.append(tmp)
            moves.append([i, j])
        return new_paths, moves

    def ts(self):
        for cnt in range(self.iteration):
            new_paths, moves = self.ts_search(self.cur_path)
            new_lengths = self.compute_paths(new_paths)
            sort_index = np.argsort(new_lengths)
            min_l = new_lengths[sort_index[0]]
            min_path = new_paths[sort_index[0]]
            min_move = moves[sort_index[0]]

            if min_l < self.best_length:
                self.best_length = min_l
                self.best_path = min_path
                self.cur_path = min_path
                if min_move in self.taboo:
                    self.taboo.remove(min_move)

                self.taboo.append(min_move)
            else:
                while min_move in self.taboo:
                    sort_index = sort_index[1:]
                    min_path = new_paths[sort_index[0]]
                    min_move = moves[sort_index[0]]
                self.cur_path = min_path
                self.taboo.append(min_move)
            if len(self.taboo) > self.taboo_size:
                self.taboo = self.taboo[1:]
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_length)
            print(cnt, self.best_length)
        print(self.best_length)

    def run(self):
        self.ts()
        return self.location[self.best_path], self.best_length


def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSP_PATH = os.path.join(BASE_DIR, "ch150.tsp")
data = read_tsp(TSP_PATH)

data = np.array(data)
plt.suptitle('TS in ch150.tsp')
data = data[:, 1:]
plt.subplot(2, 2, 1)
plt.title('raw data')
show_data = np.vstack([data, data[0]])
plt.plot(data[:, 0], data[:, 1])

model = TS(num_city=data.shape[0], data=data.copy())


# Best_path, Best_length = model.run()

# Best_path = np.vstack([Best_path, Best_path[0]])
# fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
# axs[0].scatter(Best_path[:, 0], Best_path[:,1])
# Best_path = np.vstack([Best_path, Best_path[0]])
# axs[0].plot(Best_path[:, 0], Best_path[:, 1])
# axs[0].set_title('Planning Results')
# iterations = model.iter_x
# best_record = model.iter_y
# axs[1].plot(iterations, best_record)
# axs[1].set_title('Planning Outcomes')
# plt.show()