'''
 Implementing Nearest Neighbor Heuristic to solve Traveling Salesman problems
'''

def tspNearestNeighborHeuristic(cities, distance):
    visited = []
    minimum_distance_traveled = []

    neighbor = 'A'
    start_node_index = cities.index(neighbor)

    
    no_nodes = len(cities)
    noN = 0
    while noN < no_nodes and neighbor not in visited:

        visited.append(neighbor)
        neighbor_index = cities.index(neighbor)
        noNeigjbour = 0
        MIN = 0

        while noNeigjbour < len(distance[neighbor_index]):

            if cities[noNeigjbour] not in visited: #look for unvisitied cities.
                if MIN == 0:
                    MIN = distance[neighbor_index][noNeigjbour]
                    neighbor = cities[noNeigjbour]
                else:
                    min_distance = min(distance[neighbor_index][noNeigjbour], MIN)
                    if distance[neighbor_index][noNeigjbour] < MIN:
                        MIN = min_distance
                        neighbor = cities[noNeigjbour]
            noNeigjbour += 1
        minimum_distance_traveled.append(MIN)
        noN += 1
    last_node_index = cities.index(visited[-1])
    minimum_distance_traveled[-1] = distance[last_node_index][start_node_index]
    print('Shortest route : ', " -> ".join(visited))
    # for _i in range(len(visited)):
    #     print("City " + visited[_i] + "'s Nearest Neighbor's Distance is ", minimum_distance_traveled[_i])
    print("total traveled distance: ", sum(minimum_distance_traveled))

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


data = read_tsp('ch150.tsp')

cities = ['A', 'B', 'C', 'D', 'E']
distance = [[0, 60, 217, 164, 69],
            [60, 0, 290, 201, 79],
            [217, 290, 0, 113, 303],
            [164, 201, 113, 0, 196],
            [69, 79, 303, 196, 0]]

tspNearestNeighborHeuristic(cities, distance)