import pandas as pd

cities = {
    'Вильнюс': 0,
    'Брест': 1,
    'Витебск': 2,
    'Воронеж': 3,
    'Волгоград': 4,
    'Ниж.Новгород': 5,
    'Даугавпилс': 6,
    'Калининград': 7,
    'Каунас': 8,
    'Киев': 9,
    'Житомир': 10,
    'Донецк': 11,
    'Кишинев': 12,
    'С.Петербург': 13,
    'Рига': 14,
    'Москва': 15,
    'Казань': 16,
    'Минск': 17,
    'Мурманск': 18,
    'Орел': 19,
    'Одесса': 20,
    'Таллин': 21,
    'Харьков': 22,
    'Симферополь': 23,
    'Ярославль': 24,
    'Уфа': 25,
    'Самара': 26
}

cities1 = {
    'Рига': 0,
    'Хельсинки': 1,
    'Таллин': 2,
    'С.Петербург': 3,
    'Тверь': 4,
    'Москва': 5,
}

cities_reverse = {
    0: 'Вильнюс',
    1: 'Брест',
    2: 'Витебск',
    3: 'Воронеж',
    4: 'Волгоград',
    5: 'Ниж.Новгород',
    6: 'Даугавпилс',
    7: 'Калининград',
    8: 'Каунас',
    9: 'Киев',
    10: 'Житомир',
    11: 'Донецк',
    12: 'Кишинев',
    13: 'С.Петербург',
    14: 'Рига',
    15: 'Москва',
    16: 'Казань',
    17: 'Минск',
    18: 'Мурманск',
    19: 'Орел',
    20: 'Одесса',
    21: 'Таллин',
    22: 'Харьков',
    23: 'Симферополь',
    24: 'Ярославль',
    25: 'Уфа',
    26: 'Самара'
}

cities_reverse1 = {
    0: 'Рига',
    1: 'Хельсинки',
    2: 'Таллин',
    3: 'СПБ',
    4: 'Тверь',
    5: 'Москва',
}

dist = [529, 819, 575, 1250, 1745, 1185, 410, 584, 507, 1064, 1051, 1527, 1406, 317, 279, 868, 1497, 639, 1132, 1002,
        1494, 0, 1280, 1729, 900, 1937, 1697]

dist1 = [839, 897, 864, 634, 161, 0]


def dfs(g, v, f, visited, path):
    visited[v] = 1
    path.append(cities_reverse[v])
    if v == f:
        return True
    for i in g[v]:
        if visited[i[0]] == 0:
            if dfs(g, i[0], f, visited, path):
                return True
    path.remove(cities_reverse[v])
    return False


def bfs(g, v, f, visited1, path1):
    queue = []
    queue.insert(0, v)
    visited1[v] = 1
    while len(queue) > 0:
        v = queue.pop(0)
        for i in g[v]:
            if visited1[i[0]] == 0:
                queue.append(i[0])
                visited1[i[0]] = 1
                path1[i[0]] = v
            if i[0] == f:
                return True
    return False


def dls(g, v, f, depth, max_depth, visited2, path):
    if depth > max_depth:
        return False
    visited2[v] = 1
    if v == f:
        path.append(cities_reverse[v])
        return True
    for i in g[v]:
        if visited2[i[0]] == 0:
            if dls(g, i[0], f, depth + 1, max_depth, visited2, path):
                path.insert(0, cities_reverse[v])
                return True
    visited2[v] = 0
    return False


def iddfs(g, v, f, depth, path):
    while not dls(g, v, f, 0, depth, [0 for i in range(0, len(cities))], path):
        depth = depth + 1
        if depth > len(cities):
            return False
    return depth


def bds(g, v, f, visited1, path1, path2, q):
    queue1 = []
    queue2 = []
    queue1.append(v)
    queue2.append(f)
    while len(queue1) > 0 and len(queue2) > 0:
        v1 = queue1.pop(0)
        v2 = queue2.pop(0)
        visited1[v1] = 1
        visited1[v2] = 1

        for i in g[v1]:
            if i[0] == f:
                q.append(i[0])
            if path2[i[0]] != -1:
                path1[i[0]] = v1
                q.append(i[0])
                return True
            if visited1[i[0]] == 0:
                queue1.append(i[0])
                path1[i[0]] = v1

        for i in g[v2]:
            if path1[i[0]] != -1:
                path2[i[0]] = v2
                q.append(i[0])
                return True
            if visited1[i[0]] == 0:
                queue2.append(i[0])
                path2[i[0]] = v2
    return False


def not_inform_search(graph, s, f):
    path = []
    dfs(graph, cities[s], cities[f], [0 for i in range(0, len(cities))], path)
    print("DFS:")
    if len(path) > 0:
        print(path)
        path.clear()

    path1 = [-1 for i in range(0, len(cities))]
    bfs(graph, cities[s], cities[f], [0 for i in range(0, len(cities))], path1)
    f1 = cities[f]
    print("BFS:")
    if path1[f1] != -1:
        path.append(cities_reverse[f1])
        while path1[f1] != -1:
            f1 = path1[f1]
            path.insert(0, cities_reverse[f1])
        print(path)
        path.clear()
        path1.clear()

    dls(graph, cities[s], cities[f], 0, 10, [0 for i in range(0, len(cities))], path)
    print("DLS:")
    if len(path) > 0:
        print(path)
        path.clear()

    iddfs(graph, cities[s], cities[f], 0, path)
    print("IDDFS:")
    if len(path) > 0:
        print(path)
        path.clear()
        path1.clear()

    path1 = [-1 for i in range(0, len(cities))]
    path2 = [-1 for i in range(0, len(cities))]
    q = []
    bds(graph, cities[s], cities[f], [0 for i in range(0, len(cities))], path1, path2, q)
    print("Bi_Direction_Search:")
    if len(q) > 0:
        mid = q[0]
        f1 = mid
        path.append(cities_reverse[f1])
        while f1 != cities[s]:
            f1 = path1[f1]
            path.insert(0, cities_reverse[f1])
        f1 = mid
        while f1 != cities[f]:
            f1 = path2[f1]
            path.append(cities_reverse[f1])
        print(path)


def best_first_search(g, v, f, visited, path):
    path.append(cities_reverse[v[0]])
    if v[0] == f:
        print(v[1])
        return True
    v1 = v
    visited[v[0]] = 1
    queue = []
    for i in g[v1[0]]:
        if visited[i[0]] == 0:
            if len(queue) > 0:
                for j in range(0, len(queue)):
                    if dist[i[0]] < queue[j][1]:
                        queue.insert(j, [i[0], dist[i[0]], i[1] + v1[1]])
                        break
                    if j == len(queue) - 1:
                        queue.append([i[0], dist[i[0]], i[1] + v1[1]])
            else:
                queue.append([i[0], dist[i[0]], i[1] + v1[1]])
    while len(queue) > 0:
        v2 = queue.pop(0)
        if best_first_search(g, [v2[0], v2[2]], f, visited, path):
            return True
    path.remove(cities_reverse[v[0]])
    return False


def a_star(g, v, f, visited, path, d):
    path.append(cities_reverse[v[0]])
    if v[1] + dist[v[0]] > d[v[0]]:
        path.remove(cities_reverse[v[0]])
        return False
    if v[0] == f:
        print(v[1])
        return True
    v1 = v
    visited[v[0]] = 1
    queue = []
    for i in g[v1[0]]:
        if visited[i[0]] == 0:
            s = v1[1] + i[1] + dist[i[0]]
            if len(queue) > 0:
                for j in range(0, len(queue)):
                    if s < queue[j][1]:
                        queue.insert(j, [i[0], s, i[1] + v1[1]])
                        break
                    if j == len(queue) - 1:
                        queue.append([i[0], s, i[1] + v1[1]])
            else:
                queue.append([i[0], s, i[1] + v1[1]])
            if d[i[0]] > s:
                d[i[0]] = s
    while len(queue) > 0:
        v2 = queue.pop(0)
        if a_star(g, [v2[0], v2[2]], f, visited, path, d):
            return True
    path.remove(cities_reverse[v[0]])
    return False


def inform_search(graph, s, f):
    path = []
    print("Best_First_Search:")
    best_first_search(graph, [cities[s], 0], cities[f], [0 for i in range(0, len(cities))], path)
    print(path)
    path.clear()
    print("A*:")
    a_star(graph, [cities[s], 0], cities[f], [0 for i in range(0, len(cities))], path,
                 [10000 for i in range(0, len(cities))])
    print(path)


def main():
    file = pd.read_csv("path.csv", header=None)
    graph = []

    for i in range(0, len(cities)):
        graph.append([])

    for i in range(0, len(file)):
        x = file.iloc[i][0]
        y = file.iloc[i][1]
        z = file.iloc[i][2]
        graph[cities[x]].append([cities[y], z])
        graph[cities[y]].append([cities[x], z])

    f = 'Таллин'
    s = 'Казань'
    not_inform_search(graph, s, f)
    inform_search(graph, s, f)


main()
