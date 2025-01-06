import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue
from matplotlib.animation import FuncAnimation

class GridGraph:
    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.edges = {}
        self.weights = {}
        self.obstacles = obstacles if obstacles else []
        self.create_grid()

    def create_grid(self):
        # 创建网格图，初始化边和权重
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) in self.obstacles:
                    continue
                self.edges[(x, y)] = []
                if x > 0 and (x-1, y) not in self.obstacles:
                    self.edges[(x, y)].append((x-1, y))
                    self.weights[((x, y), (x-1, y))] = 1
                if x < self.width - 1 and (x+1, y) not in self.obstacles:
                    self.edges[(x, y)].append((x+1, y))
                    self.weights[((x, y), (x+1, y))] = 1
                if y > 0 and (x, y-1) not in self.obstacles:
                    self.edges[(x, y)].append((x, y-1))
                    self.weights[((x, y), (x, y-1))] = 1
                if y < self.height - 1 and (x, y+1) not in self.obstacles:
                    self.edges[(x, y)].append((x, y+1))
                    self.weights[((x, y), (x, y+1))] = 1

    def neighbors(self, node):
        # 返回节点的邻居
        return self.edges[node]

    def cost(self, from_node, to_node):
        # 返回从一个节点到另一个节点的成本
        return self.weights[(from_node, to_node)]

def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()  # 复制第一个字典
    merged_dict.update(dict2)   # 更新第二个字典的键值对
    return merged_dict

def a_star_search(graph, start, goal, heuristic):
    # A*算法实现
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    search_steps = []

    while not frontier.empty():
        current_priority, current = frontier.get()
        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current
                # 记录当前状态
                search_steps.append((dict(came_from), dict(cost_so_far)))
        filtered_dict_came_from = dict()
        filtered_dict_cost_so_far = dict()
        search_steps_new = []
        example_dict = search_steps[-1][-1]
        values_max = max(list(example_dict.values()))+1
        for num in range(values_max):
            example_dict = search_steps[-1][-1]
            key_dic = [key for key, value in example_dict.items() if value == num]
            filtered_cost_so_far = {key: value for key, value in example_dict.items() if value == num}
            example_dict1 = search_steps[-1][0]
            filtered_came_from = {key: value for key, value in example_dict1.items() if key in key_dic}
            filtered_dict_cost_so_far = merge_dicts(filtered_dict_cost_so_far, filtered_cost_so_far)
            filtered_dict_came_from = merge_dicts(filtered_dict_came_from, filtered_came_from)
            search_steps_new.append((filtered_dict_came_from, filtered_dict_cost_so_far))
    return came_from, cost_so_far, search_steps_new

def heuristic(a, b):
    # 使用曼哈顿距离作为启发函数
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def visualize_search(graph, search_steps, start, goal):
    # 可视化搜索过程
    G = nx.Graph()  # 使用有向图
    GG = nx.DiGraph()
    for node in graph.edges:
        for neighbor in graph.edges[node]:
            G.add_edge(node, neighbor, weight=graph.weights[(node, neighbor)])

    pos = {node: (node[0], -node[1]) for node in G.nodes()}  # 使用网格坐标
    fig, ax = plt.subplots(figsize=(10, 10))


    def update(num):
        # 更新动画帧
        ax.clear()
        nx.draw(G, pos, with_labels=False, node_size=700, node_color='lightblue', font_size=15, font_weight='bold')
        if num < len(search_steps):
            came_from, cost_so_far = search_steps[num]
            edges = [(came_from[node], node) for node in came_from if came_from[node] is not None]
            nx.draw_networkx_edges(GG, pos, edgelist=edges, edge_color='r', width=2, arrows=True, arrowsize=20)
            node_colors = []
            for node in G.nodes():
                if node == start:
                    node_colors.append('green')
                elif node == goal:
                    node_colors.append('red')
                elif node in came_from:
                    node_colors.append('lightblue')
                else:
                    node_colors.append('lightblue')
            nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_colors, node_size=700)
            labels = {node: f"{cost_so_far[node]}" for node in cost_so_far}
            nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')
        plt.title("A* Search Visualization")

    ani = FuncAnimation(fig, update, frames=len(search_steps), interval=200, repeat=False)
    return ani

# # 定义网格图
# width, height = 15, 15
# graph = GridGraph(width, height)

# start = (2, 1)
# goal = (4, 10)
# came_from, cost_so_far, search_steps = a_star_search(graph, start, goal, heuristic)



# 示例使用
obstacles = [(1, 1), (2, 2), (3, 3)]  # 障碍物位置
graph = GridGraph(15, 15, obstacles)
start = (0, 0)
goal = (10, 4)
came_from, cost_so_far, search_steps = a_star_search(graph, start, goal, heuristic)

print("路径:", came_from)
print("成本:", cost_so_far)
print("搜索步骤:", search_steps[-1])
ani = visualize_search(graph, search_steps, start, goal)
plt.show()








