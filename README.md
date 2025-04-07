# A* 寻路算法可视化项目

## 项目简介
这是一个基于Python实现的A*寻路算法可视化项目。该项目在网格地图上实现了A*寻路算法，并提供了实时的可视化效果，可以直观地展示算法的搜索过程和最终路径。

## 算法原理
A*算法是一种启发式搜索算法，它结合了Dijkstra算法的优点和最佳优先搜索的思想。在本项目中：
- 使用曼哈顿距离作为启发函数
- 支持网格地图中的障碍物处理
- 实时展示搜索过程和路径生成

## 功能特点
- 支持自定义网格大小
- 可设置障碍物位置
- 动态可视化搜索过程
- 显示节点访问顺序
- 高亮显示最终路径
- 计算并显示最短路径长度

## 安装使用

### 依赖库
```bash
pip install matplotlib networkx
```

### 使用方法
1. 创建网格图实例：
```python
graph = GridGraph(width, height, obstacles)
```

2. 设置起点和终点：
```python
start = (0, 0)
goal = (3, 4)
```

3. 运行算法并可视化：
```python
came_from, cost_so_far, search_steps = a_star_search(graph, start, goal, heuristic)
ani = visualize_search(graph, search_steps, start, goal)
plt.show()
```

## 代码结构

### 核心类
- `GridGraph`: 网格图的实现
  - `create_grid()`: 创建网格并初始化边和权重
  - `neighbors()`: 获取节点的邻居节点
  - `cost()`: 计算节点间的移动成本

### 主要函数
- `a_star_search()`: A*算法的核心实现
- `heuristic()`: 启发函数（曼哈顿距离）
- `visualize_search()`: 搜索过程可视化
- `merge_dicts()`: 辅助函数，用于合并字典

## 示例
```python
# 设置障碍物
obstacles = [(1, 1), (2, 2), (3, 3)]

# 创建5x5网格
graph = GridGraph(5, 5, obstacles)

# 设置起点和终点
start = (0, 0)
goal = (3, 4)

# 运行算法
came_from, cost_so_far, search_steps = a_star_search(graph, start, goal, heuristic)

# 可视化结果
ani = visualize_search(graph, search_steps, start, goal)
plt.show()
```

## 可视化说明
- 绿色节点：起点
- 红色节点：终点
- 浅蓝色节点：已访问节点
- 红色箭头：搜索过程
- 蓝色箭头：最终路径
- 节点中的数字：从起点到该节点的实际代价
