import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_fea_model(file_path, label_interval):
    # 读取节点数据
    with open(file_path, 'r') as f:
        content = f.read()
    node_data_start = content.find('*Node') + len('*Node\n')
    node_data_end = content.find('*Element, type=S4R')
    node_data = content[node_data_start:node_data_end].strip()
    node_lines = node_data.split('\n')
    nodes = []
    for line in node_lines:
        parts = line.strip().split(',')
        node = [float(part) for part in parts]
        nodes.append(node)

    # 读取单元数据
    elements = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_read = False
        for line in lines:
            if line.startswith('*Element'):
                start_read = True
                continue
            if line.startswith('*'):
                start_read = False
            if start_read:
                element = list(map(int, line.split(',')))
                elements.append(element[1:])

    # 绘制模型#15, 15
    fig, ax = plt.subplots(figsize=(10, 4))
    for element in elements:
        n1, n2, n3, n4 = [nodes[i-1] for i in element]
        polygon = patches.Polygon([n1[1:3], n2[1:3], n3[1:3], n4[1:3]], closed=True, fill=False)
        ax.add_patch(polygon)

    for i, node in enumerate(nodes):
        if i % label_interval == 0:
            ax.scatter(*node[1:3], color='red')
            ax.annotate(f'{node[0]:.0f}', (node[1], node[2]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_xlim(min(node[1] for node in nodes), max(node[1] for node in nodes))
    ax.set_ylim(min(node[2] for node in nodes), max(node[2] for node in nodes))
    plt.show()

    return nodes

# 使用封装好的函数
# file_path = "Job-1_largemesh.inp"
# label_interval = 2
# plot_fea_model(file_path, label_interval)
