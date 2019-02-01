import networkx
import networkx as nx
import json
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

G_data = json.load(open('./json_graphs/right_test_thro/right_test-G.json'))
G = json_graph.node_link_graph(G_data)
nx.draw_networkx(G, node_size=5,with_labels=False)
plt.draw()
plt.show()
