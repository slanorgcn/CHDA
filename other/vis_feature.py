import matplotlib.pyplot as plt
from dhg.random import hypergraph_Gnm
from dhg.random import uniform_hypergraph_Gnm
# hg = uniform_hypergraph_Gnm(3, 20, 5)
hg = hypergraph_Gnm(10, 8, method='low_order_first')
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
hg.draw(v_label=labels, font_size=1.5, font_family='serif')
plt.show()