import matplotlib.pyplot as plt
import mistree
import numpy as np
import torch
from fast_pytorch_kmeans import KMeans


def plot_tour(points, pi):
    labels = KMeans(points.size(0) // 100, max_iter=50).fit_predict(points)
    assert len(points.shape) == 2
    assert len(pi.shape) == 1
    # generate dgl graph
    x = points[:, 0].cpu().numpy()
    y = points[:, 1].cpu().numpy()
    plt.scatter(x, y, s=5, c=labels, cmap="tab20c")
    # for i in range(x.shape[0]):
    #   plt.text(x[i]+0.001, y[i]+0.001, s=str(labels[i].item()), fontsize=5)
    # for (l, r) in torch.stack([pi, pi.roll(-1)]).t().cpu().numpy():
    #     plt.arrow(x[l],
    #               y[l],
    #               x[r] - x[l],
    #               y[r] - y[l],
    #               length_includes_head=True,
    #               head_width=0.005,
    #               linewidth=0.5)
    mst = mistree.GetMST(x=x, y=y)
    d, l, b, s, l_index, b_index = mst.get_stats(include_index=True)
    plt.plot(
        [x[l_index[0]], x[l_index[1]]],
        [y[l_index[0]], y[l_index[1]]],
        color="k",
        linewidth=0.5,
    )
    plt.savefig("visualization.jpg", dpi=1000)


all_points = torch.load("./train_instances.pt")
all_pi = torch.load("./train_solutions.pt")
plot_tour(all_points[10], all_pi[10])
