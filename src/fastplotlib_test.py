"""
Scatter Plot
============
Example showing scatter plot.
"""

# test_example = true

import fastplotlib as fpl
import numpy as np
from pathlib import Path

plot = fpl.Plot()
# to force a specific framework such as glfw:
# plot = fpl.Plot(canvas="glfw")

data_path = Path(__file__).parent.parent.joinpath("data", "iris.npy")
data = np.load(data_path)

n_points = 50
colors = ["yellow"] * n_points + ["cyan"] * n_points + ["magenta"] * n_points

scatter_graphic = plot.add_scatter(data=data[:, :-1], sizes=6, alpha=0.7, colors=colors)

plot.show()

plot.canvas.set_logical_size(800, 800)

plot.auto_scale()

input("Press Enter to continue...")
for i in range(100):
    scatter_graphic.data[:, 0] += 0.01
    plot.render()
    plot.present()
    input("Press Enter to continue...")
    # plot.renderer.render(plot.canvas, plot.camera)


if __name__ == "__main__":
    print(__doc__)
    fpl.run()
