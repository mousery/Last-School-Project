from main import *
from matplotlib import pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

handle_base_arc = ArcEdge(
    start=[0,0,0],
    end=[17,17,0],
    center=[17,0,0]
)

handle_base_straight = StraightEdge(
    start=[17,17,0],
    end=[43,17,0]
)

handle_base_arc2 = ArcEdge(
    start=[43,17,0],
    end=[60,0,0],
    center=[43,0,0]
)

handle_base_combined = CombinedEdge([handle_base_arc,
                                     handle_base_straight,
                                     handle_base_arc2])

handle_base_combined.plot_all(ax)

plt.show()
