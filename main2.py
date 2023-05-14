import numpy as np
from IPython.core.pylabtools import figsize

from main import *
from matplotlib import pyplot as plt
from copy import deepcopy
from skimage import measure
import open3d as o3d

handle_base = BSplineEdge(ccp=[
    [0, 0, 0],
    [17, 17, 0],
    [30, 17, 0],
    [43, 17, 0],
    [60, 0, 0]
])

handle_ceil = BSplineEdge(ccp=[
    [20, 0, 90],
    [23, 17, 90],
    [50, 17, 90],
    [93, 17, 90],
    [95, 0, 90]
])

handle_vert = BSplineEdge(ccp=[
    [0, 0, 0],
    [5, 0, 30],
    [25, 0, 80],
    [20, 0, 90]
])

handle_vert_2 = BSplineEdge(ccp=[
    [60, 0, 0],
    [70, 0, 40],
    [80, 0, 80],
    [85, 0, 90]
])

handle = Surface(handle_base,
                 handle_ceil,
                 handle_vert,
                 handle_vert_2,
                 nu=20, nv=20
                 )

barrel_bottom_back_edge = handle_ceil.points[-6:].tolist()
barrel_bottom_front_edge = StraightEdge(start=[202, 0, 90],
                                        end=[202, 17, 90],
                                        n=6)
barrel_bottom_side_edge = StraightEdge(start=[202, 17, 90],
                                       end=handle_ceil.points[-6].tolist(), n=20)
barrel_bottom_middle_edge = StraightEdge(start=[202, 0, 90],
                                         end=handle_ceil.points[-1].tolist(), n=20)

barrel_bottom = Surface(barrel_bottom_back_edge,
                        barrel_bottom_front_edge,
                        barrel_bottom_side_edge,
                        barrel_bottom_middle_edge,
                        nu=6, nv=20
                        )

barrel_side_back_edge = StraightEdge(start=handle_vert.points[-1].tolist(),
                                     end=[20, 0, 113], n=6)
barrel_side_bottom_edge = CombinedEdge([handle_ceil.points[:-5].tolist(),
                                        barrel_bottom_side_edge]
                                       )
barrel_side_top_edge = deepcopy(barrel_side_bottom_edge)
barrel_side_top_edge.translate([0, 0, 23])
barrel_side_front_edge = StraightEdge(start=barrel_side_bottom_edge.points[-1].tolist(),
                                      end=barrel_side_top_edge.points[-1].tolist(), n=6)

barrel_side = Surface(barrel_side_back_edge,
                      barrel_side_front_edge,
                      barrel_side_bottom_edge,
                      barrel_side_top_edge
                      )

cos_a = np.dot(normalize(barrel_side_top_edge.points[3] - [20, 0, 130]), [0, 0, -1])
a = np.arccos(cos_a)
b = np.pi - 2 * a
z = np.sqrt(dist(barrel_side_top_edge.points[3] - [20, 0, 130]) ** 2 / 2 / (1 - np.cos(b)))

# barrel_top_back_edge = BSplineEdge(ccp=[barrel_side_top_edge.points[3].tolist(),
#                                         [20, 15, 120],
#                                         [20, 10, 125],
#                                         [20, 0, 130]], n=6)
barrel_top_back_edge = ArcEdge(start=barrel_side_top_edge.points[3].tolist(),
                               end=[barrel_side_top_edge.points[3, 0], 0, 130],
                               center=[20, 0, 130 - z], n=6)
# barrel_side_top_edge.points[:6].tolist()
# barrel_top_front_edge = StraightEdge(start=barrel_side_top_edge.points[-1].tolist(),
#                                      end=[202, 0, 130], n=6)
barrel_top_front_edge = ArcEdge(start=barrel_side_top_edge.points[-1].tolist(),
                                end=[202, 0, 130],
                                center=[202, 0, 113], n=6)
barrel_top_side_edge = barrel_side_top_edge.points[3:].tolist()
nv = len(barrel_top_side_edge)
barrel_top_middle_edge = StraightEdge(start=barrel_top_back_edge.points[-1].tolist(),
                                      end=[202, 0, 130], n=nv)

barrel_top = Surface(barrel_top_back_edge,
                     barrel_top_front_edge,
                     barrel_top_side_edge,
                     barrel_top_middle_edge, nu=6, nv=nv)

handle_bottom_side_edge = handle_base.points[6:-6].tolist()
handle_bottom_middle_edge = StraightEdge(start=[0, 0, 0],
                                         end=[60, 0, 0])
handle_bottom_front_edge = handle_base.points[-7:].tolist()
handle_bottom_back_edge = handle_base.points[:7].tolist()

handle_bottom = Surface(handle_bottom_side_edge,
                        handle_bottom_middle_edge,
                        handle_bottom_front_edge,
                        handle_bottom_back_edge
                        )

trigger_guard_side_bottom_edge = CombinedEdge([
    BSplineEdge(ccp=[
        [115, 14, 90],
        [112, 14, 85],
        [112, 14, 70],
        [115, 14, 65]
    ], n=6),
    StraightEdge(start=[115, 14, 65],
                 end=[80, 14, 65],
                 n=3),
    BSplineEdge(ccp=[
        [80, 14, 65],
        [78, 14, 67],
        [72, 14, 73],
        [70, 14, 70]
    ], n=4)
], ccp_as_point=True)

trigger_guard_side_upper_edge = CombinedEdge([
    BSplineEdge(ccp=[
        [104, 14, 90],
        [107, 14, 85],
        [107, 14, 75],
        [104, 14, 70]
    ], n=6),
    StraightEdge(start=[104, 14, 70],
                 end=[85, 14, 70],
                 n=3),
    BSplineEdge(ccp=[
        [85, 14, 70],
        [80, 14, 75],
        [78, 14, 80],
        [75, 14, 90]
    ], n=4)
], ccp_as_point=True)

trigger_guard_side_front_edge = StraightEdge(start=[115, 14, 90],
                                             end=[104, 14, 90],
                                             n=4)

trigger_guard_side_back_edge = StraightEdge(start=[70, 14, 70],
                                            end=[70, 14, 90],
                                            n=4)

trigger_guard_side = Surface(trigger_guard_side_bottom_edge,
                             trigger_guard_side_upper_edge,
                             trigger_guard_side_front_edge,
                             trigger_guard_side_back_edge,
                             ccp_as_point=True)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(elev=0, azim=90)
# handle.plot_all(ax)
# handle_base.plot_all(ax)

#  generate point cloud space
length = [101, 11, 51]
x, y, z = np.mgrid[-20:210:length[0] * 1j, 0:30:length[1] * 1j, -10:140:length[2] * 1j]
xs, ys, zs = x[:, 0, 0], y[0, :, 0], z[0, 0, :]
vol = x * 0 + np.inf

for sur in [handle,
            barrel_side,
            barrel_top,
            ]:
    points = sur.points
    for i, _ in enumerate(points[:-1]):
        for j, _ in enumerate(points[0, :-1]):
            # looping through every quadrilateral element on the surface
            quad = points[i:i + 2, j:j + 2]

            # calculate projection of point in point cloud on quad in y-direction,
            # then set vol value = y displacement from projection on surface to point
            for index in np.argwhere(point_in_rect(xs, zs, quad[:, :, [0, 2]])):
                i0, i1 = index[0], index[1]

                pt_x, pt_z = xs[i0], zs[i1]
                u, v = find_uv_in_quad(pt_x, pt_z, quad[:, :, [0, 2]])
                if u is None:
                    continue

                u_vector = (1 - v) * (quad[1, 0] - quad[0, 0]) + v * (quad[1, 1] - quad[0, 1])
                v_vector = (1 - u) * (quad[0, 1] - quad[0, 0]) + u * (quad[1, 1] - quad[1, 0])

                pt = quad[0, 0] + v * (quad[0, 1] - quad[0, 0]) + u * u_vector
                pt_y = pt[1]

                tmp_row = ys - pt_y
                # vol[index[0], :, index[2]] = tmp_row
                vol[i0, :, i1] = np.where(tmp_row < vol[i0, :, i1],
                                          tmp_row,
                                          vol[i0, :, i1])

for sur in [handle]:
    points = sur.points
    for i, _ in enumerate(points[:-1]):
        for j, _ in enumerate(points[0, :-1]):
            # looping through every quadrilateral element on the surface
            quad = points[i:i + 2, j:j + 2]

            # calculate projection of point in point cloud on quad in x-direction,
            # then set vol value = x displacement from projection on surface to point
            for index in np.argwhere(point_in_rect(ys, zs, quad[:, :, [1, 2]])):
                i0, i1 = index[0], index[1]

                pt_y, pt_z = ys[i0], zs[i1]
                u, v = find_uv_in_quad(pt_y, pt_z, quad[:, :, [1, 2]])
                if u is None:
                    continue

                u_vector = (1 - v) * (quad[1, 0] - quad[0, 0]) + v * (quad[1, 1] - quad[0, 1])
                v_vector = (1 - u) * (quad[0, 1] - quad[0, 0]) + u * (quad[1, 1] - quad[1, 0])

                pt = quad[0, 0] + v * (quad[0, 1] - quad[0, 0]) + u * u_vector
                pt_x = pt[0]

                tmp_row = xs - pt_x
                # vol[index[0], :, index[2]] = tmp_row
                vol[:, i0, i1] = np.where(np.abs(tmp_row) < np.abs(vol[:, i0, i1]),
                                          np.sign(vol[:, i0, i1]) * np.abs(tmp_row),
                                          vol[:, i0, i1])

for sur in [barrel_top]:
    points = sur.points
    for i, _ in enumerate(points[:-1]):
        for j, _ in enumerate(points[0, :-1]):
            # looping through every quadrilateral element on the surface
            quad = points[i:i + 2, j:j + 2]

            # calculate projection of point in point cloud on quad in z-direction,
            # then set vol value = z displacement from projection on surface to point
            for index in np.argwhere(point_in_rect(ys, zs, quad[:, :, [1, 2]])):
                i0, i1 = index[0], index[1]

                pt_x, pt_y = ys[i0], zs[i1]
                u, v = find_uv_in_quad(pt_x, pt_y, quad[:, :, [0, 1]])
                if u is None:
                    continue

                u_vector = (1 - v) * (quad[1, 0] - quad[0, 0]) + v * (quad[1, 1] - quad[0, 1])
                v_vector = (1 - u) * (quad[0, 1] - quad[0, 0]) + u * (quad[1, 1] - quad[1, 0])

                pt = quad[0, 0] + v * (quad[0, 1] - quad[0, 0]) + u * u_vector
                pt_z = pt[2]

                tmp_row = xs - pt_x
                # vol[index[0], :, index[2]] = tmp_row
                vol[i0, i1, :] = np.where(np.abs(tmp_row) < np.abs(vol[i0, i1, :]),
                                          np.sign(vol[i0, i1, :]) * np.abs(tmp_row),
                                          vol[i0, i1, :])

vol = np.concatenate((vol[:, ::-1, :], vol),
                     axis=1)[:, ::2, :]

verts, faces, _, _ = measure.marching_cubes(vol, 0,
                                            spacing=(xs[1] - xs[0],
                                                     2 * (ys[1] - ys[0]),
                                                     zs[1] - zs[0]),
                                            step_size=2,
                                            allow_degenerate=False)

verts -= [20, 30, 10]

# simplify mesh
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(verts.astype(float))
# mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
#
# mesh = mesh.simplify_quadric_decimation(1000)
# verts = np.asarray(mesh.vertices)
# faces = np.asarray(mesh.triangles)

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax = fig.add_subplot(111, projection='3d')
ax.title.set_text('gun')
ax.view_init(elev=0, azim=90)
ax.plot_trisurf(verts[:, 0],
                verts[:, 1],
                faces,
                verts[:, 2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# xlim = [xs[0], xs[-1]]
# ylim = [-ys[-1], ys[-1]]
# zlim = [zs[0], zs[-1]]
xlim = [verts[:, 0].min(), verts[:, 0].max()]
ylim = [verts[:, 1].min(), verts[:, 1].max()]
zlim = [verts[:, 2].min(), verts[:, 2].max()]
# ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)
ax.set_box_aspect([xlim[1] - xlim[0],
                   ylim[1] - ylim[0],
                   zlim[1] - zlim[0]])

# # for debugging by looking at vol directly
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(x.flatten(),
#            y.flatten(),
#            z.flatten(),
#            s=-vol.flatten())
# ax2.set_xlim(xlim)
# ax2.set_ylim(ylim)
# ax2.set_zlim(zlim)
# ax2.set_box_aspect([xlim[1] - xlim[0],
#                    ylim[1] - ylim[0],
#                    zlim[1] - zlim[0]])

plt.show()
