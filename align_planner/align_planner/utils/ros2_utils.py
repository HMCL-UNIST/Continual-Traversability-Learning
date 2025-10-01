

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import jax.numpy as jnp
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

def _cost_to_color():
    return (0.0, 1.0 - 0.0, 0.0, 0.8)


def _make_linestrip_marker(clock, traj_xy, ns, idx, color):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = clock
    marker.ns = ns
    marker.id = idx
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.points = [Point(x=float(x), y=float(y), z=0.1) for x, y in traj_xy]
    return marker


def publish_all_trajs(states, clock):
    if len(states.shape) != 3:
        return None

    N, T, D = states.shape  # N trajectories, T timesteps, D state dim
    trajs = np.asarray(states[:, :, :2])  # (N, T, 2)

    markers = [
        _make_linestrip_marker(
            clock,
            traj_xy=trajs[i],  # (T, 2)
            ns=f"traj_{i}",
            idx=i,
            color=_cost_to_color()
        )
        for i in range(N)  # loop over number of trajectories
    ]

    return MarkerArray(markers=markers)

    


def get_jax_array_msg(jax_array):
    if jax_array is None or jax_array.size < 1:
        return None

    array = jnp.asarray(jax_array)
    msg = Float64MultiArray()
    msg.data = array.flatten().tolist()

    b, h, d = array.shape
    msg.layout.dim = [
        MultiArrayDimension(label="batch",   size=b, stride=h * d),
        MultiArrayDimension(label="horizon", size=h, stride=d),
        MultiArrayDimension(label="xy",      size=d, stride=1)
    ]

    return msg

    

def get_goal_marker(goal_state, clock):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = clock
    marker.ns = "goal"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = float(goal_state[0])
    marker.pose.position.y = float(goal_state[1])
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0  
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    return marker



    