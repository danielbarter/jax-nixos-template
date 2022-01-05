import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

@jax.jit
def zero(point):
    return 0


@jax.jit
def test_saddle(point):
    """
    test function with a saddle point
    """
    min_1 = jnp.array([0.0,-1.0])
    min_2 = jnp.array([0.0,1.0])
    max_1 = jnp.array([-0.8,0.0])
    max_2 = jnp.array([1.2,0.0])
    displacement_min_1 = min_1 - point
    displacement_min_2 = min_2 - point
    displacement_max_1 = max_1 - point
    displacement_max_2 = max_2 - point
    squares_min_1 = displacement_min_1 * displacement_min_1
    squares_min_2 = displacement_min_2 * displacement_min_2
    squares_max_1 = displacement_max_1 * displacement_max_1
    squares_max_2 = displacement_max_2 * displacement_max_2

    return ( jnp.exp( - squares_max_1.sum()) +
             jnp.exp( - squares_max_2.sum()) -
             jnp.exp( - squares_min_1.sum()) -
             jnp.exp( - squares_min_2.sum()))


@partial(jax.jit, static_argnums=[0])
def action(function, left_point, right_point):

    displacement = right_point - left_point
    squares = displacement * displacement
    graph_component = (function(right_point) - function(left_point)) ** 2
    return squares.sum() + graph_component


@partial(jax.jit, static_argnums=[0])
def lagrangian(
        function,      # function defining graph
        points,        # n points
        start,         # start point. fixed
        end            # end point. fixed
):

    accumulator = action(function, start, points[0])

    for i in range(0, points.shape[0] - 1):
        accumulator += action(function, points[i], points[i+1])

    accumulator += action(function, points[-1], end)

    return accumulator


@partial(jax.jit, static_argnums=[0])
def update(function, points, start, end, factor):
    return points -  factor * jax.grad(lagrangian, argnums=1)(function, points, start, end)



def run(function, initial_points, start, end, num_steps, factor=0.1):
    points = initial_points
    lagrangian_vals = []

    for step in range(num_steps):
        points = update(function, points, start, end, factor)
        val = lagrangian(function, points, start, end)
        lagrangian_vals.append(val)
        if step % 1000 == 0:
            print("step:      ", step)
            print("lagrangian:", val)

    return points, lagrangian_vals
