import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def action(left_point, right_point):
    displacement = right_point - left_point
    squares = displacement * displacement
    return squares.sum()

@jax.jit
def lagrangian(
        points,        # n points
        start,         # start point. fixed
        end):          # end point. fixed


    accumulator = action(start, points[0])

    for i in range(0, points.shape[0] - 1):
        accumulator += action(points[i], points[i+1])

    accumulator += action(points[-1], end)

    return accumulator


@jax.jit
def update(points, start, end, factor=0.1):
    return points -  factor * jax.grad(lagrangian)(points, start, end)



def run(initial_points, start, end, num_steps):
    points = initial_points
    lagrangian_vals = []

    for step in range(num_steps):
        points = update(points, start, end)
        val = lagrangian(points, start, end)
        lagrangian_vals.append(val)
        if step % 1000 == 0:
            print("step:      ", step)
            print("lagrangian:", val)

    return points, lagrangian_vals



##################################################################


num_points = 100
dimension = 3000
num_steps = 50000

points = jax.random.normal(jax.random.PRNGKey(42), shape=(num_points,dimension))
start = jnp.full(dimension, 0.0)
end = jnp.full(dimension, 1.0)

result, lagrangian_vals = run(points, start, end, num_steps)
