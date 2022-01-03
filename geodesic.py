import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def action(displacement, dt=1):
    squares = displacement * displacement
    return squares.sum() * dt

@jax.jit
def lagrangian(
        points,        # n points
        start,         # start point. fixed
        end):          # end point. fixed


    accumulator = 0.0

    displacement = points[0] - start
    accumulator += action(displacement)

    for i in range(0, points.shape[0] - 1):
        displacement = points[i+1] - points[i]
        accumulator += action(displacement)


    displacement = end - points[-1]
    accumulator += action(displacement)

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

points = jax.random.normal(jax.random.PRNGKey(42), shape=(num_points,dimension))
start = jnp.full(dimension, 0.0)
end = jnp.full(dimension, 1.0)

result, lagrangian_vals = run(points, start, end, 50000)
