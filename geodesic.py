import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def action(displacement):
    squares = displacement * displacement
    return jnp.sqrt(squares.sum())

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



def update(points, start, end):
    grad = jax.grad(lagrangian)(points, start, end)
    factor = grad.max()
    return points -  factor * grad



dimension = 300

points = jax.random.normal(jax.random.PRNGKey(42), shape=(100,dimension))
start = jnp.full(dimension, 0.0)
end = jnp.full(dimension, 1.0)

while(True):
    points = update(points, start, end)
    error = (jnp.abs(points[7][1:] - points[7][:-1])).sum()
    print(error)
    if error < 1.0:
        break
