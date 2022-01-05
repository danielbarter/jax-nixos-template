from geodesic import *
import matplotlib.pyplot as plt

def straight_line_test():
    num_points = 100
    dimension = 3000
    num_steps = 50000

    points = jax.random.normal(jax.random.PRNGKey(42), shape=(num_points,dimension))
    start = jnp.full(dimension, 0.0)
    end = jnp.full(dimension, 1.0)

    result, lagrangian_vals = run(zero, points, start, end, num_steps)
    print("total error:", result.var(axis=1).sum())
    fig, ax = plt.subplots()
    ax.set_title("straight line test: lagrangian vals")
    ax.set_ylim([20.0, 100.0])
    ax.plot(lagrangian_vals)
    fig.savefig("/tmp/straight_line_test.pdf")


def contor_map(x_vals, y_vals):
    z_vals = np.zeros((x_vals.shape[0], y_vals.shape[0]))
    for i in range(x_vals.shape[0]):
        for j in range(y_vals.shape[0]):
            z_vals[j,i] = (test_saddle(np.array([x_vals[i],y_vals[j]])))
    return z_vals

def saddle_test_2d():
    num_points = 200
    dimension = 2
    num_steps = 10000

    points = jax.random.normal(jax.random.PRNGKey(42), shape=(num_points,dimension))
    start = jnp.array([0.5,-1.2])
    end = jnp.array([0.5, 1.2])

    result, lagrangian_vals = run(test_saddle, points, start, end, num_steps)
    fig, axs = plt.subplots(3, 1, figsize=(5,15), gridspec_kw={'height_ratios':[1,1, 1]})
    axs[2].set_ylim([0.0,3.0])
    axs[2].plot(lagrangian_vals)
    axs[2].set_title("lagrangian values")
    x_vals = np.arange(-2.0,2.0,0.01)
    y_vals = np.arange(-2.0,2.0,0.01)
    z_vals = contor_map(x_vals, y_vals)


    axs[0].set_title("contor plot")
    axs[0].contour(x_vals, y_vals, z_vals)
    axs[0].scatter(result[:,0], result[:,1])


    grad_norms = []
    for i in range(result.shape[0]):
        grad_norms.append(jnp.square(jnp.square(jax.grad(test_saddle)(result[i])).sum()))

    axs[1].set_title("gradient magnitude")
    axs[1].plot(grad_norms)

    fig.savefig("/tmp/2d_saddle_test.pdf")


# straight_line_test()
saddle_test_2d()
