import os
os.environ['JAX_ENABLE_X64'] = 'True'
import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp=jax.numpy
jsp=jax.scipy
random = jax.random
from jax import grad, jit, vmap
from jaxns import Prior, Model, NestedSampler, bruteforce_evidence
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions


@jit
def linear_model(x, m, q):
    return m * x + q

def generate_data(key, params_tuple, x_range, num_points, noise_std_x, noise_std_y):
    m, q = params_tuple
    x = jnp.linspace(x_range[0], x_range[1], num_points)
    y = linear_model(x, m, q) + random.normal(key, shape=x.shape) * noise_std_y
    x = x + random.normal(key, shape=x.shape) * noise_std_x
    return x, y

def log_normal(x, mean, std):
    return tfpd.Normal(loc=mean, scale=std).log_prob(x)

if __name__ == "__main__":
    print(f"JAX is using: {jax.devices()}")
    key = random.PRNGKey(0)
    true_params = jnp.array([2.0, 1.0])  # m, q
    x_range = jnp.array([0,10.0])
    noise_sigma_x = 2.0
    noise_sigma_y = 1.0
    x_obs, y_obs = generate_data(key, true_params, x_range, num_points=50, noise_std_x=noise_sigma_x, noise_std_y=noise_sigma_y)
    
    plt.figure('linear_generated')
    plt.title('Generated Data for Linear Model')
    plt.plot(x_obs, y_obs, marker='o', label='Observed Data', markersize=5.0, linestyle='None')
    plt.plot(x_obs, linear_model(x_obs, *true_params), label='True Model', color='black', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.savefig('linear_generated_x_uncert.png', dpi=600)
    plt.close()
    #plt.show()

    def log_likelihood(m, q, error_std):
        return jnp.sum(tfpd.Normal(linear_model(x_obs, m, q), error_std).log_prob(y_obs))
    
    def log_likelihood_with_x_uncert(m, q, epsilon_y, epsilon_x):
        return jnp.sum(tfpd.Normal(linear_model(x_obs, m, q), jnp.sqrt(epsilon_y**2 + (m * epsilon_x)**2)).log_prob(y_obs))
    
    def prior_model():
        m = yield Prior(tfpd.Uniform(-1.0,5.0),name='m')
        q = yield Prior(tfpd.Uniform(-1.0,5.0),name='q')
        epsilon_y = yield Prior(tfpd.Uniform(1.0e-14,3.0),name=r'$\sigma_y$')
        epsilon_x = yield Prior(tfpd.Uniform(1.0e-14,3.0),name=r'$\sigma_x$')
        return m,q,epsilon_y, epsilon_x
    
    model = Model(prior_model=prior_model, log_likelihood=log_likelihood_with_x_uncert)
    model.sanity_check(random.PRNGKey(1), S=10)

    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*2000)
    termination_reason, state = jax.jit(ns)(random.PRNGKey(2))
    results = ns.to_results(termination_reason, state=state)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results, save_name='./jaxns_tests/linear_corner_x_uncert.png', kde_overlay=True)

    exit()