import os
os.environ['JAX_ENABLE_X64'] = 'True'
from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np
import jax
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

def generate_data(key, params_tuple, x_range, num_points, noise_std):
    m, q = params_tuple
    x = jnp.linspace(x_range[0], x_range[1], num_points)
    y = linear_model(x, m, q) + random.normal(key, shape=x.shape) * noise_std
    return x, y

def log_normal(x, mean, std):
    return tfpd.Normal(loc=mean, scale=std).log_prob(x)

def main_body():
    return 0

if __name__ == "__main__":
    print(f"JAX is using: {jax.devices()}")
    key = random.PRNGKey(0)
    true_params = jnp.array([2.0, 1.0])  # m, q
    x_range = jnp.array([0,10.0])
    noise_sigma = 2.0
    x_obs, y_obs = generate_data(key, true_params, x_range, num_points=50, noise_std=noise_sigma)
    
    plt.figure('linear_generated')
    plt.title('Generated Data for Linear Model')
    plt.plot(x_obs, y_obs, marker='o', label='Observed Data', markersize=5.0, linestyle='None')
    plt.plot(x_obs, linear_model(x_obs, *true_params), label='True Model', color='black', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.savefig('linear_generated.png', dpi=600)
    #plt.show()

    def log_likelihood(m, q, error_std):
        return jnp.sum(tfpd.Normal(linear_model(x_obs, m, q), error_std).log_prob(y_obs))
    
    def prior_model():
        m = yield Prior(tfpd.Uniform(-100.0,100.0),name='m')
        q = yield Prior(tfpd.Uniform(-100.0,100.0),name='q')
        error_std = yield Prior(tfpd.Uniform(0,25.0),name='error_std')
        return m,q,error_std
    
    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    model.sanity_check(random.PRNGKey(1), S=10)

    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*1000)
    termination_reason, state = jax.jit(ns)(random.PRNGKey(2))
    results = ns.to_results(termination_reason, state=state)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results, save_name='./jaxns_tests/linear_corner.png', kde_overlay=True)

    exit()