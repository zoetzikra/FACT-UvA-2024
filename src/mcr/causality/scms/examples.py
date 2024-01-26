import jax.nn

from mcr.causality.scms import GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import numpy as np
import numpyro
import numpyro.distributions as dist
from mcr.causality.scms.functions import *
import sys

# # EXAMPLE 1 SCM
#
# sigma_high = torch.tensor(0.5)
# sigma_medium = torch.tensor(0.09)
# sigma_low = torch.tensor(0.05)
#
# SCM_EX1 = BinomialBinarySCM(
#     dag=DirectedAcyclicGraph(
#         adjacency_matrix=np.array([[0, 1, 0],
#                                    [0, 0, 1],
#                                    [0, 0, 0]]),
#         var_names=['vaccinated', 'covid-free', 'symptom-free']
#     ),
#     p_dict={'vaccinated': sigma_high,
#             'symptom-free': sigma_low, 'covid-free': sigma_medium}
# )
#
# costs = np.array([0.5, 0.1])
# y_name = 'covid-free'
# SCM_EX1.set_prediction_target(y_name)

# GENERIC SCMS for experiments

y_name = 'y'

## NOISE DISTRIBUTIONS

unif_dist = numpyro.distributions.Uniform(low=jnp.array(0.0), high=jnp.array(1.0))
normal_dist = numpyro.distributions.Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
normal_dist_small_var = numpyro.distributions.Normal(loc=jnp.array(0.0),
                                                    scale=jnp.array(0.1))

mixing_dist = numpyro.distributions.Categorical(probs=jnp.ones(3)/3)
multinormal_dist = numpyro.distributions.Normal(loc=jnp.array([-4, 0, 4]), scale=jnp.ones([3]))
mog_dist = numpyro.distributions.MixtureSameFamily(mixing_dist, multinormal_dist)


## SCMS

SCM_3_VAR_CAUSAL = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'y']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist, 'x3': normal_dist_small_var, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial},
    fnc_torch_dict={y_name: sigmoidal_binomial_torch},
    sigmoidal=[y_name],
    costs=[1.0, 1.0, 1.0],
    y_name=y_name
)

SCM_3_VAR_NONCAUSAL = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'y', 'x3']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist, 'x3': normal_dist, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial},
    fnc_torch_dict={y_name: sigmoidal_binomial_torch},
    sigmoidal=[y_name],
    costs=[1.0, 1.0, 1.0],
    y_name=y_name
)


fn_2 = lambda x_1, u_2:  -1 + 3 * jax.nn.sigmoid(-2 * x_1[..., 0]) + u_2
fn_2 = StructuralFunction(fn_2, additive=True)

fn_2_torch = lambda x_1, u_2: -1 + 3 * torch.sigmoid(-2 * x_1[..., 0]) + u_2
fn_2_torch = StructuralFunction(fn_2_torch, additive=True)

# assuming x is ordered as (x1, x2)
fn_3 = lambda x, u_3: -0.05 * x[..., 0] + 0.25 * x[..., 1]**2 + u_3
fn_3 = StructuralFunction(fn_3, additive=True)

# assuming the parents are ordered as (x3, y, x4)
fn_5 = lambda x, u_5: x[..., 0] * 0.2 - x[..., 1] - 0.2 * x[..., 2] + u_5
fn_5 = StructuralFunction(fn_5, additive=True)

SCM_5_VAR_NONLINEAR = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'y', 'x4', 'x5']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist_small_var, 'x3': normal_dist, 'x4': normal_dist,
                'x5': normal_dist_small_var, 'y': unif_dist},
    fnc_dict={'x2': fn_2, 'x3': fn_3, 'x5': fn_5, 'y': sigmoidal_binomial},
    fnc_torch_dict={'x2': fn_2_torch, 'x3': fn_3, 'x5': fn_5, 'y': sigmoidal_binomial_torch},
    sigmoidal=['y'],
    costs=[1.0, 1.0, 1.0, 1.0, 1.0],
    y_name='y'
)


# COVID EXAMPLE
def unif_transform(raw_value, observed):
    if observed:
        return numpyro.distributions.Uniform(low=0.0, high=raw_value.item())
    else:
        return numpyro.distributions.Uniform(low=raw_value.item(), high=1.0)

fn_covid_raw = lambda x: jax.nn.sigmoid(-(-3 + x[..., 0] - x[..., 1] - 2.5 * x[..., 2] + 0.2 * x[..., 3]**2))
fn_covid = lambda x, u: jnp.greater_equal(fn_covid_raw(x), u)
fn_covid_transf = lambda x, x_j: unif_transform(fn_covid_raw(x), x_j)
fn_covid = StructuralFunction(fn_covid, raw=fn_covid_raw, transform=fn_covid_transf, binary=True)

fn_appetite_raw = lambda x: jax.nn.sigmoid(- 2 * x[..., 0])
fn_appetite = lambda x, u: jnp.greater_equal(fn_appetite_raw(x), u)
fn_appetite_transf = lambda x, x_j: unif_transform(fn_appetite_raw(x), x_j)
fn_appetite = StructuralFunction(fn_appetite, raw=fn_appetite_raw, transform=fn_appetite_transf, binary=True)

fn_fever_raw = lambda x: jax.nn.sigmoid(+ 6 - 9 * x[..., 0].astype('float32'))
fn_fever = lambda x, u: jnp.greater_equal(fn_fever_raw(x.astype('float32')), u.astype('float32'))
fn_fever_transf = lambda x, x_j: unif_transform(fn_fever_raw(x), x_j)
fn_fever = StructuralFunction(fn_fever, raw=fn_fever_raw, transform=fn_fever_transf, binary=True)

fn_fatigue_raw = lambda x: jax.nn.sigmoid(-1 + x[..., 0]**2 - 2 * x[..., 1])
fn_fatigue = lambda x, u: jnp.greater_equal(fn_fatigue_raw(x), u)
fn_fatigue_transf = lambda x, x_j: unif_transform(fn_fatigue_raw(x), x_j)
fn_fatigue = StructuralFunction(fn_fatigue, raw=fn_fatigue_raw, transform=fn_fatigue_transf, binary=True)

SCM_COVID = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0]]),
        var_names=['pop_density', 'flu_shot', 'covid_shots', 'bmi_diff', 'covid-free', 'appetite_loss',
                   'fever', 'fatigue']
    ),
    noise_dict={'pop_density': dist.Gamma(4, rate=4/3),
                'flu_shot': dist.Bernoulli(probs=0.39),
                'covid_shots': dist.Categorical(probs=np.array([0.24, 0.02, 0.15, 0.59])),
                'bmi-diff': dist.Normal(0, 1),
                'covid-free': unif_dist,
                'flu': unif_dist,
                'appetite_loss': unif_dist,
                'fever': unif_dist,
                'fatigue': unif_dist
                },
    fnc_dict={'covid-free': fn_covid, 'appetite': fn_appetite, 'fever': fn_fever,
              'fatigue': fn_fatigue},
    y_name= 'covid-free',
    sigmoidal=['covid-free', 'appetite', 'fever', 'fatigue'],
    costs=np.ones(7),
    bound_dict={'covid_shots': (0, 3), 'flu_shot': (0, 1), 'pop_density': (0, float('Inf'))}
)

#

# COVID EXAMPLE

fn_skilled_raw = lambda x: jax.nn.sigmoid((-10 + 3*x[..., 0] + 4*x[..., 1]))
fn_skilled = lambda x, u: jnp.greater_equal(fn_skilled_raw(x), u)
fn_skilled_transf = lambda x, x_j: unif_transform(fn_skilled_raw(x), x_j)
fn_skilled = StructuralFunction(fn_skilled, raw=fn_skilled_raw,
                                transform=fn_skilled_transf, binary=True)

fn_nr_commits_raw = lambda x: 10 * x[..., 0] * (1 + 100 * x[..., 1])
fn_nr_commits = lambda x, u: fn_nr_commits_raw(x) + u
fn_nr_commits = StructuralFunction(fn_nr_commits, raw=fn_nr_commits_raw,
                                   additive=True)

fn_nr_languages_raw = lambda x: jax.nn.sigmoid(10 * x[..., 0])
fn_nr_languages = lambda x, u: fn_nr_languages_raw(x) + u
fn_nr_languages = StructuralFunction(fn_nr_languages, raw=fn_nr_languages_raw,
                                     additive=True)

fn_nr_stars_raw = lambda x: 10 * x[..., 0]
fn_nr_stars = lambda x, u: fn_nr_stars_raw(x) + u
fn_nr_stars = StructuralFunction(fn_nr_stars, raw=fn_nr_stars_raw,
                                 additive=True)



SCM_PROGRAMMING = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 1, 1, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]]),
        var_names=['years_experience', 'degree', 'senior-level_skill', 'nr_commits', 'nr_languages', 'nr_stars']
    ),
    noise_dict={'years_experience': dist.GammaPoisson(8, rate=8/3),
                'degree': dist.Categorical(probs=np.array([0.4, 0.2, 0.3, 0.1])),
                'senior-level_skill': unif_dist,
                'nr_commits': dist.GammaPoisson(40, rate=40/4),
                'nr_languages': dist.GammaPoisson(2, rate=2/4),
                'nr_stars': dist.GammaPoisson(5, rate=5/4)
                },
    fnc_dict={'senior-level_skill': fn_skilled, 'nr_commits': fn_nr_commits, 'nr_stars': fn_nr_stars, 'fever': fn_fever,
              'fatigue': fn_fatigue},
    y_name='senior-level_skill',
    sigmoidal=['senior-level_skill'],
    costs=[5.0, 5.0, 0.0001, 0.01, 0.1],
    bound_dict={'years_experience': (0, sys.maxsize), 'degree': (0, 3),
                'nr_commits': (0, sys.maxsize),
                'nr_languages': (0, sys.maxsize),
                'nr_stars': (0, sys.maxsize)}
)


fn_age_raw = lambda x: -35 
fn_age = lambda x, u: fn_age_raw(x) + u
fn_age = StructuralFunction(fn_age, raw=fn_age_raw,
                                   additive=True)

fn_gender_raw = lambda x: 0
fn_gender = lambda x, u: jnp.greater_equal(fn_gender_raw(x), u)
fn_gender_transf = lambda x, x_j: unif_transform(fn_gender_raw(x), x_j)
fn_gender = StructuralFunction(fn_gender, raw=fn_gender_raw,
                                transform=fn_gender_transf, binary=True)

fn_experience_raw = lambda x: -jax.numpy.invert(-1+0.5*x[..., 1]+jax.numpy.invert(1+jax.numpy.exp(-0.1*x[..., 0])))
fn_experience = lambda x, u: -0.5+jax.numpy.invert(1 + jax.numpy.exp(fn_age_raw(x) + u))
fn_experience = StructuralFunction(fn_experience, raw=fn_experience_raw,
                                   additive=True)

fn_loan_raw = lambda x: 1+0.001*(x[...,0]-5)*(5-x[...,0])+x[...,1]
fn_loan = lambda x, u: fn_age_raw(x) + u
fn_loan = StructuralFunction(fn_loan, raw=fn_loan_raw,
                                   additive=True)

fn_duration_raw = lambda x: -1+0.1*x[...,0]+2*x[...,1]+x[...,3]
fn_duration = lambda x, u: fn_age_raw(x) + u
fn_duration = StructuralFunction(fn_duration, raw=fn_duration_raw,
                                   additive=True)

fn_income_raw = lambda x: -4+0.1*(x[...,0]+35)+2*x[...,1]+x[...,1]*x[...,2]
fn_income = lambda x, u: fn_age_raw(x) + u
fn_income = StructuralFunction(fn_income, raw=fn_income_raw,
                                   additive=True)

fn_saving_raw = lambda x: -4+1.5*jnp.where(jnp.greater_equal(x[...,5], 0), 1, 0)
fn_saving = lambda x, u: fn_age_raw(x) + u
fn_saving = StructuralFunction(fn_saving, raw=fn_saving_raw,
                                   additive=True)

key=jax.random.PRNGKey(42)
fn_credit_raw = lambda x: jax.random.bernoulli(key,jax.numpy.invert(1+jax.numpy.exp(-0.3*(-x[...,3]-x[...,4]+x[...,5]+x[...,6]+x[...,5]*x[...,6]))),1) 
fn_credit = lambda x, u: jnp.greater_equal(fn_credit_raw(x), u)
fn_credit_transf = lambda x, x_j: unif_transform(fn_credit_raw(x), x_j)
fn_credit = StructuralFunction(fn_credit, raw=fn_credit,
                                transform=fn_credit_transf, binary=True)

SCM_CREDIT=GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0]]),
        var_names=['age', 'gender', 'experience', 'loan_amount', 'loan_duration', 'income','saving','credit']
    ),
    noise_dict={'age': dist.Bernoulli(probs=0.5),
                'gender': dist.Gamma(10, rate=3.5),
                'experience': dist.Normal(0,0.25),
                'loan_amount': dist.Normal(0,4),
                'loan_duration': dist.Normal(0,9),
                'income': dist.Normal(0,4),
                'saving': dist.Normal(0,25),
                'credit': unif_dist
                },
    fnc_dict={'age': fn_age, 'gender': fn_gender, 'experience': fn_experience, 'loan_amount': fn_loan,
              'loan_duration': fn_duration,'income': fn_income,"saving": fn_saving},
    y_name='credit',
    sigmoidal=['credit'],
    costs=[1,1,1,1,1,1,1],
    bound_dict={'age': (0, 100), 'gender': (0, 1),
                'experience': (0, sys.maxsize),
                'loan_amount': (0, sys.maxsize),
                'loan_duration': (0, sys.maxsize),
                'income': (0, sys.maxsize),
                'saving': (0, sys.maxsize)}
)
#  OVERVIEW

scm_dict = {'3var-noncausal': SCM_3_VAR_NONCAUSAL, '3var-causal': SCM_3_VAR_CAUSAL,
            '5var-nonlinear': SCM_5_VAR_NONLINEAR, '7var-covid': SCM_COVID,
            '5var-skill': SCM_PROGRAMMING, '7var-credit': SCM_CREDIT
            }