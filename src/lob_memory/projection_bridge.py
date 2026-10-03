"""Population-only diagnostics reusing the finite LOB and its Mori conventions.

No trajectory generator, fitted predictor, or random number generator is used.
Finite-state factors extend the covariance recursion with a geometric tail bound.
"""
import numpy as np
from .finite_lob import (event_model, block_population, feature_map,
                        population_covariances, target_kernel, direct_event_mori)
from .mori import from_covariances
from .rg import memory_strength, characteristic_block_lag, characteristic_event_horizon

BASES = ('baseline', 'polynomial', 'complete')
SCALES = (1, 2, 4, 8, 16)
NULL_TOL = 1e-11
IDENTITY_TOL = 1e-10
STATIONARY_TOL = 1e-13
TAIL_TOL = 1e-10
HORIZON_TOL = 1e-7
CHECK_LAGS = 32


def factors(pop, basis, c0):
    """Omega_k = right.T A**k G, including k=0 (instantaneous term).

    C_l = right.T (P.T)**(l-1) left for l>=1 after centering.
    A=P.T-pi*1.T-left*C0^-1*right.T; G=left*C0^-1.
    Removing the stationary mode has no effect on centered covariances.
    This follows by substitution in the existing covariance recursion.
    """
    t = feature_map(pop.cap, basis)
    mu = t @ pop.mean
    weighted = np.einsum('ij,jst->ist', t, pop.weighted)
    left = np.einsum('s,ist->ti', pop.stationary, weighted) - pop.stationary[:, None]*mu
    right = weighted.sum(axis=2).T - mu
    g = np.linalg.solve(c0.T, left.T).T
    a = pop.transition.T - pop.stationary[:, None] - g @ right.T
    return right.T, a, g


def tail_bounds(f, a, g, cutoff, power=32):
    """Upper bounds on sum_{k>L} ||F A^k G|| and sum k||F A^k G||.

    In exact arithmetic, q=||A^m||_2<1 gives geometric block bounds.
    These computed bounds do not include floating-point roundoff.
    """
    q = float(np.linalg.norm(np.linalg.matrix_power(a, power), 2))
    if q >= 1:
        raise ArithmeticError('No contraction certificate at the fixed power')
    left = float(np.linalg.norm(f @ np.linalg.matrix_power(a, cutoff+1), 'fro'))
    z = g.copy(); total = weighted = 0.
    for j in range(power):
        norm = float(np.linalg.norm(z, 2))
        total += norm/(1-q)
        weighted += norm*((cutoff+1+j)/(1-q) + power*q/(1-q)**2)
        z = a @ z
    return left*total, left*weighted, q


def population_case(pop, basis):
    cov = population_covariances(pop, CHECK_LAGS+1, basis)
    c0 = cov[0]
    symmetry = float(np.max(abs(c0-c0.T)))
    eigen_min = float(np.linalg.eigvalsh(c0).min())
    condition = float(np.linalg.cond(c0))
    if symmetry > STATIONARY_TOL or eigen_min <= 0 or condition > 1e12:
        raise ArithmeticError('Invalid or ill-conditioned feature covariance')
    reference = from_covariances(cov)
    f, a, g = factors(pop, basis, c0)
    eig, vec = np.linalg.eigh(c0)
    root = (vec*np.sqrt(eig)) @ vec.T
    target = feature_map(pop.cap, 'baseline')
    t = target if basis == 'complete' else np.eye(3, len(c0))
    ct = t @ c0 @ t.T
    ce, cv = np.linalg.eigh(ct)
    if ce.min() <= 0:
        raise ArithmeticError('Invalid common-target covariance')
    inverse = (cv/np.sqrt(ce)) @ cv.T
    nf = inverse @ t @ f
    ng = g @ root
    cutoff = 64
    while True:
        z = g.copy(); coefficients = []
        for _ in range(cutoff+1):
            coefficients.append(f @ z)
            z = a @ z
        coefficients = np.array(coefficients)
        common = target_kernel(coefficients, c0, pop.cap, basis)
        memory = common[1:]
        strength = memory_strength(memory)
        theta = characteristic_block_lag(memory)
        xi = characteristic_event_horizon(memory, pop.block_size)
        tail, first_tail, q = tail_bounds(nf, a, ng, cutoff)
        null = strength+tail < NULL_TOL
        low = theta*strength/(strength+tail) if strength+tail else 0.
        high = theta+first_tail/strength if strength else 0.
        if tail <= TAIL_TOL and (null or pop.block_size*(high-low) <= HORIZON_TOL):
            break
        cutoff *= 2
        if cutoff > 2048:
            raise ArithmeticError('Tail or horizon bound did not converge')
    factor_error = float(np.max(abs(coefficients[:CHECK_LAGS+1]-reference)))
    normalized_error = float(np.max(abs(common[:CHECK_LAGS+1]-target_kernel(reference,c0,pop.cap,basis))))
    if max(factor_error, normalized_error) > IDENTITY_TOL:
        raise ArithmeticError('Factor and covariance recursion disagree')
    direct_error = 0.
    if pop.block_size == 1:
        direct = direct_event_mori(pop.cap,basis,CHECK_LAGS)
        direct_error = float(np.max(abs(direct-reference)))
        if direct_error > IDENTITY_TOL:
            raise ArithmeticError('Direct event and covariance Mori disagree')
    reference_norms = np.linalg.norm(target_kernel(reference,c0,pop.cap,basis)[1:],axis=(1,2))
    norms = np.linalg.norm(memory,axis=(1,2))
    if basis == 'complete' and max(strength+tail, float(reference_norms.sum())) >= NULL_TOL:
        raise ArithmeticError('Complete-basis null rejected; investigate, do not zero residuals')
    summary = dict(b=pop.block_size,basis=basis,features=len(c0),memory_lags=cutoff,
        memory_strength=strength,tail_bound=tail,weighted_tail_bound=first_tail,
        maximum_lag_norm=float(norms.max()),characteristic_block_lag=theta,
        characteristic_event_horizon=xi,horizon_resolved=not null,
        theta_lower_bound=low,theta_upper_bound=high,
        covariance_min_eigenvalue=eigen_min,covariance_condition=condition,
        covariance_symmetry_error=symmetry,target_covariance_condition=float(np.linalg.cond(ct)),
        contraction_power=32,contraction_norm=q,
        factor_recursion_max_error=factor_error,normalized_factor_recursion_max_error=normalized_error,
        direct_event_checked=pop.block_size==1,direct_event_recursion_max_error=direct_error,
        covariance_recursion_memory_sum_32=float(reference_norms.sum()),
        covariance_recursion_max_lag_norm_32=float(reference_norms.max()))
    if not all(np.isfinite(v) for v in summary.values() if isinstance(v,(int,float))):
        raise ArithmeticError('Nonfinite population metric')
    lags=[dict(b=pop.block_size,basis=basis,lag=k,target_memory_kernel_norm=float(n),
               event_horizon=k*pop.block_size,target_distance=(k+1)*pop.block_size)
          for k,n in enumerate(norms,1)]
    return summary,lags


def population_study(scales=SCALES, cap=8):
    _,_,_,_,p,pi=event_model(cap)
    checks=dict(row_sum_error=float(np.max(abs(p.sum(1)-1))),
        stationary_normalization_error=float(abs(pi.sum()-1)),
        stationarity_error=float(np.max(abs(pi@p-pi))),
        minimum_probability=float(p.min()),minimum_stationary_mass=float(pi.min()))
    if max(checks[k] for k in ('row_sum_error','stationary_normalization_error','stationarity_error'))>STATIONARY_TOL or pi.min()<=0 or p.min()<0:
        raise ArithmeticError('Markov-chain control failed')
    rows=[];lags=[];scale_checks=[]
    for b in scales:
        pop=block_population(cap,b)
        transition_error=float(np.max(abs(pop.transition-np.linalg.matrix_power(p,b))))
        if transition_error > STATIONARY_TOL:
            raise ArithmeticError('Block transition differs from P**b')
        common_cov=population_covariances(pop,0,'baseline')[0]
        for basis in BASES:
            row,detail=population_case(pop,basis)
            c=population_covariances(pop,0,basis)[0]
            t=feature_map(cap,'baseline') if basis=='complete' else np.eye(3,len(c))
            common_error=float(np.max(abs(t@c@t.T-common_cov)))
            if common_error>STATIONARY_TOL:
                raise ArithmeticError('Targets differ between representations')
            row['common_target_covariance_error']=common_error
            rows.append(row);lags.extend(detail)
        scale_checks.append(dict(b=b,block_transition_error=transition_error))
    return rows,lags,dict(**checks,block_checks=scale_checks)
