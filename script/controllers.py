from time import time
import numpy as np
import sys
from casadi import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def CEC(curr_state, ref_state, time_step, horizon, gamma, cur_iter, sim_time):

    ref_state = DM(ref_state)
    curr_state = DM(curr_state)
    curr_state = curr_state.T
    
    # Cost parameters
    Q = DM(np.array([[1.4, 0],[0, 2.8]])) * 2.5
    R = DM(np.array([[0.8, 0],[0, 0.01]])) * 10
    q = 7000

    tuning_info = 'gamma_decrease'

    vmax = 1
    wmax = 1
    vmin = 0
    wmin = -1

    size = int(min(horizon, sim_time - cur_iter))# int(horizon - cur_iter)

    lbx = vertcat(repmat([0, -1], size, 1))
    ubx = vertcat(repmat([1, 1], size, 1))

    counter = cur_iter

    controls = MX.sym('x', size, 2)
    controls_vec = controls.reshape((2 * size, 1))

    # constr1 = controls_vec[0::2] <= vmax
    # constr2 = controls_vec[0::2] >= vmin
    # constr3 = controls_vec[1::2] <= wmax
    # constr4 = controls_vec[1::2] >= wmin

    # constraint = vertcat(constr1, constr2, constr3, constr4)

    objective = 0
    objective = DM(objective)

    error = curr_state - ref_state[counter,:]
    
    objective += (error[:2] @ Q @ error[:2].T) + (q * (1 - cos(error[2]))**2)

    while counter < (size + cur_iter - 1):
        
        if (counter - cur_iter) == 0:

            objective += controls_vec[(counter - cur_iter):(counter - cur_iter)+2].T @ R @ controls_vec[(counter - cur_iter):(counter - cur_iter)+2]
            
            new_angle = np.float64(error[2] + ref_state[counter, 2])

            G = np.array([[time_step * np.cos(new_angle), 0],
                          [time_step * np.sin(new_angle), 0], 
                          [0, time_step]])
            G = DM(G)

            p = error + (G @ controls_vec[(counter - cur_iter):(counter - cur_iter)+2]).T 
            + (ref_state[counter, :] - ref_state[counter + 1, :])

        else:

            objective += ((p[:2] @ Q @ p[:2].T + (q * (1 - cos(p[2]))**2)) * (gamma**(counter - cur_iter)))

            objective += ((controls_vec[(counter - cur_iter):(counter - cur_iter)+2].T @ R @ controls_vec[(counter - cur_iter):(counter - cur_iter)+2]) * (gamma**(counter - cur_iter)))
            
            new_angle = p[2] + ref_state[counter, 2]

            G = MX(3,2)
            G[0,0] = time_step * cos(new_angle)
            G[0,1] = 0
            G[1,0] = time_step * sin(new_angle)
            G[1,1] = 0
            G[2,0] = 0
            G[2,1] = time_step
        
            p = p + (G @ controls_vec[(counter - cur_iter):(counter - cur_iter) + 2]).T + (ref_state[counter, :] - ref_state[counter + 1, :])
        
        counter += 1

    nlp = {'x': controls_vec, 'f': objective}#, 'g': constraint}
    solver = nlpsol('solver', 'ipopt', nlp)

    sol = solver(x0 = np.zeros((2 * size, 1)), lbx = lbx, ubx = ubx)

    return np.array(sol['x'][:2]), tuning_info

def GPI(curr_state, ref_traj, cur_iter, time_step, gamma, policy):

    # Implementation of policy iteration algorithm
    # Discretize the state space between -3 and 3

    Q = np.array([[1, 0],[0, 1]]) * 10
    R = np.array([[1, 0],[0, 1]])
    q = 10
    
    ref_state = ref_traj[cur_iter, :]
    next_ref = ref_traj[cur_iter + 1, :]

    window = np.max(abs(curr_state[:2] - ref_state[:2]))
    x = np.linspace(-window, window, 8) + curr_state[0]
    y = np.linspace(-window, window, 8) + curr_state[1]
    theta = np.linspace(-np.pi, np.pi, 10)

    var = np.array([0.04, 0.04, 0.004])
    sigma = np.diag(var)


    ref_disc = np.array([np.digitize(ref_state[0], x) - 1, np.digitize(ref_state[1], y) - 1, np.digitize(ref_state[2], theta) - 1])
    ref_disc[0] = x[int(ref_disc[0])]
    ref_disc[1] = y[int(ref_disc[1])]
    ref_disc[2] = theta[int(ref_disc[2])]

    next_ref_disc = np.array([np.digitize(next_ref[0], x) - 1, np.digitize(next_ref[1], y) - 1, np.digitize(next_ref[2], theta) - 1])
    next_ref_disc[0] = x[int(next_ref_disc[0])]
    next_ref_disc[1] = y[int(next_ref_disc[1])]
    next_ref_disc[2] = theta[int(next_ref_disc[2])]

    cur_disc = np.array([np.digitize(curr_state[0], x) - 1, np.digitize(curr_state[1], y) - 1, np.digitize(curr_state[2], theta) - 1])
    cur_disc[0] = x[int(cur_disc[0])]
    cur_disc[1] = y[int(cur_disc[1])]
    cur_disc[2] = theta[int(cur_disc[2])]

    V = np.zeros((len(x), len(y), len(theta)))

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(theta)):
                state = np.array([x[i], y[j], theta[k]])
                error = state - ref_disc
                V[i, j, k] = (error[:2] @ Q @ error[:2].T) + (q * (1 - cos(error[2]))**2)
                u = policy[i, j, k, :]
                V[i, j, k] += u.T @ R @ u
    
    
    new_policy = np.zeros((len(x), len(y), len(theta), 2))

    v = np.linspace(0, 1, 10)
    w = np.linspace(-1, 1, 10)

    counter = 0

    while True:


        # Implementing policy iteration algorithm

        for l in range(len(x)):
            for m in range(len(y)):
                for n in range(len(theta)):

                    state = np.array([x[l], y[m], theta[n]])
                    error = state - ref_disc

                    cost = (error[:2] @ Q @ error[:2].T) + (q * (1 - cos(error[2]))**2)

                    new_angle = error[2] + ref_state[2]
                    new_angle = theta[int(np.digitize(new_angle, theta) - 1)]
                    G = np.array([[time_step * np.cos(new_angle), 0], 
                                  [time_step * np.sin(new_angle), 0], 
                                  [0, time_step]])

                    V_temp = np.zeros((len(v), len(w)))

                    for i in range(len(v)):
                        for j in range(len(w)):

                            u = np.array([v[i], w[j]])

                            p = error + (G @ u).T + (ref_state - next_ref)

                            V_temp[i, j] = u.T @ R @ u

                            num_samples = 100
                            samples = np.random.multivariate_normal(p, sigma, num_samples)
                            mvn = multivariate_normal(mean = p, cov = sigma)

                            prob = mvn.pdf(samples)
                            prob = prob / np.sum(prob)

                            samples [:, 0] = np.clip(samples[:, 0], min(x), max(x))
                            samples [:, 1] = np.clip(samples[:, 1], min(y), max(y))
                            samples [:, 2] = np.clip(samples[:, 2], min(theta), max(theta))

                            samples[:, 0] = (np.digitize(samples[:, 0], x) - 1)
                            samples[:, 1] = (np.digitize(samples[:, 1], y) - 1)
                            samples[:, 2] = (np.digitize(samples[:, 2], theta) - 1)

                            samples = samples.astype(int)

                            expectation = np.sum(prob * V[samples[:, 0], samples[:, 1], samples[:, 2]])
                            V_temp[i, j] = cost + gamma * expectation

                    ids = np.unravel_index(np.argmin(V_temp, axis=None), V_temp.shape)
                    new_policy[l, m, n, :] = np.array([v[ids[0]], w[ids[1]]])

        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(theta)):
                    state = np.array([x[i], y[j], theta[k]])
                    error = state - ref_disc
                    cost = (error[:2] @ Q @ error[:2].T) + (q * (1 - cos(error[2]))**2)
                    u = new_policy[i, j, k, :]
                    cost += u.T @ R @ u

                    new_angle = error[2] + ref_state[2]
                    new_angle = theta[int(np.digitize(new_angle, theta) - 1)]
                    G = np.array([[time_step * np.cos(new_angle), 0], 
                                  [time_step * np.sin(new_angle), 0], 
                                  [0, time_step]])

                    p = error + (G @ u).T + (ref_state - next_ref)

                    num_samples = 100
                    samples = np.random.multivariate_normal(p, sigma, num_samples)
                    mvn = multivariate_normal(mean = p, cov = sigma)

                    prob = mvn.pdf(samples)
                    prob = prob / np.sum(prob)

                    samples [:, 0] = np.clip(samples[:, 0], min(x), max(x))
                    samples [:, 1] = np.clip(samples[:, 1], min(y), max(y))
                    samples [:, 2] = np.clip(samples[:, 2], min(theta), max(theta))

                    samples[:, 0] = (np.digitize(samples[:, 0], x) - 1)
                    samples[:, 1] = (np.digitize(samples[:, 1], y) - 1)
                    samples[:, 2] = (np.digitize(samples[:, 2], theta) - 1)

                    samples = samples.astype(int)
                    expectation = np.sum(prob * V[samples[:, 0], samples[:, 1], samples[:, 2]])

                    V[i, j, k] = cost + gamma * expectation


        temp = new_policy
        new_policy = policy
        policy = temp
        counter += 1
        print(counter)
        if counter == 10:
            break

    
    control = new_policy[int(np.digitize(curr_state[0], x) - 1), int(np.digitize(curr_state[1], y) - 1), int(np.digitize(curr_state[2], theta) - 1), :]
    return control

    