import numpy as np
from scipy.optimize import minimize, LinearConstraint
import pandas as pd
from progressbar import ProgressBar

# EXPERIMENT: Performance of the TL-BAI algorithm 

# The following function depicts an indicator random varible and takes an argument 'expression'
# It evaluates to 1 if the expression is true and 0 otherwise
def indicator(expression):
    if expression:
        return 1
    else:
        return 0

# The following function evaluates the hardness T^\star(\mu) through an optimisation routine
# It takes as input the optimisation variable x, the transfer function B, the number and means of source and target arms
def objective_function(x, B_matrix, num_source_arms, num_target_arms, source_arm_means, target_arm_means ):
    
    #np.random.seed(10)  # to obtain the same set of random numbers in every run
    a_star = np.argmax(target_arm_means)
    
    num = 0
    for a in range(num_target_arms):
      if(a == a_star):
        continue
      else:
        num = num + 1
        diff_vector = np.subtract(B_matrix[a], B_matrix[a_star]) 
        num_exp_for_arm_a = np.dot(diff_vector, source_arm_means)
        num_exp_for_arm_a = num_exp_for_arm_a**2
        den_exp_for_arm_a = 0
        for i in range(num_source_arms):
          den_exp_for_arm_a = den_exp_for_arm_a + (diff_vector[i]**2)/x[i]
        
        exp_for_arm_a = num_exp_for_arm_a / (2 * den_exp_for_arm_a) # psi(mu, w)
        if(num == 1):
            inner_min = exp_for_arm_a
        else:
            inner_min = min(inner_min, exp_for_arm_a)

    return -inner_min

def tl_bai(delta, rho, num_source_arms, num_target_arms, B_matrix, mu):

    # Initialize the counts and rewards for each source arm
    counts = np.zeros(num_source_arms) # we sample each arm once 
    n = 0; # each arm has been sampled once 
    rewards_sum = np.random.normal(mu, 1, num_source_arms) 
    estimated_source_rewards = np.zeros(num_source_arms)
    estimated_target_rewards = np.zeros(num_target_arms)
    #estimated_source_rewards = rewards_sum/counts
    #estimated_target_rewards = np.dot(B_matrix, estimated_source_rewards[:, np.newaxis])

    # calculate the optimal allocations for w(num_source_arms)
    w_accumalated = np.zeros(num_source_arms)
    constraint = LinearConstraint(np.ones(num_source_arms), lb=1, ub=1) # \sum_i w_i = 1
    bounds = [(0, 1) for n in range(num_source_arms)] # each vector of w should have entries in [0,1]
    unif_vector = np.ones(num_source_arms)/num_source_arms # uniform vector

    for i in range(num_source_arms):
        counts[i] += 1
        reward = rewards_sum[i]
        estimated_source_rewards[i] = reward
        estimated_target_rewards = np.dot(B_matrix, estimated_source_rewards[:, np.newaxis])
        res = minimize(
            objective_function,
            x0 = unif_vector,
            args=(B_matrix, num_source_arms, num_target_arms, estimated_source_rewards, estimated_target_rewards),
            constraints=constraint,
            bounds=bounds
        )
        w_accumalated = w_accumalated + res.x 
        n += 1

    # Compute Z_GLRT
    #i = 0
    list_of_lists = [];
    
    for a in range(num_target_arms):
        r_a = B_matrix[a];
        b_list = []
        for b in range(num_target_arms):
            if b != a:
                r_b = B_matrix[b];
                diff = np.array(r_a) - np.array(r_b)
                num = np.sign( np.dot( diff, estimated_source_rewards[:, np.newaxis] ) ) * ( np.dot(diff, estimated_source_rewards[:,np.newaxis]) )**2
                den = 2 * np.dot(np.dot(diff, np.diag(1/counts)), diff[:,np.newaxis])
                val_GLRT = num/den;
                b_list.append(val_GLRT)
                #i = i + 1;
        list_of_lists.append(b_list)

    inner_min_values = map(min, list_of_lists)
    Z_GLRT = max(inner_min_values).item()

    # Compute Threshold
    #Zeta_Thresh = np.log(C*n**(1+rho)/delta);
    Zeta_Thresh = np.log((np.log(n) + 1)/delta);


    #Z_GLRT = 0; # The GLR Test Statistic
    #Zeta_Thresh = 99999; # The Informational Threshold
    flag = 1 # Flag value
    s = 0; # same as s_0 in Lemma 11
    
    while flag == 1:

        if (Z_GLRT >= Zeta_Thresh and min(counts) >= 1):
            best_target_arm = np.argmax(estimated_target_rewards);
            flag = 0;

        else:
                   
            # Select an arm using Sampling rule (Lemma 11)
            f = np.sqrt(n/num_source_arms)

            if min(counts) < f:
                arm = s;
                s = s % num_source_arms + indicator(min(counts) < f);
                if (s == 3):
                    s = 0
            else:
                difference = counts - w_accumalated
                arm = np.argmin(difference)
            
            n = n + 1;

            # Increment the count and reward for the selected arm

            counts[arm] += 1
            reward = np.random.normal(mu[arm], 1)
            rewards_sum[arm] += reward
            estimated_source_rewards[arm] = rewards_sum[arm]/counts[arm];
            estimated_target_rewards = np.dot(B_matrix, estimated_source_rewards[:, np.newaxis]);
            #estimated_target_rewards = np.dot(B_matrix, estimated_source_rewards);

            # calculate the optimal allocations 
            res = minimize(
                objective_function,
                x0 = unif_vector,
                args=(B_matrix, num_source_arms, num_target_arms, estimated_source_rewards, estimated_target_rewards),
                constraints=constraint,
                bounds=bounds
            )
            w_accumalated = w_accumalated + res.x 
          
            # Compute Z_GLRT
            #i = 0
            list_of_lists = [];
            
            for a in range(num_target_arms):
                r_a = B_matrix[a];
                b_list = []
                for b in range(num_target_arms):
                    if b != a:
                        r_b = B_matrix[b];
                        diff = np.array(r_a) - np.array(r_b)
                        num = np.sign( np.dot( diff, estimated_source_rewards[:, np.newaxis] ) ) * ( np.dot(diff, estimated_source_rewards[:,np.newaxis]) )**2
                        den = np.dot(np.dot(diff, np.diag(1/counts)), diff[:,np.newaxis])
                        val_GLRT = num/den;
                        b_list.append(val_GLRT)
                        #i = i + 1;
                list_of_lists.append(b_list)

            inner_min_values = map(min, list_of_lists)
            Z_GLRT = max(inner_min_values).item()
            C = 1e20
            # Compute Threshold
            #Zeta_Thresh = np.log(C*n**(1+rho)/delta);
            Zeta_Thresh = np.log(C * n**(1+rho) /delta);

    return n, best_target_arm

# Main body starts here.
num_source_arms = 3
num_target_arms = 3
B_matrix = np.array([[1,0,-1], [4,12,3], [30,11,8]])
mu = np.array([0.3, 0.4, 0.5]); # source arm means
nu = np.dot(B_matrix, mu) # target arm means
best_target_arm = np.argmax(nu)
rho = 0.01 # needed for this experiment

# Parameters for plotting
delta = np.array([np.exp(-i) for i in range(1,10,1)])
average_stopping_time = np.zeros(len(delta))
std_deviations = np.zeros(len(delta))
error_probability_condition_indicator = np.zeros(len(delta))
num_times = 500 # number of runs of the experiment
pbar = ProgressBar()

for j in pbar(range(len(delta))):
    n2 = np.zeros(num_times)
    error_prob_indicators = np.zeros(num_times)
    
    for k in range(num_times):
        n2[k], best_target_arm2 = tl_bai(delta[j], rho, num_source_arms, num_target_arms, B_matrix, mu)
        error_prob_indicators[k] = best_target_arm != best_target_arm2
        print("Round %d iteration %d" % (j,k))
    
    average_stopping_time[j] = np.mean(n2)
    std_deviations[j] = np.std(n2)
    error_probability_condition_indicator[j] = np.mean(error_prob_indicators) <= delta[j]
    
    print('\n', delta[j], np.mean(n2), np.mean(error_prob_indicators), error_probability_condition_indicator[j])
        
# Output .csv files 
df = pd.DataFrame({'delta': np.log(1/delta),
                   'stopping_time': average_stopping_time,
                   'std_dev': std_deviations * 0.1}, index=None)
df.to_csv('/Users/bharatikamakoti/Downloads/TL-BAI_new1.csv', index=None)

