import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

np.random.seed(0)

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here

        # Noise variances
        self.sig_f = 0.15
        self.sig_v = 0.0001
    
        # Gaussian Process priors
        kernel_f = 0.5*Matern(length_scale=0.5 ,nu=2.5)
        self.f  = GaussianProcessRegressor(kernel=kernel_f, random_state=0, n_restarts_optimizer=10)
        m = 1.5
        kernel_v = np.sqrt(2)*Matern(length_scale=0.5 ,nu=2.5) + ConstantKernel(constant_value=m)
        # We use the constant kernel to modify the mean of the Gaussian Process
        self.v = GaussianProcessRegressor(kernel=kernel_v, random_state=0, n_restarts_optimizer=10)

        # Fix the tollerance
        self.k = 1.2
        self.budget = 0.05

        # Add placeholders for the data
        self.x_sample = np.array([]).reshape(-1, domain.shape[0])
        self.f_sample = np.array([]).reshape(-1, domain.shape[0])
        self.v_sample = np.array([]).reshape(-1, domain.shape[0])


    def next_recommendation(self):
        """
        Recommend the next input to sample.
        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.


        if len(self.x_sample) == 0:
            # We start from a random point in the domain (we cannot optimize the function at the beginning)
            #x = domain[:, 0] + (domain[:, 1] - domain[:, 0])*np.random.rand(domain.shape[0])

            x=self.optimize_speed_acquisition_function() #we search for a starting safe points by maximizing speed lower bound

        else: # We can optimize the function otherwise
            x = self.optimize_acquisition_function()

        """
        not_safe = True
        max_iter = 10
        n = 0
        while not_safe and n<max_iter:
            n+=1
            x = self.optimize_acquisition_function()
            v, s = self.v.predict(x, return_std=True)
            lower_bound = v-s
            if lower_bound>=self.k:
                not_safe = False
        """
        return(x)

    def optimize_speed_acquisition_function(self):
        """
        Optimizes the acquisition function.
        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.speed_acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 50 times and pick best solution
        for _ in range(50):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.
        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 50 times and pick best solution
        '''for _ in range(50):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])'''

        beta_v=1
        beta_f=1

        x_values = np.linspace(domain[:, 0], domain[:, 1], num=1000)
        f_values = np.zeros(x_values.shape)

        mean_v, std_v=self.v.predict(x_values, return_std=True)
        mean_f, std_f=self.f.predict(x_values, return_std=True)

        v_lower_bound=mean_v-beta_v*std_v
        f_upper_bound=mean_f+beta_f*std_f
        f_lower_bound=mean_f-beta_f*std_f

        safe_set=x_values[v_lower_bound[0, :]>=self.k] #the safe set corresponds to the values where the lower bound of the speed is higher then 1.2
        maximization_set=safe_set[f_upper_bound[0, :]>=max(f_lower_bound[0, :])] #we constrain our search to the values of x where the upper bound of f is higher then the maximum of the lower bound of f
        
        s = np.intersect1d(safe_set, maximization_set)

        

        '''    # Remove points with a variance that is too small
        s[s] = (np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >max_var)
        s[s] = np.any(u[s, :] - l[s, :] > self.threshold * beta, axis=1)

        if not np.any(s):
                # no need to evaluate any points as expanders in G, exit
                return'''
        
        exploration_set=[]

        count=0

        for x in s:

            x=np.reshape(x, (1, -1))

            exploration_gp=self.v
            v, sv=exploration_gp.predict(x, return_std=True)
            exploration_x_sample=np.vstack((self.x_sample, x))
            exploration_v_sample=np.vstack((self.v_sample, v+beta_v*sv))
            exploration_gp.fit(exploration_x_sample, exploration_v_sample)

            beta_new=1

            mean_v_new, st_v_new=exploration_gp.predict(x_values, return_std=True)
            v_new_lower_bound=mean_v_new-beta_new*st_v_new

            if np.any(v_new_lower_bound>self.k):
                exploration_set.append(x)


        for i, x in enumerate(x_values):
            f_values[i] = self.acquisition_function(x)
        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.
        Parameters
        ----------
        x: np.ndarray
            x in domain of f
        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        # We decided to use UCB but tweak it in order to prefer points that are safe
        #x = x.reshape(1, -1)

        beta = 2
        b = 3
        f, sf = self.f.predict([x], return_std=True)
        v, sv = self.v.predict([x], return_std=True)

        if v[0]-b*sv[0]<self.k:
            alfa = 0
        else:
            alfa = 10
        
        a = f[0] + beta*sf[0] + alfa#*(v[0]-sv[0])
        return(a) 
    
    def speed_acquisition_function(self, x):

        beta=1.5
        v, sv=self.v.predict([x], return_std=True)
        
        return float(v-beta*sv)


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.
        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        # get the posteriors with the new data point

        self.x_sample = np.vstack((self.x_sample, x))
        self.f_sample = np.vstack((self.f_sample, f))
        self.v_sample = np.vstack((self.v_sample, v))

        self.f.fit(self.x_sample, self.f_sample)
        self.v.fit(self.x_sample, self.v_sample)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.
        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here

        # We return the best value out of all of those that we saw taking care to pick a valid sample
        mas = -np.inf
        index = -1
        for i in range(len(self.f_sample)):
            if self.v_sample[i][0] < self.k:
                self.f_sample[i][0] = -np.inf # this way they will (hopefully) not get picked
            if self.f_sample[i][0] > mas:
                mas = self.f_sample[i][0]
                index = i
       
        x = self.x_sample[index]

        return x

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()
    
    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
            domain.shape[0])
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)
    
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()