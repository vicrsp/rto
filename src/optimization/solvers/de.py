import numpy as np


class DifferentialEvolution:
    def __init__(self, lb, ub, mutation_prob=0.5, pop_size=10, max_generations=100, de_type='rand/1/bin', callback=None):
        self.lb = np.asarray(lb).reshape(1, -1)
        self.ub = np.asarray(ub).reshape(1, -1)
        self.population_size = pop_size
        self.max_generations = max_generations
        self.num_variables = len(lb)
        self.mutation_prob = mutation_prob
        self.callback = callback
        self.base, self.d, self.rec = de_type.split('/')

        self.norm_lb = self.normalize(self.lb).flatten()
        self.norm_ub = self.normalize(self.ub).flatten()

        self.reset()

    def reset(self):
        self.fobj_evals = 0
        self.population_fobj = []
        self.best_objective = np.Infinity
        self.best_solution = []

    def normalize(self, x):
        norm_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            norm_x[i, :] = 100 * (x[i] - self.lb) / (self.ub - self.lb)
        return norm_x

    def denormalize(self, x):
        xr = x.reshape(-1, self.num_variables)
        denorm_x = np.zeros_like(xr)
        for i in range(xr.shape[0]):
            denorm_x[i, :] = xr[i] * (self.ub - self.lb) / 100 + self.lb
        return denorm_x

    def initialize_population(self):
        pop_size = (self.population_size, self.num_variables)
        self.population = np.random.uniform(
            low=self.norm_lb, high=self.norm_ub, size=pop_size)
        self.initial_population = self.population

    def evaluate_population_cost(self, population):
        pop_fobj = []
        pop_g = []
        # Calculating the fitness value of each solution in the current population.
        for sol in population:
            cost, g = self.eval_objective(sol)

            if((cost < self.best_objective) & (not np.any(g > 0))):
                self.best_objective = cost
                self.best_solution = sol

            pop_fobj.append(cost)
            pop_g.append(g)

        pop_fobj = np.array(pop_fobj)
        pop_g = np.asarray(pop_g)

        self.fobj_evals = self.fobj_evals + pop_fobj.shape[0]
        self.population_fobj = pop_fobj
        self.population_g = pop_g

        if(self.callback != None):
            self.callback(self.denormalize(population), pop_fobj, pop_g)

        return pop_fobj, pop_g

    def select_base_vector(self, population, cost):
        if(self.base == 'rand'):
            r1 = np.random.randint(0, self.population_size)
            return r1, population[r1]
        elif(self.base == 'mean'):
            return None, np.mean(population, axis=0)
        elif(self.base == 'best'):
            best_idx = np.argmin(cost)
            return None, population[best_idx]
        else:
            raise ValueError('Base={} is not implemented!'.format(self.base))

    def select_difference_vector(self, r1, population):
        if(self.d == "1"):
            r2 = np.random.randint(0, self.population_size)
            if(r1 != None):
                while(r2 == r1):
                    r2 = np.random.randint(0, self.population_size)
            r3 = np.random.randint(0, self.population_size)
            if(r1 != None):
                while(r3 == r1 | r3 == r2):
                    r3 = np.random.randint(0, self.population_size)
            return population[r2] - population[r3]
        else:
            raise ValueError(
                'd={} is not implemented!'.format(self.d))

    def select_scale_factor(self):
        return np.random.rand() * 0.5 + 0.5  # U(0.5, 1.0)

    def mutate(self, target, scale_factor, difference):
        return target + scale_factor * difference

    def recombine(self, v, x):
        if (self.rec == "bin"):
            u_i = []
            for i, v_i in enumerate(v):
                u_j = []
                delta = np.random.randint(0, self.num_variables)
                for j in range(self.num_variables):
                    randnum = np.random.rand()
                    if((randnum <= self.mutation_prob) | (j == delta)):
                        u_j.append(v_i[j])
                    else:
                        u_j.append(x[i, j])
                u_i.append(u_j)
            return np.asarray(u_i)
        else:
            raise ValueError(
                'Recombination={} is not implemented!'.format(self.rec))

    def validate_bounds(self, x):
        xc = []
        for i, value in enumerate(x):
            if((value < self.norm_lb[i]) | (value > self.norm_ub[i])):
                # replace the variable by a random value inside the bounds
                xc.append(np.random.rand() *
                          (self.norm_ub[i] - self.norm_lb[i]) + self.norm_lb[i])
            else:
                xc.append(value)

        return np.asarray(xc)

    def eval_objective(self, x):
        cost, g = self.fobj(self.denormalize(x).flatten())
        return cost, np.asarray(g)

    def select_survivors(self, u, x, fx, gx):
        survivors = []
        for i in range(self.population_size):
            u_i = self.validate_bounds(u[i])
            gx_i = gx[i]
            fx_i = fx[i]
            fu, gu = self.eval_objective(u_i)

            is_valid = (fu <= fx_i)
            # only use the rule for restricted problems
            if(len(gu) > 0):
                rule1 = np.all(gu <= 0) & np.all(gx_i <= 0) & (fu <= fx_i)
                rule2 = np.all(gu <= 0) & np.any(gx_i > 0)
                rule3 = np.any(gu > 0) & np.all(np.maximum(gu, np.zeros_like(
                    gu)) <= np.maximum(gx_i, np.zeros_like(gx_i)))
                is_valid = rule1 | rule2 | rule3

            if(is_valid):
                survivors.append(u_i)
            else:
                survivors.append(x[i])

        return np.asarray(survivors)

    def run(self, func, debug=False):
        self.reset()
        self.fobj = func
        self.initialize_population()
        for i in range(self.max_generations):
            fobj, g = self.evaluate_population_cost(self.population)
            v = []
            # use penalization for base vector selection only
            # fobj_penalized = fobj + 1000 * np.maximum(np.zeros(self.population_size), np.max(np.asarray(g), axis=1))
            for _ in range(self.population_size):
                r1, base = self.select_base_vector(self.population, None)
                difference = self.select_difference_vector(r1, self.population)
                scale_factor = self.select_scale_factor()
                v.append(self.mutate(base, scale_factor, difference))

            v = np.asarray(v)
            u = self.recombine(v, self.population)
            self.population = self.select_survivors(
                u, self.population, fobj, g)

            # if(debug == True):
            #     print('Progress: {:.2f}%'.format(
            #         100 * i / self.max_generations))

            if((debug == True) & (self.best_objective != np.Infinity)):
                print('Best fobj: {}'.format(self.best_objective))
                # print('Best sol: {}'.format(
                #     self.denormalize(self.best_solution)))

        if(self.best_objective != np.Infinity):
            return self.best_objective, self.denormalize(self.best_solution).flatten()
        else:
            return np.Infinity, None
