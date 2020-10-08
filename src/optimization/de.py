import numpy as np


class DifferentialEvolution:
    def __init__(self, func, ub, lb, mutation_prob=0.5, pop_size=50, max_generations=100, de_type='rand/1/bin'):
        self.fobj = func
        self.lb = lb
        self.ub = ub
        self.population_size = pop_size
        self.max_generations = max_generations
        self.num_variables = len(lb)
        self.mutation_prob = mutation_prob

        self.base, self.d, self.rec = de_type.split('/')

        self.reset()

    def reset(self):
        self.fobj_evals = 0
        self.population_fobj = []
        self.best_objective = np.Infinity
        self.best_solution = []

    def initialize_population(self):
        pop_size = (self.population_size, self.num_variables)
        self.population = np.random.uniform(
            low=self.lb, high=self.ub, size=pop_size)
        self.initial_population = np.copy(self.population)

    def evaluate_population_cost(self, population):
        pop_fobj = []
        pop_g = []
        # Calculating the fitness value of each solution in the current population.
        for sol in population:
            cost, g = self.eval_objective(sol)

            if(cost > self.best_objective):
                self.best_objective = cost
                self.best_solution = sol

            pop_fobj.append(cost)
            pop_g.append(g)

        pop_fobj = np.array(pop_fobj)

        self.fobj_evals = self.fobj_evals + pop_fobj.shape[0]
        self.population_fobj = pop_fobj
        self.population_g = np.asarray(pop_g)

        return pop_fobj

    def select_base_vector(self):
        if(self.base == 'rand'):
            r1 = np.random.randint(0, self.population_size)
            return r1, self.population[r1]
        elif(self.base == 'mean'):
            return np.NaN, np.mean(self.population, axis=0)
        else:
            raise ValueError('Base={} is not implemented!'.format(self.base))

    def select_difference_vector(self, r1):
        if(self.d == "1"):
            r2 = np.random.randint(0, self.population_size)
            while(r2 == r1):
                r2 = np.random.randint(0, self.population_size)
            r3 = np.random.randint(0, self.population_size)
            while(r3 == r1 | r3 == r2):
                r3 = np.random.randint(0, self.population_size)
            return r2, r3, self.population[r2] - self.population[r3]
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
            if((value < self.lb[i]) | (value > self.ub[i])):
                # replace the variable by a random value inside the bounds
                xc.append(np.random.rand() *
                          (self.ub[i] - self.lb[i]) + self.lb[i])
            else:
                xc.append(value)

        return np.asarray(xc)

    def eval_objective(self, x):
        # truncate x
        cost, g = self.fobj(x)

        # handle constraints with penalty method
        g = np.maximum(g, np.zeros_like(g))
        cost = cost + 1000 * np.sum(g)

        return cost, g

    def select_survivors(self, u, x, fx):
        survivors = []
        for i in range(self.population_size):
            u_i = self.validate_bounds(u[i])
            fu = self.eval_objective(u_i)
            # TODO: implement the constraint handling from Lampinen 2002 here
            if(fu <= fx[i]):
                survivors.append(u_i)
            else:
                survivors.append(x[i])

        return np.asarray(survivors)

    def run(self, debug=True):
        self.initialize_population()
        for _ in range(self.max_generations):
            fitness = self.evaluate_population_cost(self.population)
            v = []

            for _ in range(self.population_size):
                r1, base = self.select_base_vector()
                difference = self.select_difference_vector(r1)
                scale_factor = self.select_scale_factor()
                v.append(self.mutate(base, scale_factor, difference))

            v = np.asarray(v)
            u = self.recombine(v, self.population)
            self.population = self.select_survivors(
                u, self.population, fitness)

        return self.best_objective, self.best_solution
