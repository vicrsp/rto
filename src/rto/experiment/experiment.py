import multiprocessing


class Experiment:
    def __init__(self, initial_data, rto):
        self.initial_data = initial_data
        self.rto = rto

    def eval(self, repetitions, data_array, exp_name):
        for i in range(repetitions):
            initial_data = data_array[i]
            u_0_feas = initial_data[0][-1]
            self.rto.run(u_0_feas, exp_name)

    def run(self, config):
        # create the list of jobs to be run
        jobs = []
        adaptation = config['adaptation']
        repetitions = config['repetitions']

        for cfg in config:
            neighbors = cfg['neighbors']
            noise = cfg['noise']
            backoff = cfg['backoff']
            name = cfg['name']

            exp_name = f'{adaptation}-{name}-{neighbors}-{noise}-{backoff}'

            p = multiprocessing.Process(target=self.eval, args=(repetitions, self.initial_data))
            p.start()
            jobs.append(p)

        # wait for all to finish
        [job.join() for job in jobs]