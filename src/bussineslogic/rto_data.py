from .rto_db import create_connection
import datetime


class RTODataModel:
    def __init__(self, file=r"D:\rto\src\data\rto.db"):
        self.conn = create_connection(file)

    def create_rto(self, name='rto', rto_type='two-step', model='semi-batch'):
        last_rto_id = self.get_last_rto_id()
        current_id = last_rto_id + 1

        sql = ''' INSERT INTO rto(id,name,type,model,date)
              VALUES(?,?,?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, (current_id, name, rto_type,
                          model, datetime.datetime.now()))
        self.conn.commit()

        return current_id

    def create_run(self, rto_id, run_type, iteration, status='completed'):
        last_id = self.get_last_run_id()
        current_id = last_id + 1

        sql = ''' INSERT INTO run(id,rto_id,iteration,status,type)
              VALUES(?,?,?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, (current_id, rto_id, iteration,
                          status, run_type))
        self.conn.commit()

        return current_id

    def save_samples(self, run_id, samples):
        cur = self.conn.cursor()
        run_samples = []
        for time, sample in samples.items():
            for i in range(len(sample)):
                run_samples.append(
                    (run_id, time, 'sample_{}'.format(i), sample[i], 1))

        cur.executemany(
            'INSERT INTO sample_values VALUES (?,?,?,?,?)', run_samples)
        self.conn.commit()

    def save_parameters(self, run_id, parameters):
        cur = self.conn.cursor()
        run_params = []
        for p_id, value in parameters.items():
            run_params.append((run_id, p_id, value, 1))

        cur.executemany(
            'INSERT INTO parameter_values VALUES (?,?,?,?)', run_params)
        self.conn.commit()

    def save_input_data(self, run_id, input_data):
        cur = self.conn.cursor()
        run_params = []
        for inp_id, value in input_data.items():
            run_params.append((run_id, inp_id, value))

        cur.executemany(
            'INSERT INTO input_data_values VALUES (?,?,?)', run_params)
        self.conn.commit()

    def save_results(self, run_id, result_data):
        cur = self.conn.cursor()
        run_params = []
        for rdv_id, value in result_data.items():
            run_params.append((run_id, rdv_id, value))

        cur.executemany(
            'INSERT INTO result_variable_values VALUES (?,?,?)', run_params)
        self.conn.commit()

    def save_simulation_results(self, run_id, sim_values):
        cur = self.conn.cursor()
        run_values = []
        for rdv_id, value in sim_values.items():
            run_values.append((run_id, rdv_id, value[0], value[1]))

        cur.executemany(
            'INSERT INTO simulation_values VALUES (?,?,?,?)', run_values)
        self.conn.commit()

    def get_last_rto_id(self):
        cur = self.conn.cursor()
        cur.execute('SELECT MAX(id) from rto')
        result = cur.fetchone()
        return result[0]

    def get_last_run_id(self):
        cur = self.conn.cursor()
        cur.execute('SELECT MAX(id) from run')
        result = cur.fetchone()
        return result[0]
