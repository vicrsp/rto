import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES |
                               sqlite3.PARSE_COLNAMES)
    except Error as e:
        print(e)
    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def clear_database(conn):
    try:
        sql_rto = '''delete from rto;'''
        sql_run = '''delete from run;'''
        sql_results = '''delete from result_variable_values;'''

        queries = [sql_results, sql_run, sql_rto]
        cur = conn.cursor()
        for query in queries:        
            cur.execute(query)
            conn.commit()

    except Error as e:
      print(f'Error clearing database: {e}')
    


def init_data(conn):

    clear_database(conn)

    sql_rto = '''INSERT INTO rto(id, name, type, model, date) VALUES (0, 'first RTO ever', 'oh', 'yeah',  NULL);'''
    sql_run = '''INSERT INTO run(id, iteration, status, rto_id) VALUES (0, 0, 'none',0);'''

    cur = conn.cursor()
    cur.execute(sql_rto)
    conn.commit()

    cur.execute(sql_run)
    conn.commit()


def create_rto_db(database):
    sql_create_rto_table = """ CREATE TABLE IF NOT EXISTS rto (
                                        id integer PRIMARY KEY,
                                        name text NOT NULL,
                                        type text NOT NULL,
                                        model text NOT NULL,
                                        date timestamp
                                    ); """

    sql_create_run_table = """CREATE TABLE IF NOT EXISTS run (
                                    id integer PRIMARY KEY,
                                    iteration integer NOT NULL,
                                    status text NOT NULL,
                                    rto_id integer NOT NULL REFERENCES rto (id) ON DELETE CASCADE,
                                    FOREIGN KEY (rto_id) REFERENCES rto (id)
                                );"""

    sql_create_run_samples = """CREATE TABLE IF NOT EXISTS sample_values (
                                    run_id integer NOT NULL REFERENCES run (id) ON DELETE CASCADE,
                                    timestamp real NOT NULL,
                                    symbol text NOT NULL,                                    
                                    value real NOT NULL,
                                    enabled integer,
                                    PRIMARY KEY (run_id, timestamp, symbol),
                                    FOREIGN KEY (run_id) REFERENCES run (id)
                                );"""

    sql_create_parameters = """CREATE TABLE IF NOT EXISTS parameter (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL
                                );"""

    sql_create_run_parameters = """CREATE TABLE IF NOT EXISTS parameter_values (
                                    run_id integer NOT NULL REFERENCES run (id) ON DELETE CASCADE,
                                    parameter_name text NOT NULL,
                                    value real,
                                    enabled integer,
                                    PRIMARY KEY (run_id, parameter_name),
                                    FOREIGN KEY (run_id) REFERENCES run (id)
                                );"""

    sql_create_input_data = """CREATE TABLE IF NOT EXISTS input_data (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL
                                );"""

    sql_create_input_data_values = """CREATE TABLE IF NOT EXISTS input_data_values (
                                    var_name integer,
                                    run_id integer REFERENCES run (id) ON DELETE CASCADE,
                                    value real,
                                    PRIMARY KEY (run_id, var_name),
                                    FOREIGN KEY (run_id) REFERENCES run (id)
                                );"""

    sql_create_result_variable = """CREATE TABLE IF NOT EXISTS result_variable (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL
                                );"""

    sql_create_result_values = """CREATE TABLE IF NOT EXISTS result_variable_values (
                                    run_id integer REFERENCES run (id) ON DELETE CASCADE,
                                    var_name text,
                                    value real,
                                    PRIMARY KEY (run_id, var_name),
                                    FOREIGN KEY (run_id) REFERENCES run (id)
                                );"""

    sql_create_simulation_values = """CREATE TABLE IF NOT EXISTS simulation_values (
                                    run_id integer REFERENCES run (id) ON DELETE CASCADE,
                                    var_name text,
                                    sim_type text,
                                    timestamp real,
                                    value real,
                                    PRIMARY KEY (run_id, var_name, timestamp, sim_type),
                                    FOREIGN KEY (run_id) REFERENCES run (id)
                                );"""
    
    sql_create_model_values = """CREATE TABLE IF NOT EXISTS model_values (
                                    run_id integer REFERENCES run (id) ON DELETE CASCADE,
                                    model_name text,
                                    value blob,
                                    PRIMARY KEY (run_id, model_name),
                                    FOREIGN KEY (run_id) REFERENCES run (id)
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        create_table(conn, sql_create_rto_table)
        create_table(conn, sql_create_run_table)
        create_table(conn, sql_create_run_samples)
        create_table(conn, sql_create_parameters)
        create_table(conn, sql_create_run_parameters)
        create_table(conn, sql_create_input_data)
        create_table(conn, sql_create_input_data_values)
        create_table(conn, sql_create_result_variable)
        create_table(conn, sql_create_result_values)
        create_table(conn, sql_create_simulation_values)
        create_table(conn, sql_create_model_values)
        init_data(conn)
    else:
        print("Error! cannot create the database connection.")
