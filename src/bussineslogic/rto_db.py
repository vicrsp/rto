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


def create_rto(conn, project):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO projects(name,begin_date,end_date)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, project)
    conn.commit()
    return cur.lastrowid


def create_run(conn, task):
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """

    sql = ''' INSERT INTO tasks(name,priority,status_id,project_id,begin_date,end_date)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()
    return cur.lastrowid


def update_task(conn, task):
    """
    update priority, begin_date, and end date of a task
    :param conn:
    :param task:
    :return: project id
    """
    sql = ''' UPDATE tasks
              SET priority = ? ,
                  begin_date = ? ,
                  end_date = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()


def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks")

    rows = cur.fetchall()

    for row in rows:
        print(row)


def select_task_by_priority(conn, priority):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks WHERE priority=?", (priority,))

    rows = cur.fetchall()

    for row in rows:
        print(row)


def init_rto_db(database):
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
                                    rto_id integer NOT NULL,
                                    FOREIGN KEY (rto_id) REFERENCES rto (id)
                                );"""

    sql_create_run_samples = """CREATE TABLE IF NOT EXISTS sample_values (
                                    run_id integer NOT NULL,
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
                                    run_id integer NOT NULL,
                                    parameter_id integer NOT NULL,
                                    value real,
                                    enabled integer,
                                    PRIMARY KEY (run_id, parameter_id),
                                    FOREIGN KEY (run_id) REFERENCES run (id),
                                    FOREIGN KEY (parameter_id) REFERENCES parameter (id)
                                );"""

    sql_create_input_data = """CREATE TABLE IF NOT EXISTS input_data (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL
                                );"""

    sql_create_input_data_values = """CREATE TABLE IF NOT EXISTS input_data_values (
                                    input_data_id integer,
                                    run_id integer,
                                    value real,
                                    PRIMARY KEY (run_id, input_data_id),
                                    FOREIGN KEY (run_id) REFERENCES run (id),
                                    FOREIGN KEY (input_data_id) REFERENCES input_data (id)
                                );"""

    sql_create_result_variable = """CREATE TABLE IF NOT EXISTS result_variable (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL
                                );"""

    sql_create_result_values = """CREATE TABLE IF NOT EXISTS result_variable_values (
                                    run_id integer,
                                    var_id integer,
                                    value real,
                                    PRIMARY KEY (run_id, var_id),
                                    FOREIGN KEY (run_id) REFERENCES run (id),
                                    FOREIGN KEY (var_id) REFERENCES result_variable (id)
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
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    init_rto_db(r"D:\rto\src\data\rto.db")
