import sys, getopt, os
from bussineslogic.rto_db import init_rto_db

MEMORY_DATABASE = ":memory:"

def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()

if __name__ == '__main__':
    db_name = None
    folder_name = '~'

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hn:f:",["db=","folder="])
    except getopt.GetoptError:
        print('create_database.py -n <db name> [-f] <folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('create_database.py -n <db name> [-f] <folder>')
            sys.exit()
        elif opt in ("-f", "--folder"):
            folder_name = arg
        elif opt in ("-n", "--db"):
            db_name = arg
     
    if(db_name is None):
        print('Please provide a valid database name.')
        sys.exit(2)
    db_path = f'{folder_name}/{db_name}.db'
    print(f'Creating database to {db_path}')
    try:
        touch(db_path)
        init_rto_db(db_path)
    except Exception as e:
      print(f'Error creating database: {e}')
    