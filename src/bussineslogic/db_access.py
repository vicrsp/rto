from context.context import Context
from bussineslogic.converter import SqlAlchemyResultToDataFrame

class UnitOfWork():
    def __init__(self):
        self.dbcontext = None
    
    def OpenConnection(self, connection_string):
        self.dbcontext = Context(connection_string)

    def CloseConnection(self):
        self.dbcontext.Dispose()
    
    def GetById(self, table, id):
        session = self.dbcontext.GetSession()
        table = self.dbcontext.metadata.tables['public.{}'.format(table)]
        query = session.query(table)
        return SqlAlchemyResultToDataFrame(query.filter_by(id=id).all(), query)

    def Get(self, table):
        session = self.dbcontext.GetSession()
        table = self.dbcontext.metadata.tables['public.{}'.format(table)]
        query = session.query(table)

        return SqlAlchemyResultToDataFrame(query.all(), query)

    def Query(self, table, join = []):
        session = self.dbcontext.GetSession()
        table = self.dbcontext.metadata.tables['public.{}'.format(table)]
        
        query = session.query(table)
        for relationship in join:
            query = query.join(relationship)

        return query


    

