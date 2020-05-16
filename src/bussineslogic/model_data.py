 

class ModelData:
    def __init__(self, db):
        self.unitOfWork = db

    def CreateNewRun(self, run_id):
        pass
        #return self.unitOfWork.GetById('run', run_id)

    def SaveSimulationResults(self, run_id, simResults):
        pass

