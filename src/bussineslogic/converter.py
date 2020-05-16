import pandas as pd

def SqlAlchemyResultToDataFrame(result, query):
    df = pd.DataFrame(result)
    df.columns = [col['name'] for col in query.column_descriptions]
    return df
