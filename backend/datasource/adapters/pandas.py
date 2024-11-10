import pandas as pd
import pandasql as psql


class PandasMySQLQueryEngine:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def query(self, sql_query):
        try:
            result = psql.sqldf(sql_query, locals())
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
