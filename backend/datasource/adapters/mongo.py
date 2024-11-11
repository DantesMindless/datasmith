from .mixins import VerifyImputsMixin

# from pymongo import MongoClient


class MongoDBAdapter(VerifyImputsMixin):
    # def __init__(self, uri, database_name):
    #     self.client = MongoClient(uri)
    #     self.database = self.client[database_name]

    def find(self, collection_name, query):
        collection = self.database[collection_name]
        return collection.find(query)

    def find_one(self, collection_name, query):
        collection = self.database[collection_name]
        return collection.find_one(query)

    def insert_one(self, collection_name, document):
        collection = self.database[collection_name]
        return collection.insert_one(document)

    def insert_many(self, collection_name, documents):
        collection = self.database[collection_name]
        return collection.insert_many(documents)

    def update_one(self, collection_name, query, update):
        collection = self.database[collection_name]
        return collection.update_one(query, update)

    def delete_one(self, collection_name, query):
        collection = self.database[collection_name]
        return collection.delete_one(query)

    def close(self):
        self.client.close()
