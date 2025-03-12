import pymongo
import os
import datetime
import pymongo

def connect_to_mongo(database_name='heart_disease_db', collection_name='patient_data'):
    """Connects to a MongoDB database.

    Args:
        database_name (str, optional): Name of the database. Defaults to 'heart_disease_db'.
        collection_name (str, optional): Name of the collection. Defaults to 'predictions'.

    Returns:
        pymongo.collection.Collection: The connected MongoDB collection.
    """

    MONGO_URL = os.environ.get('MONGO_URL') or "mongodb://127.0.0.1:27017/"
    client = pymongo.MongoClient(MONGO_URL)  
    db = client[database_name]
    collection = db[collection_name]
    return collection

def insert_record(collection, record):
    """Inserts a record into the MongoDB collection.

    Args:
        collection (pymongo.collection.Collection): The connected MongoDB collection.
        record (dict): The record to insert (patient data + prediction).
    """
    record['time'] = datetime.datetime.now()
    collection.insert_one(record)
    
def get_predictions(collection):
    """Retrieves all records from the MongoDB collection.

    Args:
        collection (pymongo.collection.Collection): The connected MongoDB collection.

    Returns:
        list: A list of all records in the collection.
    """
    return list(collection.find())