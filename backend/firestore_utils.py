from google.cloud import firestore
import os

def get_firestore_client():
    # TODO just put in .env file?
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_creds.json'
    return firestore.Client()

