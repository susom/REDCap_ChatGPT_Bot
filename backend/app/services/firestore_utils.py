from google.cloud import firestore

def get_firestore_client():
    return firestore.Client()

