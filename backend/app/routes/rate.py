from .. import app
import os
import logging

from flask import request, jsonify
from app.services.firestore_utils import get_firestore_client
db = get_firestore_client()

@app.route("/rate", methods=["POST"])
def rate():
    storage_obj = request.json.get("storage_obj")

    #get ref to doc (existing or new)
    doc_ref = db.collection('cached_responses').document(storage_obj["id"])

    doc_snapshot = doc_ref.get()
    if doc_snapshot.exists:
        # Document exists, so we can update its data
        logging.debug("update just the rating!")
        result = doc_ref.update({"rating" : storage_obj["rating"]})
    else:
        # Document does not exist, so we can create a new document
        result = doc_ref.set(storage_obj) 

    save_result = 0
    if result:
        # Get the document reference and check if the data matches
        doc_data = doc_ref.get().to_dict()
        if doc_data == storage_obj:
            save_result = 1

    #at later date using the doc_id + model + original prompt (all cached here) we can retrieve the original response (even though cacheing that here too)

    return jsonify({"success" : save_result, "id" : storage_obj["id"]})