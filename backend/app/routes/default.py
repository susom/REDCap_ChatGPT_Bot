from .. import app
import os
import logging


@app.route("/")
def default():
    port = int(os.environ.get("PORT", 5001))
    return f"convert this to cloud run functions? port : {port}"