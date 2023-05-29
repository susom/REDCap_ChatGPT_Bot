from flask import Flask
import os
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# load the .env , only need it once APP wide
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
dotenv_path = os.path.join(parent_dir, 'secrets', '.env')
load_dotenv(dotenv_path)

# load errors to a logs/backend.log file
from .services import logging_setup

#load the other routes
from .routes import default, chat, prompt, scrape, rate
