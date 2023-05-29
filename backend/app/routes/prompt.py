from .. import app
import os
import logging

from flask import Response

from app.services.firestore_utils import get_firestore_client
from app.services.utils import rawPromptDesign, messageStylePromptDesign

@app.route("/prompt")
def prompt():
    user_input          = "What is r2p2?"
    use_new             = False 

    messages            = messageStylePromptDesign(user_input)
    new_style_prompt    = messages["messages"]

    rawPrompt           = rawPromptDesign(user_input)
    old_style_prompt    = rawPrompt["prompt_w_context"]

    prompt_design       = new_style_prompt if use_new else old_style_prompt

    return Response(prompt_design, content_type='application/json')