from .. import app
import os
import logging

import openai
from flask import request, jsonify

from app.services.utils import preparePreviousRawContext, getFormattedRawString, rawQA, preprocess_text, found_cache_match, rawPromptDesign

@app.route("/chat", methods=["POST"])
def chat():
    #these all be raw text 
    user_input      = request.json.get("user_input")
    prev_input      = request.json.get("prev_input")
    prev_response   = request.json.get("prev_response")
    prev_prompt     = request.json.get("prev_prompt")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    #prev prompt will contain all the session context + previous query
    #append the prev response and new user_input for new context+ prompt
    # combine this with a new most relevant context search (new system role content)
    new_prev_prompt = old_prev_prompt = prev_prompt
    if prev_prompt :       
        # logging.debug("has previous prompt? and it is raw text string")
        # add the previous response as an answer to the previous question which will be at bottom of the raw text prompt
        # need to do a little text massageing
        # new_prev_prompt += getFormattedMsgString(assistantRole(prev_response))
        old_prev_prompt = preparePreviousRawContext(old_prev_prompt)
        old_prev_prompt += getFormattedRawString(rawQA(prev_input, prev_response), True)

    #Prepocess and tokenize user input
    tokenized_user_input    = preprocess_text(user_input)

    #this finds a match from cached "upvoted" responses, getting their entire context
    found_match             = found_cache_match(tokenized_user_input)
        
    if found_match :
        logging.debug('found cached response match')
        response            = found_match.to_dict()
        context_id          = response["context_id"]
        designed_prompt     = response["literal_prompt"]
        chat_response       = response["literal_response"]
    else:
        logging.debug('hitting ChatGPT api')

        #so there is a usecase for both rawtext and message style... need to learn to finesse
        # logging.debug("using the old style of raw text prompt, messier but better looking results")
        oldPrompt           = rawPromptDesign(user_input, old_prev_prompt)
        old_style_prompt    = oldPrompt["prompt_w_context"]
        context_id          = oldPrompt["context_id"]
        designed_prompt     = old_style_prompt

        # logging.debug('using new style messages array format (not really necessary as it needs to be converted to raw text anyway and it fucking sucks)')
        # newPrompt           = messageStylePromptDesign(user_input, new_prev_prompt)
        # new_style_prompt    = newPrompt["messages"]
        # context_id          = newPrompt["context_id"]
        # designed_prompt     = new_style_prompt
        
        # logging.debug("formatted string : context + new query")
        # logging.debug(designed_prompt)

        #hit the open AI API
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=designed_prompt,
            max_tokens=3000,
            n=1,
            stop=None,
            temperature=0.5,
        )

        chat_response = response.choices[0].text.strip() 
        
    #object for firestore if response is voted on 
    obj_for_firestore = {
         "id" : response["id"]
        ,"created" : response["created"]
        ,"model" : response["model"]
        ,"object" : response["object"]
        ,"usage" : response["usage"]
        ,"tokenized_input" : tokenized_user_input
        ,"context_id" : context_id 
        ,"literal_prompt" : designed_prompt
        ,'literal_response' : chat_response
    }

    # if "rating" in response:
    #     obj_for_firestore["rating"] = response["rating"]

    return jsonify({"response": chat_response, "firestore_data" : obj_for_firestore })
