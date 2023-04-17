import os
import openai
import json
from bs4 import BeautifulSoup
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import logging
import logging_setup

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), 'secrets', '.env')
load_dotenv(dotenv_path) 

from utils import format_raw_text_to_context, preprocess_text, found_cache_match, rawQA, preparePreviousRawContext,  scrape_content, getFormattedMsgString, getFormattedRawString, assistantRole, rawPromptDesign, messageStylePromptDesign

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get("OPENAI_API_KEY")


from firestore_utils import get_firestore_client
db = get_firestore_client()

@app.route("/")
def default():
    return "convert this to cloud run functions?"

    
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
    

@app.route('/scrape', methods=['POST'])
def scrape():
    scrape_url  = request.json.get("url")
    cont_id     = request.json.get("div_id")

    if not scrape_url or not cont_id:
        return jsonify({'error': 'Missing URL or div_id'}), 400

    text = scrape_content(scrape_url, cont_id)

    if text:
        json_output = format_raw_text_to_context(text)
        return jsonify({'text': json_output})
    else:
        return jsonify({'error': f"Could not find div with ID '{div_id}'"}), 404


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



if __name__ == "__main__":
    app.run(host="0.0.0.0")






