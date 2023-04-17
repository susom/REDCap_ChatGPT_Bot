import os
import json
import re
import numpy as np
import nltk
import openai
import difflib
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

import logging
import logging_setup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

from firestore_utils import get_firestore_client
db = get_firestore_client()


#NEED TO BUILDTHE API CALL IN ORDER TO SET THE timeou
def create_completion_with_timeout(engine, prompt, max_tokens, n, stop, temperature, timeout):
    url = f'https://api.openai.com/v1/engines/{engine}/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}'
    }

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "stop": stop,
        "temperature": temperature,
    }
    

    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)

    if response.status_code != 200:
        logging.debug(f"Error {response.status_code}: {response.text}")
        return None

    return response.json()


#using NLTK , tokenize and normalize user input AND context raw data
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer    = PorterStemmer()
    tokens     = word_tokenize(text)
    tokens     = [token.lower() for token in tokens if token.isalnum()]
    tokens     = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens


#get potential contexts from firestore, currently getting all but will need a way to filter as data set gets largerz
def fetch_context_data_from_firestore():
    # maybe try to do an initial query that doesnt require getting EVERYTHING

    docs = db.collection("contexts").get()

    # Extract document data as list of dictionaries
    context_data = [doc.to_dict() for doc in docs]
    
    return context_data


#use Scikit-learn to find best context match between user input and context data
def find_most_relevant_context(preprocessed_user_input_keywords):
    most_relevant_context   = {}
    similarity_scores       = []
    best_match_index        = None

    # Fetch context data from Firestore
    context_data = fetch_context_data_from_firestore()

    # Preprocess context data
    for i in range(len(context_data)):
        context_data[i]['preprocessed_sections'] = []
        for section in context_data[i]['sections']:
            preprocessed_question = " ".join(preprocess_text(section['question']))
            preprocessed_answer = " ".join(preprocess_text(section['answer']))
            context_data[i]['preprocessed_sections'].append({
                'preprocessed_question': preprocessed_question,
                'preprocessed_answer': preprocessed_answer
            })

    preprocessed_context_data = []
    map_back = {}
    n = 0
    for i in range(len(context_data)):
        for section in context_data[i]['sections']:
            preprocessed_question = " ".join(preprocess_text(section['question']))
            preprocessed_answer = " ".join(preprocess_text(section['answer']))
            preprocessed_context = f"{preprocessed_question}\n{preprocessed_answer}"
            map_back[n] = i
            preprocessed_context_data.append(preprocessed_context)
            n += 1

    
    # Vectorize context data and user input keywords
    vectorizer                = TfidfVectorizer()
    
    if len(context_data) > 0:
        context_data_tfidf        = vectorizer.fit_transform(preprocessed_context_data)
        user_input_keywords_tfidf = vectorizer.transform([" ".join(preprocessed_user_input_keywords)])

        # Calculate similarity scores
        similarity_scores         = cosine_similarity(user_input_keywords_tfidf, context_data_tfidf)

        # Set a similarity threshold
        similarity_threshold      = 0.1 #anything less than this will be ignored, but can be manually fine tuned later.
        max_score                 = np.max(similarity_scores)

        if max_score > similarity_threshold:
            # Find the most relevant context
            best_match_index = np.argmax(similarity_scores)
        else:
            best_match_index = -1

        if 0 <= best_match_index < len(preprocessed_context_data):
            mapped_index = map_back[best_match_index]
            most_relevant_context = context_data[mapped_index]
        else:
            # Handle the case when there is no valid index found
            most_relevant_context = "{}"


    extra_info = {"most_relevant_context" : most_relevant_context, "similarity_scores" : similarity_scores, "best_match_index" : best_match_index}

    return extra_info


#find exact match from cached responses
def found_cache_match(preprocessed_user_input_keywords, similarity_threshold=85):
    found_match = None 
    
    collection_name = "cached_responses"
    property_name_1 = "tokenized_input"
    property_name_2 = "rating"
    array_to_match = preprocessed_user_input_keywords
    second_property_value = 1
    
    # Create the query for the second property
    query = db.collection(collection_name).where(property_name_2, "==", second_property_value)
    
    # Execute the query using this "generator"
    # documents = query.stream()

    documents = list(query.stream())
    """
    Please note that converting the generator to a list will store all the DocumentSnapshot objects in memory. This might not be an issue for small collections, but for large collections, it can consume a significant amount of memory. If you need to process a large number of documents, it's usually more efficient to work with a generator and process the documents one at a time as you iterate through them.
    """

    # Filter the documents based on the full array match
    matching_documents = [doc for doc in documents if doc.to_dict()[property_name_1] == array_to_match]

    # Print the document IDs of the matching documents
    found_match = None 
    if len(documents) > 0:
        for doc in matching_documents:
            #TODO need to think about this for multiple matches which could happen? maybe not actually.. since all future matches will default to the same answer
            #BUT those tokens are matched with a context,  will the context always match the same... yes actually.  wtf
            found_match = doc
            break

        if not found_match:
            logging.debug("no match found lets do fuzzy matches then")
            
            # Calculate similarity scores for each array in search_list
            similarity_scores = {average_similarity(preprocessed_user_input_keywords, doc.to_dict()[property_name_1]): doc for doc in documents}
            
            # Get the highest similarity score and its corresponding array
            max_similarity_score = max(similarity_scores, key=float)
            best_match = similarity_scores[max_similarity_score]

            # since this is pulling from cache, might want to increase the threshold 85 seems to be ok to allow through "big cat" vs "cat" , but not "lion" vs "cat"
            if max_similarity_score > similarity_threshold :
                found_match = best_match
        
    return found_match


# Custom function to calculate the average similarity between two arrays of different lengths
def average_similarity(array1, array2):
    total_similarity = 0
    matches = 0

    for item1 in array1:
        best_similarity = 0
        for item2 in array2:
            similarity = fuzz.ratio(item1, item2)
            if similarity > best_similarity:
                best_similarity = similarity
        total_similarity += best_similarity
        matches += 1

    for item2 in array2:
        best_similarity = 0
        for item1 in array1:
            similarity = fuzz.ratio(item2, item1)
            if similarity > best_similarity:
                best_similarity = similarity
        total_similarity += best_similarity
        matches += 1

    logging.debug("average similarity score for following two lists")
    logging.debug(array1)
    logging.debug(array2)
    logging.debug(total_similarity / matches)
    return total_similarity / matches if matches > 0 else 0


# THERE IS A NEW "STYLE" Prompt DESIGN THAT SOUNDED DOPE BUT STILL NEEDS TO BE CONVERTED TO RAW TEXT SO IT SUCKS
def systemRole(title, summary):
    #if context present, then this should be the first item in the "messages" array  passed to the api as the prompt along with the user input
    content = f"You are an assistant in a medical institution with knowledge about {title} : {summary}"
    system_role = {"role": "system", "content" : content}

    return system_role


def userRole(user_input):
    #finally append the most recent user's raw input as the last item in the array
    user_role = {"role": "user", "content": user_input};

    return user_role


def assistantRole(content):
    #finally append the most recent user's raw input as the last item in the array
    assistant_role = {"role": "assistant", "content": content};

    return assistant_role


def simulatedAssistantRole(sections):
    #if context present and in Q and A format, then simulate an ongoing conversation in the "messages" array between "user" role and "asssitant" roll
    simulated_context = []
    for section in sections:
        simulated_context.append(userRole(section["question"]))
        simulated_context.append(assistantRole(section["answer"]))

    return simulated_context


def getFormattedMsgString(msg):
    return f"{msg['role']}: {msg['content']}\n"


def messageStylePromptDesign(user_input, previous_prompt=""):
    preprocessed_user_input_keywords    = preprocess_text(user_input)
    relevant_context_data               = find_most_relevant_context(preprocessed_user_input_keywords)

    # lets concat the relevant_context_data to the prompt
    context_dict    = relevant_context_data["most_relevant_context"]
    context_id      = ""
    messages        = []

    if context_dict and not previous_prompt: 
        #if ther is context, then use "title" and "description" property to make the single "system" role
        #have at least one to tell HOW chatGPT should answer (ie, in the style of Dr.Suess, or return output in this json structure example {data: {}}
        system_role = systemRole(context_dict["title"], context_dict["summary"])
        messages.append(system_role)

        #if there is context it will be in Q&A format in "sections", create simulated conversation between "user" role and "assistant" role
        context_id = context_dict["id"]
        assistant_role = simulatedAssistantRole(context_dict['sections'])
        messages.extend(assistant_role)

        previous_prompt = ""
        
    #the last item in the array should be the current user_input, that the API should specifically answer
    messages.append(userRole(user_input))

    #fucking me up here chatgpt, even with messaeg format still needs to feed raw text to api prompt
    formatted_string = "".join([getFormattedMsgString(msg) for msg in messages])
  
    new_prompt = previous_prompt + formatted_string

    #if this is not the first query in the session, then keep growing the previous prompt (possibly adding a new "system" role for context)
    #use the user votes (possibly block query UI if they do not VOTE), to exclude "user/assistant" q+a if downvoted 
    return {"messages" : new_prompt, "context_id" : context_id}


#THE OLD STYLE RAWTEXT IS  AT LEAST MORE CONCISE AND SKIPS THE MIDDLES STEP OF CREATEIONG AN ARRAY FIRST actu
def rawQA(q, a):
    return {"question" : q , "answer" : a}


def preparePreviousRawContext(rawText):
    raw_text_prompt = rawText
    lines = raw_text_prompt.split("\n")
    lines = [line for line in lines if not line.startswith("AI:")]
    cleaned_prompt = "\n".join(lines)
    
    #uh what am i doing here?
    # lines = raw_text_prompt.split("\n")
    # cleaned_lines = []

    # for line in lines:
    #     if line.strip() == "":
    #         break
    #     if not line.startswith("AI:"):
    #         cleaned_lines.append(line)

    # cleaned_prompt = "\n".join(cleaned_lines)
    return cleaned_prompt + "\n"


def getFormattedRawString(msg, response_only = False):
    if response_only:
        return f"{msg['answer']}\n\n"
    else: 
        return f"{msg['question']}:\n{msg['answer']}\n\n"


def getFormattedRawContextSTring(content, context_id):
        return f"Context {context_id}:\n{content}\n\n"


def rawPromptDesign(user_input, previous_prompt=None):
    rawPrompt = ""

    if not previous_prompt:
        previous_prompt = ""

    # old way of providing context with raw text context
    # returns context python dict, so do we not need to json.dumps?
    preprocessed_user_input_keywords    = preprocess_text(user_input)
    extra_info                          = find_most_relevant_context(preprocessed_user_input_keywords)    
    relevant_context_data               = json.dumps(extra_info["most_relevant_context"])

    # lets concat th relevant_context_data to the prompt
    context_dict        = json.loads(relevant_context_data)
    formatted_context   = ""
    context_id          = ""

    if 'sections' in context_dict:
        context_id      = context_dict["id"]    

        logging.debug("Found a context, check if its already in the previous prompt befoer appending it to the whole thing")
        logging.debug(f"Context {context_id}:")

        #lets try using the context in paragraph form   
        #only include if not already in the previous_prompt
        #this seems way more concise and economical with the token usage, and gives more control about layering on more context with subsequent questions
        if f"Context {context_id}:" not in previous_prompt:
            logging.debug("only if thre IS a prev prompt and context id not in it, OR prev prompt is none")
            logging.debug(context_id)
            formatted_context += getFormattedRawContextSTring(context_dict["summary"], context_id)

        #lets use the context in chatgpt generated Q&A format
        # for section in context_dict['sections']:
        #     formatted_context += getFormattedRawString(section)
            
    #add the latest user input to the prompt package
    rawPrompt = f"{formatted_context}{user_input}\nAI:"
    new_prompt = previous_prompt + rawPrompt

    #get list of context ids used in this prompt for analysis later
    pattern         = r'Context (.*):'
    context_id_list = re.findall(pattern, new_prompt)

    logging.debug("new prompt")
    logging.debug(new_prompt)

    return {"prompt_w_context" : new_prompt, "context_id" : context_id_list}


#format raw copy and paste wiki text into firestore object in QA format    
def format_raw_text_to_context(raw_text):
    # Define the prompt to extract the relevant information
    prompt = f"""
    Please extract and format the information from the following raw text:
    
    {raw_text}
    
    Output format:
     {{
      "id" : "short_concise_id"
      "title": "The title of the information",
      "summary": "Short concise summary in paragraph form",
      "sections": "A list of sections, each containing a 'question' and an 'answer'"
    }}
    """

    # contexts = fetch_context_data_from_firestore()
    # batch = db.batch()
    # for i, item in enumerate(contexts):
    #     # Set document reference
    #     doc_ref = db.collection("contexts").document(item["id"])
        
    #     # Add data to batch
    #     batch.set(doc_ref, item)
        
    #     # Commit batch if batch size is reached or if it's the last item
    #     if (i + 1) == len(contexts):
    #         batch.commit()
            
    #         # Create a new batch object
    #         batch = db.batch()
        
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=prompt,
    #     max_tokens=2048,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )

    #set max tokens to 3200, reserve about 900 for the prompt
    response = create_completion_with_timeout(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=3200,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=60
    )

    # string representation of json object , but not actually json yet
    formatted_text          = response["choices"][0]["text"].strip() if "choices" in response else ""

    # Remove the escape characters from the string
    formatted_text_clean    = formatted_text.replace("\\n", "").replace("\\", "")

    json_formatted_text     = json_formatted_text = json.loads(formatted_text_clean) if formatted_text_clean else {}

    # store the new context in Firestore
    if formatted_text_clean:
        doc_ref = db.collection('contexts').document(json_formatted_text["id"])
        doc_ref = doc_ref.set(json_formatted_text) 

    return json_formatted_text


#fetch a URL parse it with Beauifulsoup pull raw text content from specific #div_id
def scrape_content(url, div_id):
    response    = requests.get(url)

    soup        = BeautifulSoup(response.text, 'html.parser')

    content     = soup.find('div', {'id': div_id})

    if content:
        return content.get_text(strip=True)
    else:
        return None

