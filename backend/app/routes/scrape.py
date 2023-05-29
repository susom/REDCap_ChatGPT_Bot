from .. import app
import os
import logging

from flask import request, jsonify
from app.services.utils import scrape_content, format_raw_text_to_context

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