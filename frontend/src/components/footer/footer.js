import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom"; 
import axios from "axios";

import {v4 as uuid} from "uuid";
import {saveNewSession, addSessionQuery} from "../database/dexie";


import {ChatContext} from "../../contexts/Chat";

import { Container } from 'react-bootstrap';
import { Send, ArrowClockwise, EraserFill } from 'react-bootstrap-icons';

import "./footer.css";

function Footer (){
    const [inputPH, setInputPH] = useState("Ask a question...");
    const [input, setInput]     = useState("");
    const [loading, setLoading] = useState(false);

    const chat_context          = useContext(ChatContext);
    const navigate              = useNavigate();

    const findMostRecentUpVoted = (arr) => {
        for (let i = arr.length - 1; i >= 0; i--) {
            if (Object.hasOwn(arr[i].firestore, "rating") && arr[i].firestore.rating) {
                return arr[i];
            } else {
                continue;
            }
        }

        return null;
    }

    const clearCurrent = () => {
        chat_context.clearMessages();
    }

    const handleSubmit = async () => {
        try {
            if(window.location.pathname !== "/"){
                navigate("/");
            }

            setLoading(true);

            const has_unrated = chat_context.messages.some(item => !Object.hasOwn(item.firestore, 'rating') );
            if(has_unrated){
                chat_context.setShowRatingPO(true);
                setLoading(false);
                return;
            }
            
            const last_qa   = findMostRecentUpVoted(chat_context.messages);
            // const last_qa   = chat_context.messages.length ? chat_context.messages[chat_context.messages.length - 1] : null;
            const post_data = {
                "user_input"    : input,
                "prev_input"    : last_qa ? last_qa.q : undefined,
                "prev_response" : last_qa ? last_qa.a : undefined,
                "prev_prompt"   : last_qa ? last_qa.firestore.literal_prompt : undefined
            };

            //POST USER INPUT TO BACKEND ENDPOINT
            const result = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/chat`, post_data);

            //MAKE ONE UNIT OF A Q&A TO SAVE IN THE SESSION
            const q_a = {"q" : input , "a" : result.data.response, "id" : result.data.firestore_data.id, "firestore" : result.data.firestore_data };
            const for_archive = Object.assign({}, q_a);

            //SAVE TO INDEX DB
            if(!chat_context.sessionId){
                const new_unique_id = uuid();
                const first_timestamp = new Date().getTime();
                chat_context.setSessionId(new_unique_id);
                saveNewSession(new_unique_id, first_timestamp, [for_archive]);
            }else{
                addSessionQuery(chat_context.sessionId, for_archive);
            }
            
            //CURRENT SESSIONS MESSAGES
            chat_context.addMessage(q_a);
            chat_context.setMsgCount(chat_context.messages.length+1);

            //CLEAR INPUT AND LOADING
            setInput("");
            setLoading(false);
            
            // TODO , should i concatanate the User inputs so that they might carry the context throughout the chat?
            // MAYBE ONCE TRAIN CUSTOM MODEL THEN CAN PRESERVE TOKENS
            
        } catch (error) {
            console.error("Error:", error);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            setInput(e.target.value);
            handleSubmit();
        }
    }
        
    return (
                <Container className={`container footer`}>             
                    <button onClick={clearCurrent} className={`clear_chat`}><EraserFill color="#ccc" size={20}/></button>
                    <input className={`user_input`} placeholder={inputPH} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={  handleKeyDown } />
                    <button onClick={handleSubmit}><Send color="#ccc" size={20} className={`send ${loading ? "off" : ""}`}/><ArrowClockwise color="#ccc" size={20} className={`sendfill ${loading ? "rotate" : ""}`}/></button>
                </Container>     
            );
}

export default Footer;