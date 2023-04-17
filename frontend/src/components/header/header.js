import React from "react";
import { Link } from "react-router-dom"; 

import { Container } from 'react-bootstrap';
import { Archive, ChatDots } from 'react-bootstrap-icons';

import "./header.css";

function Header (){

    return (
                <Container className={`header`}>
                    <h1>REDCapBot Support <Link to="/history" className={`archive`}><Archive size={20}/></Link> <Link to="/" className={`chat`}><ChatDots size={20}/></Link></h1> 
                </Container>
            );
}

export default Header;