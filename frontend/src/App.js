import React from "react";
import {BrowserRouter, Routes, Route} from "react-router-dom";

import {ChatContextProvider} from "./contexts/Chat";

import Header from "./components/header/header";
import Footer from "./components/footer/footer"

// import Splash from "./views/Splash";
import Home from "./views/Home";
import History from "./views/History";

import "./App.css";
import "./assets/styles/global.css";

function App() {
  return (
      <BrowserRouter>
        <ChatContextProvider>
          <div className="App">
            <Header/>
            <div className={`main_content`}>
              <Routes>
                  {/* <Route path='/' element={<Splash />} */}
                  <Route path='/' element={<Home />} />
                  <Route path='/history' element={<History />} />
              </Routes>
            </div>
            <Footer/>
          </div>
        </ChatContextProvider>
      </BrowserRouter>
  );
}

export default App;
