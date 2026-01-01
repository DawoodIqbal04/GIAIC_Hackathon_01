import React from 'react';
import ChatInterface from '../components/RagChatbot/ChatInterface';
import './ChatPage.css';

const ChatPage = () => {
  return (
    <div className="chat-page">
      <header className="chat-page-header">
        <h1>Book RAG Chatbot</h1>
        <p>Ask questions about the Physical AI and Humanoid Robotics book content</p>
      </header>
      
      <main className="chat-page-main">
        <ChatInterface apiUrl={process.env.REACT_APP_API_URL || 'http://localhost:8000'} />
      </main>
      
      <footer className="chat-page-footer">
        <p>This chatbot retrieves information from the book content to answer your questions.</p>
      </footer>
    </div>
  );
};

export default ChatPage;