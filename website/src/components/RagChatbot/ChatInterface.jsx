import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import QueryInput from './QueryInput';
import './ChatInterface.css';

const ChatInterface = ({ apiUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  // Initialize conversation
  useEffect(() => {
    const initConversation = async () => {
      try {
        const response = await fetch(`${apiUrl}/chat/conversation/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        const data = await response.json();
        setConversationId(data.conversation_id);
      } catch (error) {
        console.error('Error initializing conversation:', error);
      }
    };

    initConversation();
  }, [apiUrl]);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (query) => {
    if (!query.trim() || isLoading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call the backend API
      const response = await fetch(`${apiUrl}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          conversation_id: conversationId,
          include_citations: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Add bot response to chat
      const botMessage = {
        id: data.response_id,
        type: 'bot',
        content: data.content,
        citations: data.citations || [],
        timestamp: data.timestamp,
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error getting response:', error);
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now(),
        type: 'error',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Book RAG Chatbot</h2>
        <p>Ask questions about the book content</p>
      </div>
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h3>Welcome to the Book RAG Chatbot!</h3>
            <p>Ask me anything about the book content, and I'll find relevant information for you.</p>
          </div>
        ) : (
          messages.map((message) => (
            <Message key={message.id} message={message} />
          ))
        )}
        {isLoading && (
          <div className="loading-message">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-container">
        <QueryInput onSubmit={handleSubmit} disabled={isLoading} />
      </div>
    </div>
  );
};

export default ChatInterface;