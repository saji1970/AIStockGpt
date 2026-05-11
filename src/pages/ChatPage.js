import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, BarChart3, Zap, TrendingUp } from 'lucide-react';
import ChatMessage from '../components/ChatMessage';
import StockCard from '../components/StockCard';
import TypingIndicator from '../components/TypingIndicator';
import { sendMessage } from '../services/api';

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stockData, setStockData] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([
      {
        id: 1,
        type: 'ai',
        content: `# Welcome to AI Stock GPT!

I'm your intelligent stock analysis assistant powered by advanced LSTM neural networks and NLP. I can help you with:

**Stock Analysis**
- Predict stock prices using AI models
- Analyze technical indicators
- Provide market insights and trends

**Portfolio Analysis**
- Track your portfolio performance
- Get predictions for your holdings
- View gain/loss analysis

**Natural Language Queries**
- Ask questions in plain English
- Get detailed explanations
- Request specific analyses

**Market & Investment Advice**
- Top stocks and ETFs by sector
- Investment strategies for any budget
- Hedge funds, mutual funds, and alternatives

**Example Questions:**
- "What's the prediction for AAPL stock?"
- "Analyze the technical indicators for TSLA"
- "What are the top 10 stocks to invest in?"
- "If I have $500, what should I invest in?"

Just type your question below and I'll provide you with intelligent insights!`,
        timestamp: new Date().toISOString()
      }
    ]);
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await sendMessage(inputValue);

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: response.message,
        timestamp: new Date().toISOString(),
        stockData: response.stockData || null,
        charts: response.charts || null
      };

      setMessages(prev => [...prev, aiMessage]);

      if (response.stockData) {
        setStockData(response.stockData);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickPrompts = [
    "Predict AAPL stock price",
    "Analyze TSLA technical indicators",
    "What are the top 10 stocks to invest in?",
    "If I have $500 to invest, what are the best options?"
  ];

  return (
    <div className="flex-1 flex max-w-7xl mx-auto w-full min-h-0">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 min-h-0">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <ChatMessage message={message} />
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Prompts */}
        {messages.length === 1 && (
          <div className="px-6 py-4 border-t border-gray-200">
            <p className="text-sm text-gray-600 mb-3">Try these quick prompts:</p>
            <div className="flex flex-wrap gap-2">
              {quickPrompts.map((prompt, index) => (
                <button
                  key={index}
                  onClick={() => setInputValue(prompt)}
                  className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="border-t border-gray-200 p-6">
          <div className="max-w-4xl mx-auto">
            <div className="flex space-x-4">
              <div className="flex-1">
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me about stock predictions, technical analysis, or your portfolio..."
                  className="input-field min-h-[60px] max-h-[200px]"
                  rows="1"
                  disabled={isLoading}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="send-button flex items-center space-x-2"
              >
                <Send className="w-4 h-4" />
                <span>Send</span>
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="w-80 bg-white border-l border-gray-200 p-6 hidden lg:block overflow-y-auto">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Stock Dashboard</h3>
            {stockData ? (
              <StockCard data={stockData} />
            ) : (
              <div className="text-center py-8 text-gray-500">
                <BarChart3 className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p>Ask about a stock to see analysis</p>
              </div>
            )}
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Features</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <Zap className="w-5 h-5 text-primary-500" />
                <div>
                  <p className="text-sm font-medium text-gray-900">LSTM Predictions</p>
                  <p className="text-xs text-gray-600">Advanced neural networks</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <Bot className="w-5 h-5 text-primary-500" />
                <div>
                  <p className="text-sm font-medium text-gray-900">NLP Processing</p>
                  <p className="text-xs text-gray-600">Natural language understanding</p>
                </div>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <TrendingUp className="w-5 h-5 text-primary-500" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Technical Analysis</p>
                  <p className="text-xs text-gray-600">Comprehensive indicators</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
