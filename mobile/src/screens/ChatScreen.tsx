import React, {useState, useRef, useCallback, useEffect} from 'react';
import {View, TextInput, TouchableOpacity, FlatList, StyleSheet, Text, Keyboard, Platform, Animated, Alert} from 'react-native';
import Clipboard from '@react-native-clipboard/clipboard';
import ChatMessage from '../components/ChatMessage';
import TypingIndicator from '../components/TypingIndicator';
import QuickPrompts from '../components/QuickPrompts';
import * as api from '../api/client';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
  stockData?: any;
  isError?: boolean;
  originalPrompt?: string;
}

export default function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const keyboardPadding = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const showEvent = Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow';
    const hideEvent = Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide';

    const onShow = (e: any) => {
      Animated.timing(keyboardPadding, {
        toValue: e.endCoordinates.height,
        duration: Platform.OS === 'ios' ? 250 : 100,
        useNativeDriver: false,
      }).start();
    };

    const onHide = () => {
      Animated.timing(keyboardPadding, {
        toValue: 0,
        duration: Platform.OS === 'ios' ? 250 : 100,
        useNativeDriver: false,
      }).start();
    };

    const showSub = Keyboard.addListener(showEvent, onShow);
    const hideSub = Keyboard.addListener(hideEvent, onHide);

    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, [keyboardPadding]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isTyping) return;

      const userMsg: Message = {
        id: Date.now().toString(),
        text: text.trim(),
        isUser: true,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, userMsg]);
      setInput('');
      setIsTyping(true);

      try {
        const response = await api.sendMessage(text.trim());
        const aiMsg: Message = {
          id: (Date.now() + 1).toString(),
          text: response.message || 'No response received.',
          isUser: false,
          timestamp: new Date().toISOString(),
          stockData: response.stockData,
        };
        setMessages(prev => [...prev, aiMsg]);
      } catch (err: any) {
        const errMsg: Message = {
          id: (Date.now() + 1).toString(),
          text: `Error: ${err?.response?.data?.detail || err?.message || 'Failed to get response'}`,
          isUser: false,
          timestamp: new Date().toISOString(),
          isError: true,
          originalPrompt: text.trim(),
        };
        setMessages(prev => [...prev, errMsg]);
      } finally {
        setIsTyping(false);
      }
    },
    [isTyping],
  );

  const handleRetry = useCallback(
    (messageId: string, originalPrompt: string) => {
      // Remove the error message and the original user message before it
      setMessages(prev => {
        const errIdx = prev.findIndex(m => m.id === messageId);
        if (errIdx === -1) return prev;
        // Remove the user message right before the error + the error itself
        const userIdx = errIdx - 1;
        const filtered = prev.filter(
          (_, i) => i !== errIdx && (userIdx < 0 || i !== userIdx),
        );
        return filtered;
      });
      // Re-send the original prompt
      sendMessage(originalPrompt);
    },
    [sendMessage],
  );

  const handleCopy = useCallback((text: string) => {
    Clipboard.setString(text);
    Alert.alert('Copied', 'Message copied to clipboard');
  }, []);

  const handleCopyToInput = useCallback((text: string) => {
    setInput(text);
  }, []);

  const renderItem = ({item}: {item: Message}) => (
    <ChatMessage
      message={item.text}
      isUser={item.isUser}
      timestamp={item.timestamp}
      stockData={item.stockData}
      isError={item.isError}
      originalPrompt={item.originalPrompt}
      onRetry={item.isError ? () => handleRetry(item.id, item.originalPrompt!) : undefined}
      onCopy={() => handleCopy(item.text)}
      onCopyToInput={item.isUser ? () => handleCopyToInput(item.text) : undefined}
    />
  );

  const showQuickPrompts = messages.length === 0 && !isTyping;

  return (
    <View style={styles.flex}>
      {/* Welcome header when empty */}
      {messages.length === 0 && (
        <View style={styles.welcome}>
          <Text style={styles.welcomeTitle}>AI Stock GPT</Text>
          <Text style={styles.welcomeSubtitle}>Ask me about stocks, predictions, and market analysis</Text>
        </View>
      )}

      {/* Quick prompts */}
      {showQuickPrompts && <QuickPrompts onSelect={sendMessage} />}

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messageList}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({animated: true})}
        ListFooterComponent={isTyping ? <TypingIndicator /> : null}
        keyboardShouldPersistTaps="handled"
      />

      {/* Input bar */}
      <Animated.View style={[styles.inputBar, {marginBottom: keyboardPadding}]}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask about a stock..."
          placeholderTextColor="#9ca3af"
          multiline
          maxLength={1000}
          editable={!isTyping}
        />
        <TouchableOpacity
          style={[styles.sendBtn, (!input.trim() || isTyping) && styles.sendBtnDisabled]}
          onPress={() => sendMessage(input)}
          disabled={!input.trim() || isTyping}
          activeOpacity={0.7}>
          <Text style={styles.sendText}>Send</Text>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1, backgroundColor: '#f9fafb'},
  welcome: {alignItems: 'center', paddingTop: 60, paddingBottom: 10},
  welcomeTitle: {fontSize: 26, fontWeight: '800', color: '#6366f1'},
  welcomeSubtitle: {fontSize: 14, color: '#6b7280', marginTop: 6, textAlign: 'center'},
  messageList: {paddingVertical: 8, flexGrow: 1},
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 15,
    maxHeight: 120,
    color: '#111827',
  },
  sendBtn: {
    backgroundColor: '#6366f1',
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 10,
    marginLeft: 8,
    justifyContent: 'center',
  },
  sendBtnDisabled: {backgroundColor: '#c7d2fe'},
  sendText: {color: '#fff', fontWeight: '700', fontSize: 14},
});
