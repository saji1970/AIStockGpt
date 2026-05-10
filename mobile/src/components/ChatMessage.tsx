import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import Markdown from 'react-native-markdown-display';
import StockCard from './StockCard';

interface StockData {
  symbol: string;
  currentPrice: number;
  predictedPrice: number;
  priceChange: number;
  priceChangePercent: number;
  prediction: string;
  confidence: number;
  metrics?: Record<string, any>;
}

interface Props {
  message: string;
  isUser: boolean;
  timestamp?: string;
  stockData?: StockData | null;
}

export default function ChatMessage({message, isUser, timestamp, stockData}: Props) {
  const time = timestamp ? new Date(timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '';

  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.aiContainer]}>
      <View style={[styles.avatar, isUser ? styles.userAvatar : styles.aiAvatar]}>
        <Text style={styles.avatarText}>{isUser ? 'U' : 'AI'}</Text>
      </View>
      <View style={[styles.bubble, isUser ? styles.userBubble : styles.aiBubble]}>
        {isUser ? (
          <Text style={styles.userText}>{message}</Text>
        ) : (
          <Markdown style={markdownStyles}>{message}</Markdown>
        )}
        {stockData && <StockCard data={stockData} />}
        {time ? <Text style={styles.timestamp}>{time}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flexDirection: 'row', marginVertical: 4, marginHorizontal: 12, alignItems: 'flex-end'},
  userContainer: {flexDirection: 'row-reverse'},
  aiContainer: {flexDirection: 'row'},
  avatar: {width: 32, height: 32, borderRadius: 16, justifyContent: 'center', alignItems: 'center', marginHorizontal: 6},
  userAvatar: {backgroundColor: '#6366f1'},
  aiAvatar: {backgroundColor: '#10b981'},
  avatarText: {color: '#fff', fontSize: 12, fontWeight: '700'},
  bubble: {maxWidth: '75%', borderRadius: 16, padding: 12},
  userBubble: {backgroundColor: '#6366f1', borderBottomRightRadius: 4},
  aiBubble: {backgroundColor: '#f3f4f6', borderBottomLeftRadius: 4},
  userText: {color: '#fff', fontSize: 15, lineHeight: 22},
  timestamp: {fontSize: 10, color: '#9ca3af', marginTop: 4, alignSelf: 'flex-end'},
});

const markdownStyles = StyleSheet.create({
  body: {color: '#1f2937', fontSize: 15, lineHeight: 22},
  heading1: {fontSize: 20, fontWeight: '700', color: '#111827', marginVertical: 6},
  heading2: {fontSize: 18, fontWeight: '700', color: '#111827', marginVertical: 4},
  heading3: {fontSize: 16, fontWeight: '600', color: '#111827', marginVertical: 4},
  strong: {fontWeight: '700'},
  code_inline: {backgroundColor: '#e5e7eb', paddingHorizontal: 4, borderRadius: 4, fontFamily: 'monospace', fontSize: 13},
  fence: {backgroundColor: '#1f2937', color: '#e5e7eb', padding: 12, borderRadius: 8, fontFamily: 'monospace', fontSize: 13},
  bullet_list: {marginVertical: 4},
  list_item: {marginVertical: 2},
});
