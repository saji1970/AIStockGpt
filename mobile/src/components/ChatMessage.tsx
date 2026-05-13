import React from 'react';
import {View, Text, StyleSheet, TouchableOpacity, Alert} from 'react-native';
import Markdown from 'react-native-markdown-display';
import StockCard from './StockCard';

interface StockData {
  symbol: string;
  name?: string;
  price: number;
  previousClose?: number;
  change: number;
  changePercent: number;
  high?: number;
  low?: number;
  open?: number;
  volume?: number;
  marketCap?: number;
  currency?: string;
  currencySymbol?: string;
}

interface Props {
  message: string;
  isUser: boolean;
  timestamp?: string;
  stockData?: StockData | null;
  isError?: boolean;
  originalPrompt?: string;
  onRetry?: () => void;
  onCopy?: () => void;
  onCopyToInput?: () => void;
}

export default function ChatMessage({
  message,
  isUser,
  timestamp,
  stockData,
  isError,
  onRetry,
  onCopy,
  onCopyToInput,
}: Props) {
  const time = timestamp ? new Date(timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '';

  const bubbleContent = (
    <>
      {isUser ? (
        <Text style={styles.userText}>{message}</Text>
      ) : isError ? (
        <View>
          <Text style={styles.errorText}>{message}</Text>
          <View style={styles.errorActions}>
            <Text style={styles.retryHint}>Tap to retry</Text>
          </View>
        </View>
      ) : (
        <Markdown style={markdownStyles}>{message}</Markdown>
      )}
      {stockData && <StockCard data={stockData} />}
      {time ? <Text style={styles.timestamp}>{time}</Text> : null}
    </>
  );

  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.aiContainer]}>
      <View style={[styles.avatar, isUser ? styles.userAvatar : isError ? styles.errorAvatar : styles.aiAvatar]}>
        <Text style={styles.avatarText}>{isUser ? 'U' : isError ? '!' : 'AI'}</Text>
      </View>

      {isError ? (
        <TouchableOpacity
          style={[styles.bubble, styles.errorBubble]}
          onPress={onRetry}
          onLongPress={onCopy}
          activeOpacity={0.7}>
          {bubbleContent}
        </TouchableOpacity>
      ) : isUser ? (
        <TouchableOpacity
          style={[styles.bubble, styles.userBubble]}
          onLongPress={() => {
            Alert.alert('Message Options', undefined, [
              {text: 'Copy to Input', onPress: onCopyToInput},
              {text: 'Copy Text', onPress: onCopy},
              {text: 'Cancel', style: 'cancel'},
            ]);
          }}
          activeOpacity={0.9}>
          {bubbleContent}
        </TouchableOpacity>
      ) : (
        <TouchableOpacity
          style={[styles.bubble, styles.aiBubble]}
          onLongPress={onCopy}
          activeOpacity={0.9}>
          {bubbleContent}
        </TouchableOpacity>
      )}
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
  errorAvatar: {backgroundColor: '#ef4444'},
  avatarText: {color: '#fff', fontSize: 12, fontWeight: '700'},
  bubble: {maxWidth: '75%', borderRadius: 16, padding: 12},
  userBubble: {backgroundColor: '#6366f1', borderBottomRightRadius: 4},
  aiBubble: {backgroundColor: '#f3f4f6', borderBottomLeftRadius: 4},
  errorBubble: {backgroundColor: '#fef2f2', borderBottomLeftRadius: 4, borderWidth: 1, borderColor: '#fecaca'},
  userText: {color: '#fff', fontSize: 15, lineHeight: 22},
  errorText: {color: '#dc2626', fontSize: 15, lineHeight: 22},
  errorActions: {flexDirection: 'row', alignItems: 'center', marginTop: 8, justifyContent: 'space-between'},
  retryHint: {color: '#6366f1', fontSize: 12, fontWeight: '700'},
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
