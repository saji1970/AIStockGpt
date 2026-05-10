import React from 'react';
import {ScrollView, TouchableOpacity, Text, StyleSheet} from 'react-native';

const PROMPTS = [
  'Predict AAPL stock price',
  'Analyze TSLA technical indicators',
  'Show MSFT sensitivity analysis',
  'What affects stock prices most?',
];

interface Props {
  onSelect: (prompt: string) => void;
}

export default function QuickPrompts({onSelect}: Props) {
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.container}>
      {PROMPTS.map(prompt => (
        <TouchableOpacity key={prompt} style={styles.chip} onPress={() => onSelect(prompt)} activeOpacity={0.7}>
          <Text style={styles.chipText}>{prompt}</Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {paddingHorizontal: 12, paddingVertical: 8, gap: 8},
  chip: {
    backgroundColor: '#eef2ff',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#c7d2fe',
    marginRight: 8,
  },
  chipText: {color: '#4338ca', fontSize: 13, fontWeight: '500'},
});
