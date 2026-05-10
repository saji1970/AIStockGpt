import React from 'react';
import {View, Text, StyleSheet} from 'react-native';

interface StockData {
  symbol: string;
  currentPrice: number;
  predictedPrice: number;
  priceChange: number;
  priceChangePercent: number;
  prediction: string; // BUY | SELL | HOLD
  confidence: number;
  metrics?: Record<string, any>;
}

interface Props {
  data: StockData;
}

export default function StockCard({data}: Props) {
  const isPositive = data.priceChange >= 0;
  const predictionColor =
    data.prediction === 'BUY' ? '#10b981' : data.prediction === 'SELL' ? '#ef4444' : '#f59e0b';

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.symbol}>{data.symbol}</Text>
        <View style={[styles.badge, {backgroundColor: predictionColor}]}>
          <Text style={styles.badgeText}>{data.prediction}</Text>
        </View>
      </View>

      <View style={styles.priceRow}>
        <View>
          <Text style={styles.label}>Current</Text>
          <Text style={styles.price}>${data.currentPrice?.toFixed(2)}</Text>
        </View>
        <View>
          <Text style={styles.label}>Predicted</Text>
          <Text style={[styles.price, {color: '#6366f1'}]}>${data.predictedPrice?.toFixed(2)}</Text>
        </View>
      </View>

      <View style={styles.changeRow}>
        <Text style={[styles.change, {color: isPositive ? '#10b981' : '#ef4444'}]}>
          {isPositive ? '\u25B2' : '\u25BC'} ${Math.abs(data.priceChange)?.toFixed(2)} (
          {Math.abs(data.priceChangePercent)?.toFixed(2)}%)
        </Text>
      </View>

      {/* Confidence bar */}
      <View style={styles.confidenceContainer}>
        <Text style={styles.label}>Confidence</Text>
        <View style={styles.barBg}>
          <View style={[styles.barFill, {width: `${Math.min(data.confidence, 100)}%`}]} />
        </View>
        <Text style={styles.confidenceVal}>{data.confidence?.toFixed(0)}%</Text>
      </View>

      {/* Key metrics */}
      {data.metrics && Object.keys(data.metrics).length > 0 && (
        <View style={styles.metrics}>
          {Object.entries(data.metrics).map(([key, val]) => (
            <View key={key} style={styles.metricRow}>
              <Text style={styles.metricKey}>{key}</Text>
              <Text style={styles.metricVal}>{typeof val === 'number' ? val.toFixed(2) : String(val)}</Text>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {backgroundColor: '#fff', borderRadius: 12, padding: 14, marginTop: 10, borderWidth: 1, borderColor: '#e5e7eb'},
  header: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'},
  symbol: {fontSize: 18, fontWeight: '800', color: '#111827'},
  badge: {paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12},
  badgeText: {color: '#fff', fontSize: 12, fontWeight: '700'},
  priceRow: {flexDirection: 'row', justifyContent: 'space-between', marginTop: 10},
  label: {fontSize: 11, color: '#6b7280', marginBottom: 2},
  price: {fontSize: 18, fontWeight: '700', color: '#111827'},
  changeRow: {marginTop: 6},
  change: {fontSize: 13, fontWeight: '600'},
  confidenceContainer: {marginTop: 10},
  barBg: {height: 6, backgroundColor: '#e5e7eb', borderRadius: 3, marginTop: 4, overflow: 'hidden'},
  barFill: {height: 6, backgroundColor: '#6366f1', borderRadius: 3},
  confidenceVal: {fontSize: 12, color: '#6b7280', marginTop: 2, textAlign: 'right'},
  metrics: {marginTop: 10, borderTopWidth: 1, borderTopColor: '#f3f4f6', paddingTop: 8},
  metricRow: {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 3},
  metricKey: {fontSize: 12, color: '#6b7280'},
  metricVal: {fontSize: 12, fontWeight: '600', color: '#374151'},
});
