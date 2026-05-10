import React from 'react';
import {View, Text, StyleSheet} from 'react-native';

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
  peRatio?: number;
  fiftyTwoWeekHigh?: number;
  fiftyTwoWeekLow?: number;
}

interface Props {
  data: StockData;
}

export default function StockCard({data}: Props) {
  const isPositive = (data.change ?? 0) >= 0;
  const changeColor = isPositive ? '#10b981' : '#ef4444';

  const formatVolume = (vol: number) => {
    if (vol >= 1e9) return `${(vol / 1e9).toFixed(2)}B`;
    if (vol >= 1e6) return `${(vol / 1e6).toFixed(2)}M`;
    if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`;
    return vol.toString();
  };

  const formatMarketCap = (cap: number) => {
    if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
    return `$${cap}`;
  };

  return (
    <View style={styles.card}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.symbol}>{data.symbol}</Text>
          {data.name ? <Text style={styles.name}>{data.name}</Text> : null}
        </View>
        <View style={[styles.changeBadge, {backgroundColor: isPositive ? '#ecfdf5' : '#fef2f2'}]}>
          <Text style={[styles.changeText, {color: changeColor}]}>
            {isPositive ? '\u25B2' : '\u25BC'} {Math.abs(data.changePercent ?? 0).toFixed(2)}%
          </Text>
        </View>
      </View>

      {/* Price */}
      <View style={styles.priceSection}>
        <Text style={styles.price}>${(data.price ?? 0).toFixed(2)}</Text>
        <Text style={[styles.priceChange, {color: changeColor}]}>
          {isPositive ? '+' : ''}{(data.change ?? 0).toFixed(2)}
        </Text>
      </View>

      {/* Details */}
      <View style={styles.details}>
        {data.open ? (
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Open</Text>
            <Text style={styles.detailValue}>${data.open.toFixed(2)}</Text>
          </View>
        ) : null}
        {data.high && data.low ? (
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Day Range</Text>
            <Text style={styles.detailValue}>${data.low.toFixed(2)} - ${data.high.toFixed(2)}</Text>
          </View>
        ) : null}
        {data.previousClose ? (
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Prev Close</Text>
            <Text style={styles.detailValue}>${data.previousClose.toFixed(2)}</Text>
          </View>
        ) : null}
        {data.volume ? (
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Volume</Text>
            <Text style={styles.detailValue}>{formatVolume(data.volume)}</Text>
          </View>
        ) : null}
        {data.marketCap ? (
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Market Cap</Text>
            <Text style={styles.detailValue}>{formatMarketCap(data.marketCap)}</Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {backgroundColor: '#fff', borderRadius: 12, padding: 14, marginTop: 10, borderWidth: 1, borderColor: '#e5e7eb'},
  header: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start'},
  symbol: {fontSize: 18, fontWeight: '800', color: '#111827'},
  name: {fontSize: 12, color: '#6b7280', marginTop: 1},
  changeBadge: {paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12},
  changeText: {fontSize: 13, fontWeight: '700'},
  priceSection: {flexDirection: 'row', alignItems: 'baseline', marginTop: 8, gap: 8},
  price: {fontSize: 24, fontWeight: '800', color: '#111827'},
  priceChange: {fontSize: 14, fontWeight: '600'},
  details: {marginTop: 10, borderTopWidth: 1, borderTopColor: '#f3f4f6', paddingTop: 8},
  detailRow: {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 3},
  detailLabel: {fontSize: 12, color: '#6b7280'},
  detailValue: {fontSize: 12, fontWeight: '600', color: '#374151'},
});
