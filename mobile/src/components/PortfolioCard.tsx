import React from 'react';
import {View, Text, TouchableOpacity, StyleSheet} from 'react-native';

interface Props {
  name: string;
  description?: string;
  totalValue: number;
  totalGainLoss: number;
  stockCount: number;
  onPress: () => void;
}

export default function PortfolioCard({name, description, totalValue, totalGainLoss, stockCount, onPress}: Props) {
  const isPositive = totalGainLoss >= 0;

  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.row}>
        <Text style={styles.name}>{name}</Text>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{stockCount} stocks</Text>
        </View>
      </View>
      {description ? <Text style={styles.desc}>{description}</Text> : null}
      <View style={styles.row}>
        <View>
          <Text style={styles.label}>Total Value</Text>
          <Text style={styles.value}>${totalValue.toFixed(2)}</Text>
        </View>
        <View style={{alignItems: 'flex-end'}}>
          <Text style={styles.label}>Gain / Loss</Text>
          <Text style={[styles.value, {color: isPositive ? '#10b981' : '#ef4444'}]}>
            {isPositive ? '+' : ''}${totalGainLoss.toFixed(2)}
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {backgroundColor: '#fff', borderRadius: 12, padding: 16, marginVertical: 6, marginHorizontal: 12, borderWidth: 1, borderColor: '#e5e7eb'},
  row: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6},
  name: {fontSize: 16, fontWeight: '700', color: '#111827'},
  desc: {fontSize: 13, color: '#6b7280', marginBottom: 8},
  badge: {backgroundColor: '#eef2ff', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 3},
  badgeText: {color: '#4338ca', fontSize: 11, fontWeight: '600'},
  label: {fontSize: 11, color: '#6b7280'},
  value: {fontSize: 16, fontWeight: '700', color: '#111827'},
});
