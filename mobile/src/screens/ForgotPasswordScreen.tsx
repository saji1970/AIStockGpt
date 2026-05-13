import React, {useState} from 'react';
import {View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView} from 'react-native';
import * as api from '../api/client';

export default function ForgotPasswordScreen({navigation}: any) {
  const [email, setEmail] = useState('');
  const [token, setToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState<'email' | 'reset'>('email');

  const handleSendLink = async () => {
    if (!email.trim()) {
      Alert.alert('Error', 'Please enter your email address');
      return;
    }
    setLoading(true);
    try {
      await api.forgotPassword(email);
    } catch {
      // Always show success to avoid leaking which emails are registered
    } finally {
      setLoading(false);
      setStep('reset');
      Alert.alert('Check Your Email', 'If an account exists for that email, a reset code has been sent.');
    }
  };

  const handleReset = async () => {
    if (!token.trim()) {
      Alert.alert('Error', 'Please enter the reset token from your email');
      return;
    }
    if (!newPassword.trim() || !confirmPassword.trim()) {
      Alert.alert('Error', 'Please enter and confirm your new password');
      return;
    }
    if (newPassword !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }
    setLoading(true);
    try {
      await api.resetPasswordWithToken(token, newPassword);
      Alert.alert('Success', 'Your password has been reset successfully!', [
        {text: 'Sign In', onPress: () => navigation.goBack()},
      ]);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Failed to reset password. The token may have expired.';
      Alert.alert('Error', typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.logoContainer}>
          <Text style={styles.logo}>AI Stock GPT</Text>
          <Text style={styles.subtitle}>Reset Your Password</Text>
        </View>

        <View style={styles.card}>
          {step === 'email' ? (
            <>
              <Text style={styles.title}>Forgot Password</Text>
              <Text style={styles.desc}>
                Enter the email address associated with your account and we'll send you a reset code.
              </Text>
              <TextInput
                style={styles.input}
                placeholder="Email"
                placeholderTextColor="#9ca3af"
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
              />
              <TouchableOpacity style={styles.button} onPress={handleSendLink} disabled={loading} activeOpacity={0.8}>
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Send Reset Code</Text>
                )}
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.title}>Set New Password</Text>
              <Text style={styles.desc}>
                Enter the reset token from your email and choose a new password.
              </Text>
              <TextInput
                style={styles.input}
                placeholder="Reset Token"
                placeholderTextColor="#9ca3af"
                value={token}
                onChangeText={setToken}
                autoCapitalize="none"
              />
              <TextInput
                style={styles.input}
                placeholder="New Password"
                placeholderTextColor="#9ca3af"
                value={newPassword}
                onChangeText={setNewPassword}
                secureTextEntry
              />
              <TextInput
                style={styles.input}
                placeholder="Confirm New Password"
                placeholderTextColor="#9ca3af"
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry
              />
              <TouchableOpacity style={styles.button} onPress={handleReset} disabled={loading} activeOpacity={0.8}>
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Reset Password</Text>
                )}
              </TouchableOpacity>
              <TouchableOpacity onPress={() => setStep('email')} style={styles.toggle}>
                <Text style={styles.toggleText}>Resend reset code</Text>
              </TouchableOpacity>
            </>
          )}

          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.toggle}>
            <Text style={styles.toggleText}>Back to Sign In</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1, backgroundColor: '#f9fafb'},
  container: {flexGrow: 1, justifyContent: 'center', padding: 24},
  logoContainer: {alignItems: 'center', marginBottom: 40},
  logo: {fontSize: 32, fontWeight: '800', color: '#6366f1'},
  subtitle: {fontSize: 14, color: '#6b7280', marginTop: 4},
  card: {backgroundColor: '#fff', borderRadius: 14, padding: 20, borderWidth: 1, borderColor: '#e5e7eb'},
  title: {fontSize: 20, fontWeight: '700', color: '#111827', marginBottom: 4},
  desc: {fontSize: 13, color: '#6b7280', marginBottom: 20},
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 15,
    color: '#111827',
    marginBottom: 14,
  },
  button: {
    backgroundColor: '#6366f1',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonText: {color: '#fff', fontSize: 16, fontWeight: '700'},
  toggle: {alignItems: 'center', marginTop: 16},
  toggleText: {color: '#6366f1', fontSize: 14},
});
