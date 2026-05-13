import React, {useState} from 'react';
import {View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView} from 'react-native';
import {useAuth} from '../auth/AuthContext';

export default function LoginScreen({navigation}: any) {
  const {login, register} = useAuth();
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!email.trim() || !password.trim()) {
      Alert.alert('Error', 'Email and password are required');
      return;
    }
    if (isRegister && (!firstName.trim() || !lastName.trim())) {
      Alert.alert('Error', 'First and last name are required');
      return;
    }

    setLoading(true);
    try {
      if (isRegister) {
        await register(email, password, firstName, lastName, username || undefined);
      } else {
        await login(email, password);
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Something went wrong';
      Alert.alert('Error', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.logoContainer}>
          <Text style={styles.logo}>AI Stock GPT</Text>
          <Text style={styles.subtitle}>Intelligent Stock Analysis</Text>
        </View>

        <View style={styles.form}>
          {isRegister && (
            <>
              <TextInput
                style={styles.input}
                placeholder="First Name"
                placeholderTextColor="#9ca3af"
                value={firstName}
                onChangeText={setFirstName}
              />
              <TextInput
                style={styles.input}
                placeholder="Last Name"
                placeholderTextColor="#9ca3af"
                value={lastName}
                onChangeText={setLastName}
              />
              <TextInput
                style={styles.input}
                placeholder="Username (optional)"
                placeholderTextColor="#9ca3af"
                value={username}
                onChangeText={setUsername}
                autoCapitalize="none"
              />
            </>
          )}

          <TextInput
            style={styles.input}
            placeholder="Email"
            placeholderTextColor="#9ca3af"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
          />
          <TextInput
            style={styles.input}
            placeholder="Password"
            placeholderTextColor="#9ca3af"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          <TouchableOpacity style={styles.button} onPress={handleSubmit} disabled={loading} activeOpacity={0.8}>
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>{isRegister ? 'Create Account' : 'Sign In'}</Text>
            )}
          </TouchableOpacity>

          {!isRegister && (
            <TouchableOpacity onPress={() => navigation.navigate('ForgotPassword')} style={styles.toggle}>
              <Text style={styles.toggleText}>Forgot Password?</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity onPress={() => setIsRegister(!isRegister)} style={styles.toggle}>
            <Text style={styles.toggleText}>
              {isRegister ? 'Already have an account? Sign In' : "Don't have an account? Register"}
            </Text>
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
  form: {gap: 14},
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 15,
    color: '#111827',
  },
  button: {
    backgroundColor: '#6366f1',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonText: {color: '#fff', fontSize: 16, fontWeight: '700'},
  toggle: {alignItems: 'center', marginTop: 12},
  toggleText: {color: '#6366f1', fontSize: 14},
});
