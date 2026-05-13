import React from 'react';
import {createStackNavigator} from '@react-navigation/stack';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {useAuth} from '../auth/AuthContext';

import LoginScreen from '../screens/LoginScreen';
import ForgotPasswordScreen from '../screens/ForgotPasswordScreen';
import ChangePasswordScreen from '../screens/ChangePasswordScreen';
import ChatScreen from '../screens/ChatScreen';
import PortfolioScreen from '../screens/PortfolioScreen';
import AlertsScreen from '../screens/AlertsScreen';
import ProfileScreen from '../screens/ProfileScreen';
import {ActivityIndicator, View} from 'react-native';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: '#6366f1',
        tabBarInactiveTintColor: '#9ca3af',
        tabBarStyle: {borderTopColor: '#e5e7eb', paddingBottom: 4, height: 56},
        tabBarLabelStyle: {fontSize: 11, fontWeight: '600'},
        headerStyle: {backgroundColor: '#6366f1'},
        headerTintColor: '#fff',
        headerTitleStyle: {fontWeight: '700'},
      }}>
      <Tab.Screen name="Chat" component={ChatScreen} options={{title: 'AI Chat', tabBarLabel: 'Chat'}} />
      <Tab.Screen name="Portfolio" component={PortfolioScreen} options={{title: 'Portfolios', tabBarLabel: 'Portfolio'}} />
      <Tab.Screen name="Alerts" component={AlertsScreen} options={{title: 'Alerts', tabBarLabel: 'Alerts'}} />
      <Tab.Screen name="Profile" component={ProfileScreen} options={{title: 'Profile', tabBarLabel: 'Profile'}} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  const {isAuthenticated, isLoading} = useAuth();

  if (isLoading) {
    return (
      <View style={{flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#f9fafb'}}>
        <ActivityIndicator size="large" color="#6366f1" />
      </View>
    );
  }

  return (
    <Stack.Navigator screenOptions={{headerShown: false}}>
      {isAuthenticated ? (
        <>
          <Stack.Screen name="Main" component={MainTabs} />
          <Stack.Screen
            name="ChangePassword"
            component={ChangePasswordScreen}
            options={{headerShown: true, title: 'Change Password', headerStyle: {backgroundColor: '#6366f1'}, headerTintColor: '#fff', headerTitleStyle: {fontWeight: '700'}}}
          />
        </>
      ) : (
        <>
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen
            name="ForgotPassword"
            component={ForgotPasswordScreen}
            options={{headerShown: true, title: 'Forgot Password', headerStyle: {backgroundColor: '#6366f1'}, headerTintColor: '#fff', headerTitleStyle: {fontWeight: '700'}}}
          />
        </>
      )}
    </Stack.Navigator>
  );
}
