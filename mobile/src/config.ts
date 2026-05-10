// @ts-ignore – __DEV__ is a React Native global
const isDev: boolean = typeof __DEV__ !== 'undefined' ? __DEV__ : false;

export const API_BASE_URL = isDev
  ? 'http://10.0.2.2:8080' // Android emulator maps 10.0.2.2 to host localhost
  : 'https://your-railway-app.railway.app'; // Replace with your Railway URL
