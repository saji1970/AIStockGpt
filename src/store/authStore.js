import { create } from 'zustand';
import { loginUser, registerUser, getProfile } from '../services/api';

const useAuthStore = create((set, get) => ({
  user: null,
  token: localStorage.getItem('access_token'),
  refreshToken: localStorage.getItem('refresh_token'),
  isAuthenticated: !!localStorage.getItem('access_token'),
  isLoading: true,

  initialize: async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      set({ isLoading: false, isAuthenticated: false });
      return;
    }
    try {
      const user = await getProfile();
      set({ user, isAuthenticated: true, isLoading: false });
    } catch {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      set({ user: null, token: null, refreshToken: null, isAuthenticated: false, isLoading: false });
    }
  },

  login: async (email, password) => {
    const data = await loginUser(email, password);
    const { access_token, refresh_token } = data;
    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);
    set({ token: access_token, refreshToken: refresh_token, isAuthenticated: true });
    const user = await getProfile();
    set({ user });
  },

  register: async (email, password, firstName, lastName, username) => {
    await registerUser({
      email,
      password,
      first_name: firstName,
      last_name: lastName,
      username: username || undefined,
    });
    await get().login(email, password);
  },

  logout: () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    set({ user: null, token: null, refreshToken: null, isAuthenticated: false });
  },
}));

export default useAuthStore;
