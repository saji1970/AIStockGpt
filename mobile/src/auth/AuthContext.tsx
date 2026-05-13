import React, {createContext, useContext, useState, useEffect, useCallback, ReactNode} from 'react';
import * as api from '../api/client';
import {saveTokens, clearTokens, getAccessToken} from '../storage/tokenStorage';

interface User {
  id: string;
  user_id: string;
  email: string;
  first_name: string;
  last_name: string;
  username: string | null;
  is_active: boolean;
}

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, firstName: string, lastName: string, username?: string) => Promise<void>;
  logout: () => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  login: async () => {},
  register: async () => {},
  logout: async () => {},
  changePassword: async () => {},
});

export function AuthProvider({children}: {children: ReactNode}) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // On app load, check for existing token
  useEffect(() => {
    (async () => {
      try {
        const token = await getAccessToken();
        if (token) {
          const profile = await api.getProfile();
          setUser(profile);
        }
      } catch {
        await clearTokens();
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const data = await api.login(email, password);
    await saveTokens(data.access_token, data.refresh_token);
    const profile = await api.getProfile();
    setUser(profile);
  }, []);

  const register = useCallback(
    async (email: string, password: string, firstName: string, lastName: string, username?: string) => {
      await api.register(email, password, firstName, lastName, username);
      // Auto-login after register
      await login(email, password);
    },
    [login],
  );

  const logout = useCallback(async () => {
    await clearTokens();
    setUser(null);
  }, []);

  const changePassword = useCallback(async (currentPassword: string, newPassword: string) => {
    await api.changePassword(currentPassword, newPassword);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
        changePassword,
      }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
