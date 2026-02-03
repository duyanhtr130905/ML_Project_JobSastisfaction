/**
 * Translation API Service
 */

import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = 'http://localhost:8000'; // Configure this

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface TranslationRequest {
  text: string;
  use_memory?: boolean;
  use_terminology?: boolean;
}

export interface TranslationResponse {
  source: string;
  target: string;
  method: string;
  similar_translations?: any[];
  standardized_terms?: any[];
  timestamp: string;
}

export const translationService = {
  /**
   * Translate text from English to Vietnamese
   */
  async translate(request: TranslationRequest): Promise<TranslationResponse> {
    const response = await apiClient.post('/translate', request);
    return response.data;
  },

  /**
   * Get translation memory statistics
   */
  async getMemoryStats() {
    const response = await apiClient.get('/memory/stats');
    return response.data;
  },

  /**
   * Get terminology information
   */
  async getTerminology(term: string) {
    const response = await apiClient.get(`/terminology/${term}`);
    return response.data;
  },

  /**
   * Get terminology by category
   */
  async getCategoryTerms(category: string) {
    const response = await apiClient.get(`/terminology/category/${category}`);
    return response.data;
  },

  /**
   * Add translation to memory
   */
  async addToMemory(source: string, target: string, metadata?: any) {
    const response = await apiClient.post('/memory/add', {
      source,
      target,
      metadata,
    });
    return response.data;
  },

  /**
   * Get model information
   */
  async getModelInfo() {
    const response = await apiClient.get('/model/info');
    return response.data;
  },
};

/**
 * Local storage service for offline support
 */
export const storageService = {
  /**
   * Save translation to local history
   */
  async saveToHistory(translation: TranslationResponse) {
    try {
      const history = await this.getHistory();
      history.unshift(translation);
      
      // Keep only last 100 translations
      const trimmedHistory = history.slice(0, 100);
      
      await AsyncStorage.setItem('translation_history', JSON.stringify(trimmedHistory));
    } catch (error) {
      console.error('Error saving to history:', error);
    }
  },

  /**
   * Get translation history
   */
  async getHistory(): Promise<TranslationResponse[]> {
    try {
      const historyJson = await AsyncStorage.getItem('translation_history');
      return historyJson ? JSON.parse(historyJson) : [];
    } catch (error) {
      console.error('Error getting history:', error);
      return [];
    }
  },

  /**
   * Clear translation history
   */
  async clearHistory() {
    try {
      await AsyncStorage.removeItem('translation_history');
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  },

  /**
   * Save settings
   */
  async saveSettings(settings: any) {
    try {
      await AsyncStorage.setItem('app_settings', JSON.stringify(settings));
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  },

  /**
   * Get settings
   */
  async getSettings() {
    try {
      const settingsJson = await AsyncStorage.getItem('app_settings');
      return settingsJson ? JSON.parse(settingsJson) : {
        useMemory: true,
        useTerminology: true,
        apiEndpoint: API_BASE_URL,
      };
    } catch (error) {
      console.error('Error getting settings:', error);
      return null;
    }
  },
};
