/**
 * Translation App - Main Component
 * EN-VI Scientific Document Translation
 */

import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {Provider as PaperProvider} from 'react-native-paper';
import {Provider as ReduxProvider} from 'react-redux';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

import TranslateScreen from './screens/TranslateScreen';
import HistoryScreen from './screens/HistoryScreen';
import TerminologyScreen from './screens/TerminologyScreen';
import SettingsScreen from './screens/SettingsScreen';
import {store} from './store';

const Tab = createBottomTabNavigator();

const App = () => {
  return (
    <ReduxProvider store={store}>
      <PaperProvider>
        <NavigationContainer>
          <Tab.Navigator
            screenOptions={({route}) => ({
              tabBarIcon: ({focused, color, size}) => {
                let iconName;

                if (route.name === 'Translate') {
                  iconName = 'translate';
                } else if (route.name === 'History') {
                  iconName = 'history';
                } else if (route.name === 'Terminology') {
                  iconName = 'book-open-variant';
                } else if (route.name === 'Settings') {
                  iconName = 'cog';
                }

                return <Icon name={iconName} size={size} color={color} />;
              },
              tabBarActiveTintColor: '#6200ee',
              tabBarInactiveTintColor: 'gray',
            })}>
            <Tab.Screen
              name="Translate"
              component={TranslateScreen}
              options={{title: 'Translate'}}
            />
            <Tab.Screen
              name="History"
              component={HistoryScreen}
              options={{title: 'History'}}
            />
            <Tab.Screen
              name="Terminology"
              component={TerminologyScreen}
              options={{title: 'Terminology'}}
            />
            <Tab.Screen
              name="Settings"
              component={SettingsScreen}
              options={{title: 'Settings'}}
            />
          </Tab.Navigator>
        </NavigationContainer>
      </PaperProvider>
    </ReduxProvider>
  );
};

export default App;
