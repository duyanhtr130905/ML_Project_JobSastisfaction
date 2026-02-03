# Mobile Translation App

Cross-platform mobile application for EN-VI scientific document translation.

## Overview

This mobile app provides a user-friendly interface for:
- Real-time English to Vietnamese translation
- Offline translation using Edge AI
- Access to translation history
- Terminology lookup
- Document scanning and translation

## Technology Stack

- **Framework**: React Native (cross-platform iOS & Android)
- **State Management**: Redux Toolkit
- **API Client**: Axios
- **UI Components**: React Native Paper
- **Navigation**: React Navigation

## Project Structure

```
mobile/
├── src/
│   ├── components/          # Reusable UI components
│   ├── screens/             # App screens
│   ├── navigation/          # Navigation configuration
│   ├── services/            # API services
│   ├── store/               # Redux store
│   ├── utils/               # Utility functions
│   └── App.tsx              # Main app component
├── android/                 # Android specific files
├── ios/                     # iOS specific files
├── package.json
└── README.md
```

## Setup

### Prerequisites

- Node.js 16+ and npm/yarn
- React Native CLI
- Android Studio (for Android development)
- Xcode (for iOS development, macOS only)

### Installation

```bash
# Install dependencies
npm install

# iOS only (macOS)
cd ios && pod install && cd ..

# Run on Android
npm run android

# Run on iOS
npm run ios
```

## Features

### 1. Translation Screen
- Input English text
- Real-time translation to Vietnamese
- Copy translation result
- Share translation
- Save to history

### 2. History Screen
- View past translations
- Search translation history
- Re-translate or edit

### 3. Terminology Screen
- Browse scientific terminology
- Search terms by category
- Add custom terminology

### 4. Settings Screen
- Configure API endpoint
- Enable/disable offline mode
- Toggle translation memory
- Toggle terminology standardization

## Configuration

Create a `config.js` file:

```javascript
export const API_CONFIG = {
  BASE_URL: 'http://your-api-server:8000',
  TIMEOUT: 30000,
};
```

## API Integration

The app connects to the translation system API:

- `POST /translate` - Translate text
- `GET /memory/stats` - Get translation statistics
- `GET /terminology/{term}` - Lookup terminology

## Building for Production

### Android

```bash
cd android
./gradlew assembleRelease
```

APK will be at: `android/app/build/outputs/apk/release/app-release.apk`

### iOS

```bash
# Open Xcode
open ios/TranslationApp.xcworkspace

# Build using Xcode or:
xcodebuild -workspace ios/TranslationApp.xcworkspace \
  -scheme TranslationApp \
  -configuration Release \
  archive
```

## Testing

```bash
# Run tests
npm test

# Run e2e tests
npm run test:e2e
```

## Deployment

### Android
- Build signed APK
- Upload to Google Play Console
- Follow Play Store guidelines

### iOS
- Archive app in Xcode
- Upload to App Store Connect
- Submit for review

## Support

For issues or questions, contact the development team.
