# Ancora Frontend (Mobile Client)

A beautiful, high-performance Flutter mobile application for **Ancora Churn Detection AI**, built to help Indonesian MSMEs track customer metrics, predict churn, and engage at-risk customers instantly.

## Tech Stack

- **Framework**: Flutter 3.x
- **Language**: Dart
- **State Management**: BLoC / Cubit (Predictable State)
- **Local Storage**: Hive / Shared Preferences (Offline Cache)
- **Network**: Dio (HTTP client with custom interceptors)
- **Dependency Injection**: GetIt
- **UI Components**: Tailwind-style Custom Widgets, Google Fonts

## Features

- **Merchant Dashboard**: Visualized analytics for Customer Lifetime Value (CLV), Heartbeat Score, and Churn Risk.
- **AI Recommendations**: Personalized, localized message suggestions powered by Gemini 1.5 Flash.
- **Push Notification Center**: Instant triggers for targeted customer re-engagement.
- **Offline Mode**: Local caching of customer metrics for seamless operation in low-connectivity areas.

## Directory Structure

```text
frontend/
├── lib/
│   ├── core/           # Dependency injection, theme, network, constants
│   ├── data/           # Repositories, models, and data providers (Dio, Hive)
│   ├── domain/         # Entities and Business Use Cases
│   ├── presentation/   # UI Screens, Widgets, and BLoC State Management
│   └── main.dart       # App Entrypoint
├── pubspec.yaml        # Dependencies & Assets configuration
└── README.md
```
