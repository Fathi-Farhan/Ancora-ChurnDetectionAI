import 'package:flutter/material.dart';

void main() {
  runApp(const AncoraApp());
}

class AncoraApp extends StatelessWidget {
  const AncoraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ancora Churn Detection AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F172A),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ancora Dashboard'),
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.analytics_outlined,
              size: 80,
              color: Colors.blueAccent,
            ),
            const SizedBox(height: 16),
            const Text(
              'Ancora Customer Retention Platform',
              style: TextStyle(fontSize: 20, fontWeight: 'bold'),
            ),
            const SizedBox(height: 8),
            Text(
              'AI-Powered Customer Churn Prediction',
              style: TextStyle(color: Colors.grey.shade400),
            ),
          ],
        ),
      ),
    );
  }
}
