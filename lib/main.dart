import 'package:flutter/material.dart';
import 'screens/autoencoder_screen.dart';

void main() {
  runApp(const ANNApp());
}

class ANNApp extends StatelessWidget {
  const ANNApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: AutoencoderScreen(),
    );
  }
}
