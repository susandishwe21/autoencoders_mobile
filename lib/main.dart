import 'package:flutter/material.dart';
import 'screens/autoencoder_screen.dart';

void main() {
  runApp(const ANNApp());
}

class ANNApp extends StatelessWidget {
  const ANNApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MediaQuery(
      data: MediaQuery.of(context).copyWith(textScaleFactor: 1.0),
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        home: AutoencoderScreen(),
      ),
    );
  }
}
