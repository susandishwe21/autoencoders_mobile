import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'dart:math';
import 'package:image/image.dart' as img;

class AutoencoderModel {
  final String name;
  final String value;

  AutoencoderModel(this.name, this.value);

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is AutoencoderModel &&
        other.name == name &&
        other.value == value;
  }

  @override
  int get hashCode => Object.hash(name, value);
}

class AutoencoderScreen extends StatefulWidget {
  const AutoencoderScreen({super.key});
  @override
  AutoencoderScreenState createState() => AutoencoderScreenState();
}

class AutoencoderScreenState extends State<AutoencoderScreen> {
  late Model model;
  Uint8List? inputImage;
  Uint8List? outputImage;
  String? _inputImageSize;
  String? _outputImageSize;
  bool _modelReady = false;
  bool _validatingModel = false;
  bool _mnistReady = false;
  String? _modelStatus;
  String? _sampleInfo;
  String? _lastPredictError;
  List<String> _mnistRows = [];
  List<AutoencoderModel> autoencoderModels = [
    AutoencoderModel("Vanilla Autoencoder", "vanilla_autoencoder_ts"),
    AutoencoderModel("Multilayer Autoencoder", "multilayer_autoencoder_ts"),
    AutoencoderModel("Convolutional Autoencoder", "conv_autoencoder_ts"),
    AutoencoderModel("Sparse Autoencoder", "sparse_autoencoder_ts"),
    AutoencoderModel("Denoising Autoencoder", "denoising_autoencoder_ts"),
  ];
  AutoencoderModel selectedAutoencoder = AutoencoderModel(
    "Convolutional Autoencoder",
    "conv_autoencoder_ts",
  );
  @override
  void initState() {
    super.initState();
    _loadMnistDataset();
    loadModel();
  }

  Future _loadMnistDataset() async {
    final csv = await rootBundle.loadString('assets/mnist/mnist_test.csv');
    final rows = csv
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();
    if (!mounted) return;
    setState(() {
      _mnistRows = rows;
      _mnistReady = rows.isNotEmpty;
    });
  }

  Future loadModel() async {
    if (mounted) {
      setState(() {
        _validatingModel = true;
        _modelStatus = null;
      });
    }

    model = await PyTorchMobile.loadModel(
      "assets/models/${selectedAutoencoder.value}.pt",
    );

    final isValid = await _validateModelSignature();

    if (!mounted) return;
    setState(() {
      _modelReady = isValid;
      _validatingModel = false;
      _modelStatus = isValid
          ? "Input signature OK"
          : "Input signature failed for this model${_lastPredictError != null ? ': $_lastPredictError' : ''}";
    });
  }

  Future<bool> _validateModelSignature() async {
    final sample = List<double>.filled(28 * 28, 0.0);
    final output = await _predictWithFallbacks(
      sample,
      _shapeCandidatesForModel(),
    );
    return output != null && output.isNotEmpty;
  }

  Future runMnistSample() async {
    if (!_mnistReady || _mnistRows.isEmpty) return;

    final rowIndex = Random().nextInt(_mnistRows.length);
    final values = _mnistRows[rowIndex].split(',');
    if (values.length < 785) return;

    final label = int.tryParse(values[0]) ?? -1;
    final pixels = List<int>.generate(784, (i) => int.parse(values[i + 1]));
    final input = pixels.map((value) => value / 255.0).toList();
    final modelInput = selectedAutoencoder.value.contains("denoising")
        ? _addMaskNoise(input, keepProbability: 0.7)
        : input;

    final originalPng = _vectorToPng(modelInput);
    final output = await _predictWithFallbacks(
      modelInput,
      _shapeCandidatesForModel(),
    );
    final outputValues = _flattenTensor(output);
    if (outputValues.length < 784) {
      if (!mounted) return;
      setState(() {
        _sampleInfo = 'MNIST sample row index #$rowIndex, label: $label';
      });
      return;
    }

    final reconstructedPng = _vectorToPng(outputValues);
    final inputSize = _formatImageSize(originalPng);
    final outputSize = _formatImageSize(reconstructedPng);
    if (!mounted) return;
    setState(() {
      inputImage = originalPng;
      outputImage = reconstructedPng;
      _inputImageSize = inputSize;
      _outputImageSize = outputSize;
      _sampleInfo = 'MNIST sample row index #$rowIndex, label: $label';
    });
  }

  String? _formatImageSize(Uint8List imageBytes) {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return null;
    return '${decoded.width} × ${decoded.height} px';
  }

  Uint8List _vectorToPng(List<double> values) {
    final image = img.Image(width: 28, height: 28);
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        final index = y * 28 + x;
        if (index >= values.length) continue;
        final value = (values[index].clamp(0.0, 1.0) * 255).round();
        image.setPixelRgba(x, y, value, value, value, 255);
      }
    }
    return Uint8List.fromList(img.encodePng(image));
  }

  Future<List?> _predictWithFallbacks(
    List<double> input,
    List<List<int>> shapeCandidates,
  ) async {
    _lastPredictError = null;
    for (final shape in shapeCandidates) {
      try {
        final firstTry = await model.getPrediction(input, shape, DType.float32);
        if (firstTry != null && firstTry.isNotEmpty) return firstTry;
      } catch (e) {
        _lastPredictError =
            'shape=$shape dtype=float32 (${e.toString().split('\n').first})';
      }

      try {
        final secondTry = await model.getPrediction(
          input,
          shape,
          DType.float64,
        );
        if (secondTry != null && secondTry.isNotEmpty) return secondTry;
      } catch (e) {
        _lastPredictError =
            'shape=$shape dtype=float64 (${e.toString().split('\n').first})';
      }
    }

    return null;
  }

  List<List<int>> _shapeCandidatesForModel() {
    if (selectedAutoencoder.value.contains("conv") ||
        selectedAutoencoder.value.contains("denoising")) {
      return const <List<int>>[
        <int>[1, 1, 28, 28],
        <int>[1, 28, 28],
        <int>[1, 784],
        <int>[784],
      ];
    }
    return const <List<int>>[
      <int>[1, 784],
      <int>[784],
      <int>[1, 28, 28],
      <int>[1, 1, 28, 28],
    ];
  }

  List<double> _addMaskNoise(
    List<double> values, {
    double keepProbability = 0.7,
  }) {
    final random = Random();
    return values
        .map((v) => random.nextDouble() < keepProbability ? v : 0.0)
        .toList();
  }

  List<double> _flattenTensor(dynamic data) {
    if (data == null) return <double>[];
    if (data is num) return <double>[data.toDouble()];
    if (data is List) {
      final values = <double>[];
      for (final item in data) {
        values.addAll(_flattenTensor(item));
      }
      return values;
    }
    return <double>[];
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final previewSize = ((screenWidth - 48) / 2).clamp(80.0, 100.0);

    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Autoencoder Test",
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
        ),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(height: 15),
          Container(
            alignment: Alignment.centerLeft,
            padding: EdgeInsets.symmetric(horizontal: 16, vertical: 0),
            child: Text(
              "Select Autoencoder Model:",
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
            ),
          ),
          SizedBox(height: 15),
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            padding: EdgeInsets.symmetric(horizontal: 10),
            width: MediaQuery.of(context).size.width,
            height: 50,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey),
              borderRadius: BorderRadius.circular(8),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<AutoencoderModel>(
                value: selectedAutoencoder,
                items: autoencoderModels
                    .map((e) => DropdownMenuItem(value: e, child: Text(e.name)))
                    .toList(),
                onChanged: (val) {
                  setState(() {
                    selectedAutoencoder = val!;
                    _modelReady = false;
                    inputImage = null;
                    outputImage = null;
                    _inputImageSize = null;
                    _outputImageSize = null;
                    _sampleInfo = null;
                  });
                  loadModel();
                },
              ),
            ),
          ),
          SizedBox(height: 20),
          Container(
            width: MediaQuery.of(context).size.width,
            height: 50,
            margin: const EdgeInsets.symmetric(horizontal: 16),
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                backgroundColor: Colors.purple,
              ),
              onPressed: (_modelReady && _mnistReady) ? runMnistSample : null,
              child: Text(
                _modelReady
                    ? (_mnistReady
                          ? "Load Random MNIST Sample"
                          : "Loading MNIST...")
                    : "Loading model...",
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ),
          ),
          SizedBox(height: 20),
          if (_sampleInfo != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(_sampleInfo!),
            ),
          if (_validatingModel)
            const Padding(
              padding: EdgeInsets.only(top: 8),
              child: Text("Validating model input signature..."),
            ),
          if (_modelStatus != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                _modelStatus!,
                style: TextStyle(
                  color: _modelReady ? Colors.green : Colors.red,
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                ),
              ),
            ),
          SizedBox(height: 20),
          if (inputImage != null || outputImage != null)
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Column(
                  children: [
                    const Text(
                      "Original",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 8),
                    if (inputImage != null)
                      Container(
                        width: previewSize,
                        height: previewSize,
                        color: Colors.black,
                        child: Image.memory(
                          inputImage!,
                          width: previewSize,
                          height: previewSize,
                          fit: BoxFit.fill,
                          filterQuality: FilterQuality.none,
                        ),
                      ),
                    if (_inputImageSize != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(
                          _inputImageSize!,
                          style: const TextStyle(fontSize: 12),
                        ),
                      ),
                  ],
                ),
                Column(
                  children: [
                    const Text(
                      "Reconstructed",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 8),
                    if (outputImage != null)
                      Container(
                        width: previewSize,
                        height: previewSize,
                        color: Colors.black,
                        child: Image.memory(
                          outputImage!,
                          width: previewSize,
                          height: previewSize,
                          fit: BoxFit.fill,
                          filterQuality: FilterQuality.none,
                        ),
                      ),
                    if (_outputImageSize != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(
                          _outputImageSize!,
                          style: const TextStyle(fontSize: 12),
                        ),
                      ),
                  ],
                ),
              ],
            ),
        ],
      ),
    );
  }
}
