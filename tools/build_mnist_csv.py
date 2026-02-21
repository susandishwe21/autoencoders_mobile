import gzip
import struct
from pathlib import Path

root = Path(__file__).resolve().parents[1]
mnist_dir = root / "assets" / "mnist"
images_gz = mnist_dir / "t10k-images-idx3-ubyte.gz"
labels_gz = mnist_dir / "t10k-labels-idx1-ubyte.gz"
csv_path = mnist_dir / "mnist_test.csv"

with gzip.open(images_gz, "rb") as f:
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != 2051:
        raise RuntimeError("Invalid MNIST images file")
    images = f.read(count * rows * cols)

with gzip.open(labels_gz, "rb") as f:
    magic, label_count = struct.unpack(">II", f.read(8))
    if magic != 2049:
        raise RuntimeError("Invalid MNIST labels file")
    labels = f.read(label_count)

samples = min(count, label_count)
px_count = rows * cols

with csv_path.open("w", encoding="utf-8") as out:
    for i in range(samples):
        start = i * px_count
        pixels = images[start : start + px_count]
        line = str(labels[i]) + "," + ",".join(str(v) for v in pixels)
        out.write(line + "\n")

print(f"Wrote {samples} samples to {csv_path}")
