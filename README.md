# Simple FHE Demo

A simple demonstration of Fully Homomorphic Encryption (FHE) using Python and TenSEAL.

## What is FHE?

Fully Homomorphic Encryption allows computation on encrypted data without decrypting it first. This enables privacy-preserving cloud computing where the server can process your data without ever seeing it.

## Features

- Basic FHE operations (addition, multiplication)
- Polynomial evaluation on encrypted data
- Privacy-preserving average computation example
- Clear demonstrations with expected vs actual results

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the demo:
```bash
python simple_fhe.py
```

## Examples Included

1. **Basic Operations**: Demonstrates homomorphic addition and multiplication on encrypted vectors
2. **Polynomial Evaluation**: Computes `3x² + 2x + 1` on encrypted values
3. **Privacy-Preserving Computation**: Calculates average salary without revealing individual salaries

## Technical Details

- Uses CKKS scheme for approximate arithmetic on encrypted real numbers
- Polynomial modulus degree: 8192
- Supports SIMD operations on encrypted vectors

## Requirements

- Python 3.7+
- TenSEAL 0.3.14
- NumPy

## License

MIT