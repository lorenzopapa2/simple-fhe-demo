#!/usr/bin/env python3
"""
Simple Fully Homomorphic Encryption (FHE) Demo
This demonstrates basic FHE operations using the tenseal library
"""

import tenseal as ts
import numpy as np

def create_context():
    """Create and configure the FHE context"""
    # Create context with specific parameters
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,  # Using CKKS scheme for approximate arithmetic
        poly_modulus_degree=8192,  # Polynomial modulus degree
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # Coefficient modulus bit sizes
    )
    # Generate galois keys for rotations
    context.generate_galois_keys()
    # Set the scale for encoding
    context.global_scale = 2**40
    return context

def encrypt_data(context, data):
    """Encrypt plaintext data"""
    encrypted = ts.ckks_vector(context, data)
    return encrypted

def decrypt_data(encrypted_data):
    """Decrypt encrypted data"""
    return encrypted_data.decrypt()

def homomorphic_addition(enc1, enc2):
    """Perform addition on encrypted data"""
    return enc1 + enc2

def homomorphic_multiplication(enc1, enc2):
    """Perform multiplication on encrypted data"""
    return enc1 * enc2

def homomorphic_polynomial(encrypted_x):
    """Compute polynomial: 3x^2 + 2x + 1 on encrypted data"""
    # 3x^2
    term1 = encrypted_x * encrypted_x * 3
    # 2x
    term2 = encrypted_x * 2
    # Add all terms together with constant 1
    result = term1 + term2 + 1
    return result

def demo_basic_operations():
    """Demonstrate basic FHE operations"""
    print("=== FHE Basic Operations Demo ===\n")
    
    # Create FHE context
    print("1. Creating FHE context...")
    context = create_context()
    
    # Define plaintext values
    value1 = [3.5, 4.2, 1.1, 2.8]
    value2 = [1.5, 2.3, 0.9, 1.2]
    
    print(f"2. Original values:")
    print(f"   Value 1: {value1}")
    print(f"   Value 2: {value2}\n")
    
    # Encrypt the values
    print("3. Encrypting values...")
    enc1 = encrypt_data(context, value1)
    enc2 = encrypt_data(context, value2)
    print("   ✓ Encryption complete\n")
    
    # Perform homomorphic addition
    print("4. Performing homomorphic addition...")
    enc_sum = homomorphic_addition(enc1, enc2)
    decrypted_sum = decrypt_data(enc_sum)
    expected_sum = [a + b for a, b in zip(value1, value2)]
    print(f"   Decrypted result: {[round(x, 2) for x in decrypted_sum]}")
    print(f"   Expected result:  {[round(x, 2) for x in expected_sum]}\n")
    
    # Perform homomorphic multiplication
    print("5. Performing homomorphic multiplication...")
    enc_product = homomorphic_multiplication(enc1, enc2)
    decrypted_product = decrypt_data(enc_product)
    expected_product = [a * b for a, b in zip(value1, value2)]
    print(f"   Decrypted result: {[round(x, 2) for x in decrypted_product]}")
    print(f"   Expected result:  {[round(x, 2) for x in expected_product]}\n")
    
    # Perform polynomial evaluation
    print("6. Evaluating polynomial 3x^2 + 2x + 1 on encrypted data...")
    x_values = [2.0, 3.0, 1.5]
    enc_x = encrypt_data(context, x_values)
    enc_poly_result = homomorphic_polynomial(enc_x)
    decrypted_poly = decrypt_data(enc_poly_result)
    expected_poly = [3*x**2 + 2*x + 1 for x in x_values]
    print(f"   Input x values:   {x_values}")
    print(f"   Decrypted result: {[round(x, 2) for x in decrypted_poly]}")
    print(f"   Expected result:  {[round(x, 2) for x in expected_poly]}\n")

def demo_privacy_preserving_computation():
    """Demonstrate privacy-preserving computation scenario"""
    print("=== Privacy-Preserving Computation Demo ===\n")
    print("Scenario: Computing average salary without revealing individual salaries\n")
    
    # Create context
    context = create_context()
    
    # Individual salaries (in thousands)
    salaries = [75.5, 82.3, 91.7, 68.2, 77.9]
    print(f"Original salaries (hidden from server): ${salaries} thousand")
    
    # Client encrypts their data
    print("\nClient side: Encrypting salaries...")
    encrypted_salaries = encrypt_data(context, salaries)
    
    # Server performs computation on encrypted data
    print("Server side: Computing sum on encrypted data...")
    # Sum all salaries (homomorphic operation)
    encrypted_sum = encrypted_salaries.sum()
    
    # Divide by count to get average (homomorphic scalar multiplication)
    count = len(salaries)
    encrypted_average = encrypted_sum * (1.0 / count)
    
    # Client decrypts the result
    print("Client side: Decrypting result...")
    average_salary = decrypt_data(encrypted_average)
    expected_average = sum(salaries) / len(salaries)
    
    print(f"\nComputed average salary: ${average_salary:.2f} thousand")
    print(f"Expected average:        ${expected_average:.2f} thousand")
    print("\n✓ The server computed the average without seeing individual salaries!")

if __name__ == "__main__":
    print("=" * 50)
    print("     FULLY HOMOMORPHIC ENCRYPTION DEMO")
    print("=" * 50)
    print()
    
    # Run basic operations demo
    demo_basic_operations()
    
    print("=" * 50)
    print()
    
    # Run privacy-preserving computation demo
    demo_privacy_preserving_computation()
    
    print()
    print("=" * 50)
    print("Demo completed successfully!")