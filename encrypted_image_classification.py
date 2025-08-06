#!/usr/bin/env python3
"""
Encrypted Image Classification using ZAMA Concrete ML
Demonstrates privacy-preserving image analysis with neural networks
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from concrete.ml.torch.compile import compile_torch_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleNN(nn.Module):
    """Simple neural network for encrypted inference"""
    
    def __init__(self, input_dim=64, hidden_dim=30, output_dim=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EncryptedImageClassifier:
    """Privacy-preserving image classification system"""
    
    def __init__(self):
        self.model = None
        self.compiled_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_digit_dataset(self):
        """Load and prepare digit dataset"""
        print("\n📊 Loading digit dataset...")
        
        # Load digits dataset (8x8 images)
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # Normalize
        X = X / 16.0  # Digits dataset has values 0-16
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Dataset shape: {X.shape}")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Number of classes: {len(np.unique(y))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_neural_network(self):
        """Train a simple neural network"""
        print("\n🧠 Training neural network...")
        
        # Create model
        self.model = SimpleNN(input_dim=64, hidden_dim=30, output_dim=10)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.LongTensor(self.y_train)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Train
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test)
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted.numpy() == self.y_test).mean()
            print(f"\n   Plaintext Model Accuracy: {accuracy:.2%}")
        
        return self.model
    
    def compile_for_fhe(self):
        """Compile the model for FHE execution"""
        print("\n⚙️ Compiling model for FHE...")
        
        # Prepare input set for compilation
        inputset = self.X_train[:100].astype(np.float32)
        
        # Compile the model
        self.compiled_model = compile_torch_model(
            self.model,
            inputset,
            n_bits=8,  # Quantization bits
            rounding_threshold_bits=8
        )
        
        print("✓ Model compiled for homomorphic encryption")
        
        return self.compiled_model
    
    def encrypt_and_classify(self, image_idx=0):
        """Classify an encrypted image"""
        print(f"\n🔒 Classifying encrypted image (index {image_idx})...")
        
        # Get test image
        test_image = self.X_test[image_idx:image_idx+1].astype(np.float32)
        true_label = self.y_test[image_idx]
        
        # Show the image
        self.visualize_image(test_image[0], true_label)
        
        print(f"\n   True label: {true_label}")
        
        # Plaintext prediction
        self.model.eval()
        with torch.no_grad():
            plain_output = self.model(torch.FloatTensor(test_image))
            plain_pred = torch.argmax(plain_output, dim=1).item()
        print(f"   Plaintext prediction: {plain_pred}")
        
        # Encrypted prediction
        print("\n   Encrypting image...")
        print("   Running neural network on encrypted data...")
        
        # Get FHE circuit
        fhe_circuit = self.compiled_model.fhe_circuit
        
        # Encrypt input
        encrypted_input = fhe_circuit.encrypt(test_image)
        
        # Run encrypted inference
        encrypted_output = fhe_circuit.run(encrypted_input)
        
        # Decrypt result
        decrypted_output = fhe_circuit.decrypt(encrypted_output)
        encrypted_pred = np.argmax(decrypted_output[0])
        
        print(f"   Encrypted prediction: {encrypted_pred}")
        print(f"\n   ✓ Match: {plain_pred == encrypted_pred}")
        
        return plain_pred, encrypted_pred
    
    def visualize_image(self, image, label):
        """Visualize a digit image"""
        plt.figure(figsize=(3, 3))
        plt.imshow(image.reshape(8, 8), cmap='gray')
        plt.title(f'Digit: {label}')
        plt.axis('off')
        plt.savefig('/Users/zhangmang/simple-fhe-demo/sample_digit.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("   Image saved as 'sample_digit.png'")
    
    def batch_encrypted_inference(self, n_samples=10):
        """Perform batch encrypted inference"""
        print(f"\n📊 Batch Encrypted Inference ({n_samples} samples):")
        print("-" * 50)
        
        correct_plain = 0
        correct_encrypted = 0
        matches = 0
        
        fhe_circuit = self.compiled_model.fhe_circuit
        
        for i in range(n_samples):
            # Get test sample
            test_image = self.X_test[i:i+1].astype(np.float32)
            true_label = self.y_test[i]
            
            # Plaintext prediction
            self.model.eval()
            with torch.no_grad():
                plain_output = self.model(torch.FloatTensor(test_image))
                plain_pred = torch.argmax(plain_output, dim=1).item()
            
            # Encrypted prediction
            encrypted_input = fhe_circuit.encrypt(test_image)
            encrypted_output = fhe_circuit.run(encrypted_input)
            decrypted_output = fhe_circuit.decrypt(encrypted_output)
            encrypted_pred = np.argmax(decrypted_output[0])
            
            # Update counters
            if plain_pred == true_label:
                correct_plain += 1
            if encrypted_pred == true_label:
                correct_encrypted += 1
            if plain_pred == encrypted_pred:
                matches += 1
            
            # Print results for first 5
            if i < 5:
                match_symbol = "✓" if plain_pred == encrypted_pred else "✗"
                print(f"  Sample {i+1}: True={true_label}, Plain={plain_pred}, Encrypted={encrypted_pred} {match_symbol}")
        
        print("-" * 50)
        print(f"\n  Plaintext Accuracy: {correct_plain}/{n_samples} ({100*correct_plain/n_samples:.1f}%)")
        print(f"  Encrypted Accuracy: {correct_encrypted}/{n_samples} ({100*correct_encrypted/n_samples:.1f}%)")
        print(f"  Prediction Match Rate: {matches}/{n_samples} ({100*matches/n_samples:.1f}%)")

class EncryptedDataAnalytics:
    """Advanced encrypted data analytics using ZAMA technology"""
    
    @staticmethod
    def encrypted_statistics(data):
        """Compute statistics on encrypted data"""
        print("\n📊 Computing Statistics on Encrypted Data:")
        print("-" * 50)
        
        from concrete import fhe
        
        # Define computation function
        @fhe.compiler({"x": "encrypted"})
        def compute_stats(x):
            # Compute multiple statistics in one pass
            sum_val = np.sum(x)
            mean_val = sum_val // len(x)
            max_val = np.max(x)
            min_val = np.min(x)
            return np.array([sum_val, mean_val, max_val, min_val])
        
        # Create sample data
        sample_data = np.random.randint(0, 100, size=10)
        print(f"  Original data: {sample_data}")
        
        # Compile the function
        inputset = [np.random.randint(0, 100, size=10) for _ in range(20)]
        circuit = compute_stats.compile(inputset)
        
        # Encrypt and compute
        print("\n  Encrypting data...")
        encrypted_result = circuit.encrypt_run_decrypt(sample_data)
        
        # Compare with plaintext
        plaintext_result = [
            np.sum(sample_data),
            np.mean(sample_data).astype(int),
            np.max(sample_data),
            np.min(sample_data)
        ]
        
        print(f"\n  Results:")
        print(f"    Sum:  Encrypted={encrypted_result[0]}, Plaintext={plaintext_result[0]}")
        print(f"    Mean: Encrypted={encrypted_result[1]}, Plaintext={plaintext_result[1]}")
        print(f"    Max:  Encrypted={encrypted_result[2]}, Plaintext={plaintext_result[2]}")
        print(f"    Min:  Encrypted={encrypted_result[3]}, Plaintext={plaintext_result[3]}")

def main():
    print("=" * 70)
    print("    ENCRYPTED IMAGE CLASSIFICATION WITH ZAMA CONCRETE ML")
    print("=" * 70)
    
    # Initialize classifier
    classifier = EncryptedImageClassifier()
    
    # Load dataset
    print("\n1️⃣ Loading and preparing data...")
    classifier.load_digit_dataset()
    
    # Train model
    print("\n2️⃣ Training neural network...")
    classifier.train_neural_network()
    
    # Compile for FHE
    print("\n3️⃣ Compiling for homomorphic encryption...")
    classifier.compile_for_fhe()
    
    # Single encrypted classification
    print("\n4️⃣ Demonstrating encrypted image classification...")
    classifier.encrypt_and_classify(image_idx=0)
    
    # Batch encrypted inference
    print("\n5️⃣ Running batch encrypted inference...")
    classifier.batch_encrypted_inference(n_samples=10)
    
    # Advanced analytics
    print("\n6️⃣ Advanced Encrypted Analytics...")
    analytics = EncryptedDataAnalytics()
    analytics.encrypted_statistics(np.random.randint(0, 100, 10))
    
    print("\n" + "=" * 70)
    print("✅ Encrypted image classification demonstration complete!")
    print("\n💡 Key Achievements:")
    print("   • Neural network inference on encrypted images")
    print("   • Privacy-preserving image classification")
    print("   • Statistical computations on encrypted data")
    print("   • Near-identical accuracy to plaintext models")
    print("=" * 70)

if __name__ == "__main__":
    main()