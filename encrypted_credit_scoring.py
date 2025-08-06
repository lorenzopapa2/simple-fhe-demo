#!/usr/bin/env python3
"""
Encrypted Credit Scoring Application using ZAMA Concrete ML
Demonstrates privacy-preserving credit risk assessment
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from concrete.ml.sklearn import RandomForestClassifier as ConcreteRFC
import warnings
warnings.filterwarnings('ignore')

class EncryptedCreditScoring:
    """Privacy-preserving credit scoring system using FHE"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.concrete_model = None
        self.sklearn_model = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic credit data for demonstration"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'num_credit_cards': np.random.poisson(3, n_samples),
            'credit_utilization': np.random.beta(2, 5, n_samples),
            'payment_history': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),  # 0=good, 1=late, 2=default
            'debt_to_income': np.random.beta(2, 8, n_samples),
            'num_loans': np.random.poisson(2, n_samples),
            'bankruptcy': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'num_inquiries': np.random.poisson(1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target (credit approval) based on features
        score = (
            (df['income'] > 40000) * 0.3 +
            (df['employment_years'] > 2) * 0.2 +
            (df['credit_utilization'] < 0.3) * 0.2 +
            (df['payment_history'] == 0) * 0.3 +
            (df['debt_to_income'] < 0.4) * 0.2 +
            (df['bankruptcy'] == 0) * 0.3 +
            (df['num_inquiries'] < 3) * 0.1
        )
        
        df['approved'] = (score + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df.drop('approved', axis=1)
        y = df['approved']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_plaintext_model(self, X_train, y_train):
        """Train a standard scikit-learn model for comparison"""
        print("\n📊 Training plaintext model for comparison...")
        self.sklearn_model = RandomForestClassifier(
            n_estimators=10,
            max_depth=4,
            random_state=42
        )
        self.sklearn_model.fit(X_train, y_train)
        return self.sklearn_model
    
    def train_encrypted_model(self, X_train, y_train):
        """Train a Concrete ML model that supports FHE"""
        print("\n🔐 Training FHE-compatible model...")
        
        # Create Concrete ML model with quantization
        self.concrete_model = ConcreteRFC(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            n_bits=8  # Quantization bits for FHE
        )
        
        # Train the model
        self.concrete_model.fit(X_train, y_train)
        
        return self.concrete_model
    
    def compile_for_fhe(self, X_sample):
        """Compile the model for FHE execution"""
        print("\n⚙️ Compiling model for FHE execution...")
        
        # Compile the model with representative data
        self.concrete_model.compile(X_sample)
        
        print("✓ Model compiled for homomorphic encryption")
        
    def predict_encrypted(self, X_test, show_details=True):
        """Make predictions on encrypted data"""
        print("\n🔒 Making predictions on encrypted data...")
        
        # Get FHE circuit
        fhe_circuit = self.concrete_model.fhe_circuit
        
        predictions = []
        
        for i, sample in enumerate(X_test[:5]):  # Demo with first 5 samples
            if show_details and i == 0:
                print(f"\n  Sample {i+1}:")
                print(f"    Encrypting input data...")
            
            # Encrypt the input
            encrypted_input = fhe_circuit.encrypt(sample.reshape(1, -1))
            
            if show_details and i == 0:
                print(f"    Running inference on encrypted data...")
            
            # Run encrypted prediction
            encrypted_output = fhe_circuit.run(encrypted_input)
            
            if show_details and i == 0:
                print(f"    Decrypting result...")
            
            # Decrypt the result
            decrypted_prediction = fhe_circuit.decrypt(encrypted_output)
            predictions.append(decrypted_prediction[0])
            
            if show_details and i == 0:
                print(f"    ✓ Prediction: {'Approved' if decrypted_prediction[0] == 1 else 'Denied'}")
        
        return np.array(predictions)
    
    def compare_performance(self, X_test, y_test):
        """Compare plaintext vs encrypted model performance"""
        print("\n📈 Performance Comparison:")
        print("-" * 50)
        
        # Plaintext predictions
        plain_pred = self.sklearn_model.predict(X_test[:5])
        
        # Encrypted predictions
        enc_pred = self.predict_encrypted(X_test[:5], show_details=False)
        
        # Calculate accuracy
        plain_acc = np.mean(plain_pred == y_test[:5])
        enc_acc = np.mean(enc_pred == y_test[:5])
        
        print(f"\n  Plaintext Model Accuracy: {plain_acc:.2%}")
        print(f"  Encrypted Model Accuracy: {enc_acc:.2%}")
        print(f"  Match Rate: {np.mean(plain_pred == enc_pred):.2%}")
        
        # Show sample predictions
        print("\n  Sample Predictions:")
        print("  " + "-" * 45)
        print("  | Sample | Actual | Plaintext | Encrypted |")
        print("  " + "-" * 45)
        for i in range(5):
            actual = "Approved" if y_test.iloc[i] == 1 else "Denied"
            plain = "Approved" if plain_pred[i] == 1 else "Denied"
            enc = "Approved" if enc_pred[i] == 1 else "Denied"
            match = "✓" if plain == enc else "✗"
            print(f"  |   {i+1}    | {actual:8} | {plain:9} | {enc:9} | {match}")
        print("  " + "-" * 45)

def main():
    print("=" * 60)
    print("    ENCRYPTED CREDIT SCORING WITH ZAMA CONCRETE ML")
    print("=" * 60)
    
    # Initialize system
    system = EncryptedCreditScoring()
    
    # Generate synthetic data
    print("\n1️⃣ Generating synthetic credit data...")
    df = system.generate_synthetic_data(1000)
    print(f"   Generated {len(df)} credit applications")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Approval rate: {df['approved'].mean():.2%}")
    
    # Prepare data
    print("\n2️⃣ Preparing data for training...")
    X_train, X_test, y_train, y_test = system.prepare_data(df)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train models
    print("\n3️⃣ Training models...")
    system.train_plaintext_model(X_train, y_train)
    system.train_encrypted_model(X_train, y_train)
    
    # Compile for FHE
    print("\n4️⃣ Preparing for homomorphic encryption...")
    system.compile_for_fhe(X_train[:100])
    
    # Make encrypted predictions
    print("\n5️⃣ Demonstrating encrypted credit scoring...")
    system.predict_encrypted(X_test[:1], show_details=True)
    
    # Compare performance
    print("\n6️⃣ Comparing models...")
    system.compare_performance(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("✅ Encrypted credit scoring demonstration complete!")
    print("\n💡 Key Benefits:")
    print("   • Credit decisions without exposing personal data")
    print("   • Compliance with privacy regulations")
    print("   • No data breaches possible - data stays encrypted")
    print("   • Same accuracy as plaintext models")
    print("=" * 60)

if __name__ == "__main__":
    main()