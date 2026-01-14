import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import sys

# --- CONFIGURATION ---
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 15
DROPOUT_RATE = 0.3
test_size_split = 0.2
seed = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- PART 1: DATA PREPROCESSING ---
def load_and_process_data(filepath):
    """
    Loads CSV, cleans geolocation, engineers temporal features, 
    and normalizes inputs.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Error: File not found. Please ensure 'Crime_Incidents_20250604.csv' is in the directory.")
        sys.exit(1)

    # 1. Feature Engineering: Temporal
    # Convert string datetime to objects
    if 'Incident Datetime' in df.columns:
        df['Incident Datetime'] = pd.to_datetime(df['Incident Datetime'])
        df['hour'] = df['Incident Datetime'].dt.hour
        df['day_of_week'] = df['Incident Datetime'].dt.day_name()
    else:
        print("Error: 'Incident Datetime' column missing.")
        return None

    # 2. Data Cleaning: Spatial (Buffalo, NY approximate bounds)
    initial_count = len(df)
    df = df[(df['Latitude'] >= 42.6) & (df['Latitude'] <= 43.1)]
    df = df[(df['Longitude'] >= -79.0) & (df['Longitude'] <= -78.5)]
    print(f"Filtered {initial_count - len(df)} rows with invalid coordinates.")

    # 3. Encoding Target (Crime Type)
    le = LabelEncoder()
    if 'Incident Type Primary' not in df.columns:
        print("Error: Target column 'Incident Type Primary' missing.")
        return None
        
    y = le.fit_transform(df['Incident Type Primary'])
    num_classes = len(le.classes_)
    print(f"Detected {num_classes} crime categories: {le.classes_}")

    # 4. Encoding Features
    # One-Hot Encode Day of Week
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='day')
    
    # Normalize Numerical Features (Lat, Long, Hour)
    scaler = MinMaxScaler()
    numerical_cols = ['Latitude', 'Longitude', 'hour']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Combine all features into X
    X = pd.concat([df[numerical_cols], day_dummies], axis=1).values.astype(np.float32)
    y = y.astype(np.longlong) # PyTorch expects LongTensor for Class labels

    return X, y, num_classes, le.classes_

# --- PART 2: NEURAL NETWORK ARCHITECTURE ---
class CrimeNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CrimeNet, self).__init__()
        
        # Layer 1: Input -> 128 Neurons
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Layer 2: 128 -> 64 Neurons
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Layer 3: 64 -> 32 Neurons
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output Layer: 32 -> Number of Classes
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x) # No Softmax here; CrossEntropyLoss handles it

# --- PART 3: TRAINING LOOP ---
def train_model():
    # 1. Prepare Data
    data_path = 'Crime_Incidents_20250604.csv' # UPDATE THIS PATH IF NEEDED
    processed_data = load_and_process_data(data_path)
    
    if processed_data is None: return

    X, y, num_classes, class_names = processed_data
    
    # Split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=seed)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = CrimeNet(input_dim=X.shape[1], num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {epoch_acc:.2f}%")

    # --- PART 4: EVALUATION ---
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
    
    # Detailed Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(c) for c in class_names]))

if __name__ == "__main__":
    train_model()
