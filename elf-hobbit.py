import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# reads binaries in and preprocess them for fitting tensors
def read_and_preprocess_binaries(directory, label, max_length, verbose=False):
    data = []
    for root, _, files in os.walk(directory):
        if verbose:
            print(f"Reading and preprocessing {len(files)} files from {directory}...")
        for i, file in enumerate(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                byte_data = list(f.read())
            if len(byte_data) > max_length:
                byte_data = byte_data[:max_length]
            elif len(byte_data) < max_length:
                byte_data += [0] * (max_length - len(byte_data))
            data.append((file_path, torch.tensor(byte_data, dtype=torch.float32), label))  # Change here
            if verbose:
                print(f"File {i + 1}/{len(files)}: Processed with length {len(byte_data)}")
    return data

# define NN, using Feed Forward
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training loop
def train_model(train_loader, val_loader, input_size, num_epochs=15, learning_rate=0.001):
    model = FeedForwardNN(input_size, hidden_size=128, output_size=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    print("Training completed.")
    return model

# define the test model
def test_model(model, test_data):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for file_path, inputs, actual_label in test_data:
            inputs = inputs.unsqueeze(0).to(device)
            prediction = model(inputs)
            predicted_label = (torch.sigmoid(prediction) > 0.5).long().cpu().item()  # Ensure on CPU
            y_pred.append(predicted_label)
            y_true.append(actual_label.cpu().item() if isinstance(actual_label, torch.Tensor) else actual_label)
            label_str = "Malicious" if predicted_label else "Benign"
            print(f"File: {file_path}, Predicted Label: {label_str}")

    return y_true, y_pred

# Main body for train or testing
def main():
    global device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    max_input_size = 1024  # Example size, adjust based on your dataset

    train_malicious_dir = #Path
    train_benign_dir = #Path
    test_malicious_dir = #Path
    test_benign_dir = #Path

    mode = input("Choose mode (train/test): ").lower()

    if mode == 'train':
        train_malicious = read_and_preprocess_binaries(train_malicious_dir, 1, max_input_size, verbose=True)
        train_benign = read_and_preprocess_binaries(train_benign_dir, 0, max_input_size, verbose=True)
        train_data = train_malicious + train_benign

        train_features, train_labels = zip(*train_data)  # This line should work now
        train_features = torch.stack(train_features)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = FeedForwardNN(max_input_size, hidden_size=128, output_size=1).to(device)
        trained_model = train_model(train_loader, None, max_input_size, num_epochs=15, learning_rate=0.001)
        torch.save(trained_model.state_dict(), 'trained_model.pth')
        print("Model trained and saved successfully.")

    elif mode == 'test':
        if not os.path.exists('trained_model.pth'):
            print("Model file not found. Please train the model first.")
            return

        model = FeedForwardNN(max_input_size, hidden_size=128, output_size=1).to(device)
        model.load_state_dict(torch.load('trained_model.pth', map_location=device))

        test_malicious = read_and_preprocess_binaries(test_malicious_dir, 1, max_input_size, verbose=False)
        test_benign = read_and_preprocess_binaries(test_benign_dir, 0, max_input_size, verbose=False)
        test_data = test_malicious + test_benign

        y_true, y_pred = test_model(model, test_data)

        # Evaluate Accuracy
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")

if __name__ == "__main__":
    main()
