import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        output = self.sigmoid(output)
        return output

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.data['input_ids'][index]
        attention_mask = self.data['attention_mask'][index]
        token_type_ids = self.data['token_type_ids'][index]
        sentiment_label = self.data['sentiment_label'][index]
        
        return input_ids, attention_mask, token_type_ids, sentiment_label
    
def plot_accuracy(epochs_range, train_accuracy_history, val_accuracy_history, parameters):
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_accuracy_history, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(parameters['result_path'], 'LSTM_Accuracy.png'))
def plot_loss(epochs_range, train_loss_history, val_loss_history, parameters):
    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(parameters['result_path'], 'LSTM_Loss.png'))

def train_validate_test_LSTM(tokenized_data, vocab_size,logger, parameters):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    vocab_size = vocab_size  # Assuming you are using the base model of BERT, adjust based on your tokenizer
    embed_size = 250
    hidden_size = 128
    output_size = 1
    num_epochs = int(parameters['epochs'])
    batch_size = int(parameters['batch_size'])
    learning_rate = float(parameters['learning_rate'])
    val_threshold = float(parameters['val_threshold'])
    test_threshold = float(parameters['test_threshold'])
    torch.manual_seed(parameters['seed'])
    
    # Initialize model, loss function, and optimizer
    model = LSTMModel(vocab_size, embed_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    dataset = CustomDataset(tokenized_data)
 
    
    # Split dataset into train, validation, and test sets
    dataset_size = len(tokenized_data['input_ids'])
    indices = list(range(dataset_size))
    split1 = int(np.floor(val_threshold * dataset_size))  # 80-20 train-validation split
    split2 = int(np.floor(test_threshold * dataset_size))  # 90-10 train-test split

    # Define samplers for each set
    train_sampler = SubsetRandomSampler(indices[:split1])
    val_sampler = SubsetRandomSampler(indices[split1:split2])
    test_sampler = SubsetRandomSampler(indices[split2:])

    # Prepare data loaders for each set
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Lists to store training statistics for plotting
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    # Training loop
    for epoch in tqdm(range(num_epochs), desc=f'Epochs'):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            input_ids, attention_mask, token_type_ids, labels = [tensor.to(device) for tensor in batch]

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_ids, attention_mask, token_type_ids)
            # Calculate the loss
            loss = criterion(outputs.squeeze(), labels.float())  # Squeeze to remove extra dimensions
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.round(outputs).squeeze()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
                  
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        # Save training statistics for plotting
        train_loss_history.append(avg_loss)
        train_accuracy_history.append(accuracy)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f'Validation', leave=False):
                val_input_ids, val_attention_mask, val_token_type_ids, val_labels = [tensor.to(device) for tensor in val_batch]

                # Forward pass (no backward pass during validation)
                val_outputs = model(val_input_ids, val_attention_mask, val_token_type_ids)
                # Calculate the validation loss
                val_loss = criterion(val_outputs.squeeze(), val_labels.float())
                total_val_loss += val_loss.item()

                # Calculate validation accuracy
                val_predicted = torch.round(val_outputs).squeeze()
                correct_val_predictions += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_predictions / total_val_samples
        # Save validation statistics for plotting
        val_loss_history.append(avg_val_loss)
        val_accuracy_history.append(val_accuracy)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            # Optionally, you can also save the best model checkpoint to a file
            torch.save(model.state_dict(), os.path.join(parameters['model_path'],'best_model.pth'))
            
            
    # Load the best model state
    model.load_state_dict(best_model_state)

    # Testing loop
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0.0
    correct_test_predictions = 0
    total_test_samples = 0
    with torch.no_grad():
        for test_batch in tqdm(test_loader, desc=f'Testing'):
            test_input_ids, test_attention_mask, test_token_type_ids, test_labels = [tensor.to(device) for tensor in test_batch]

            # Forward pass (no backward pass during validation)
            test_outputs = model(test_input_ids, test_attention_mask, test_token_type_ids)
            # Calculate the validation loss
            test_loss = criterion(test_outputs.squeeze(), test_labels.float())
            total_test_loss += test_loss.item()

            # Calculate validation accuracy
            test_predicted = torch.round(test_outputs).squeeze()
            correct_test_predictions += (test_predicted == test_labels).sum().item()
            total_test_samples += test_labels.size(0)
            
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = correct_test_predictions / total_test_samples
    logger.info(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

    # Plotting
    epochs_range = range(1, num_epochs + 1)
    plot_accuracy(epochs_range, train_accuracy_history, val_accuracy_history, parameters)
    plot_loss(epochs_range, train_loss_history, val_loss_history, parameters)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from torch.utils.data.sampler import SubsetRandomSampler
# from tqdm import tqdm

# # Define LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_ids, attention_mask, token_type_ids):
#         embedded = self.embedding(input_ids)
#         lstm_out, _ = self.lstm(embedded)
#         output = self.fc(lstm_out[:, -1, :])
#         output = self.sigmoid(output)
#         return output

# # Define custom dataset
# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         input_ids = self.data['input_ids'][index]
#         attention_mask = self.data['attention_mask'][index]
#         token_type_ids = self.data['token_type_ids'][index]
#         sentiment_label = self.data['sentiment_label'][index]
        
#         return input_ids, attention_mask, token_type_ids, sentiment_label

# def test_and_evaluate_LSTM(tokenized_data, vocab_size,logger, parameters):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Hyperparameters
#     vocab_size = vocab_size  # Assuming you are using the base model of BERT, adjust based on your tokenizer
#     embed_size = 128
#     hidden_size = 128
#     output_size = 1
#     num_epochs = int(parameters['epochs'])
#     batch_size = int(parameters['batch_size'])
#     learning_rate = float(parameters['learning_rate'])
#     val_threshold = float(parameters['val_threshold'])
#     test_threshold = float(parameters['test_threshold'])
#     torch.manual_seed(parameters['seed'])
    
#     # Initialize model, loss function, and optimizer
#     model = LSTMModel(vocab_size, embed_size, hidden_size, output_size)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     dataset = CustomDataset(tokenized_data)
 
    
#     # Split dataset into train, validation, and test sets
#     dataset_size = len(tokenized_data['input_ids'])
#     indices = list(range(dataset_size))
#     split1 = int(np.floor(val_threshold * dataset_size))  # 80-20 train-validation split
#     split2 = int(np.floor(test_threshold * dataset_size))  # 90-10 train-test split

#     # Define samplers for each set
#     train_sampler = SubsetRandomSampler(indices[:split1])
#     val_sampler = SubsetRandomSampler(indices[split1:split2])
#     test_sampler = SubsetRandomSampler(indices[split2:])

#     # Prepare data loaders for each set
#     train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#     val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
#     test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    
#     best_val_loss = float('inf')
#     best_model_state = None

#     # Training loop
#     for epoch in tqdm(range(num_epochs), desc=f'Epochs'):
#         model.train()  # Set the model to training mode
#         total_loss = 0.0
#         correct_predictions = 0
#         total_samples = 0

#         for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
#             input_ids = batch[0]
#             attention_mask = batch[1]
#             token_type_ids = batch[2]
#             labels = batch[3]

#             # Zero the gradients
#             optimizer.zero_grad()
#             # Forward pass
#             outputs = model(input_ids, attention_mask, token_type_ids)
#             # Calculate the loss
#             loss = criterion(outputs.squeeze(), labels.float())  # Squeeze to remove extra dimensions
#             # Backward pass
#             loss.backward()
#             # Update weights
#             optimizer.step()
#             total_loss += loss.item()
            
#             # Calculate accuracy
#             predicted = torch.round(outputs).squeeze()
#             correct_predictions += (predicted == labels).sum().item()
#             total_samples += labels.size(0)
                  
#         avg_loss = total_loss / len(train_loader)
#         accuracy = correct_predictions / total_samples
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

        
#         # Validation loop
#         model.eval()  # Set the model to evaluation mode
#         total_val_loss = 0.0
#         correct_val_predictions = 0
#         total_val_samples = 0

#         with torch.no_grad():
#             for val_batch in tqdm(val_loader, desc=f'Validation', leave=False):
#                 val_input_ids = val_batch[0]
#                 val_attention_mask = val_batch[1]
#                 val_token_type_ids = val_batch[2]
#                 val_labels = val_batch[3]

#                 # Forward pass (no backward pass during validation)
#                 val_outputs = model(val_input_ids, val_attention_mask, val_token_type_ids)
#                 # Calculate the validation loss
#                 val_loss = criterion(val_outputs.squeeze(), val_labels.float())
#                 total_val_loss += val_loss.item()

#                 # Calculate validation accuracy
#                 val_predicted = torch.round(val_outputs).squeeze()
#                 correct_val_predictions += (val_predicted == val_labels).sum().item()
#                 total_val_samples += val_labels.size(0)

#         avg_val_loss = total_val_loss / len(val_loader)
#         val_accuracy = correct_val_predictions / total_val_samples
#         print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

#         # Save the best model based on validation loss
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_state = model.state_dict().copy()
#             # Optionally, you can also save the best model checkpoint to a file
#             torch.save(model.state_dict(), 'best_model.pth')
            
            
#     # Load the best model state
#     model.load_state_dict(best_model_state)

#     # Testing loop
#     model.eval()  # Set the model to evaluation mode
#     total_test_loss = 0.0
#     correct_test_predictions = 0
#     total_test_samples = 0
#     with torch.no_grad():
#         for test_batch in tqdm(test_loader, desc=f'Testing'):
#             test_input_ids = test_batch[0]
#             test_attention_mask = test_batch[1]
#             test_token_type_ids = test_batch[2]
#             test_labels = test_batch[3]

#             # Forward pass (no backward pass during validation)
#             test_outputs = model(test_input_ids, test_attention_mask, test_token_type_ids)
#             # Calculate the validation loss
#             test_loss = criterion(test_outputs.squeeze(), test_labels.float())
#             total_val_loss += test_loss.item()

#             # Calculate validation accuracy
#             test_predicted = torch.round(test_outputs).squeeze()
#             correct_test_predictions += (test_predicted == test_labels).sum().item()
#             total_test_samples += test_labels.size(0)
            
#     avg_test_loss = total_test_loss / len(test_loader)
#     test_accuracy = correct_test_predictions / total_test_samples
#     print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
