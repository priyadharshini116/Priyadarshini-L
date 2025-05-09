import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset (replace 'your_file.csv' with your actual file path)
data = pd.read_csv('speech_emotions.csv')

# Print the first few rows of the dataset to check
print(data.head())

# Visualizations
# Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x='gender', data=data)
plt.title('Gender Distribution')
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(data['age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# Prepare the data (basic preprocessing)
X = data.drop(columns=['gender', 'set_id', 'text', 'country'])  # Assuming these columns are not useful for prediction
y = data['gender'].map({'MALE': 0, 'FEMALE': 1})  # Convert gender to numerical labels

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build and compile the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix
val_predictions = model.predict(X_val)
val_predictions = (val_predictions > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, val_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
