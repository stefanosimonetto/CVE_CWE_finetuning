import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import Callback
import joblib

class F1ScoreCallback(Callback):
    def __init__(self, X_val, y_val):
        super(F1ScoreCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = 0.0
        self.best_model = None
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_val_pred = np.argmax(self.model.predict(self.X_val), axis=1)
        f1 = f1_score(self.y_val, y_val_pred, average='weighted')
        self.f1_scores.append(f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_model = self.model
            print(f"Epoch {epoch + 1} - F1 Score: {f1:.4f}")
            print("Saved best model")
            print(self.f1_scores)

with open('/home/simonettos/comparison/phi-2/phi_train.pickle', 'rb') as f1:
    balanced = pickle.load(f1)

with open('/home/simonettos/comparison/phi-2/phi_test_0.pickle', 'rb') as f2:
    unbalanced = pickle.load(f2)

train = np.array([item['processed_description_phi_last'] for item in balanced if item['cwe'] != 'None'])
test = np.array([item['cwe'] for item in balanced if item['cwe'] != 'None'])

np.random.seed(42)
X_train, X_val, y_train, y_val = train_test_split(train, test, test_size=0.1, random_state=42)

X_test = np.array([item['processed_description_phi_last'] for item in unbalanced if item['cwe'] != 'None'])
y_test = np.array([item['cwe'] for item in unbalanced if item['cwe'] != 'None'])

label_encoder_train = LabelEncoder()
y_train_encoded = label_encoder_train.fit_transform(y_train)
label_encoder_test = LabelEncoder()
y_test_encoded = label_encoder_test.fit_transform(y_test)

input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))

def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model 3 times
models = []
for i in range(3):
    model = create_model(input_dim, output_dim)
    f1_callback = F1ScoreCallback(X_val, label_encoder_train.transform(y_val))
    model.fit(X_train, y_train_encoded, epochs=40, batch_size=64, 
              validation_data=(X_val, label_encoder_train.transform(y_val)), 
              verbose=1, callbacks=[f1_callback])
    models.append(f1_callback.best_model)

# Initialize a variable to accumulate predictions for averaging
predictions = np.zeros((X_test.shape[0], output_dim))

# Evaluate each model and print its classification report
for i, model in enumerate(models):
    # Make predictions on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_pred_original = label_encoder_train.inverse_transform(y_pred)

    # Print the classification report for the current model
    print(f"Classification Report for Model {i+1}:\n", classification_report(y_test, y_pred_original, digits=4))

    # Accumulate predictions for averaging
    predictions += y_pred_probs

    # Save the model in .h5 format
    model.save(f'/home/simonettos/comparison/phi-2/paper/nlp/best_model_descr_{i+1}.h5')

# Average the accumulated predictions
predictions /= len(models)

# Convert the averaged predictions to class labels
y_pred_avg = np.argmax(predictions, axis=1)
y_pred_avg_original = label_encoder_train.inverse_transform(y_pred_avg)

# Print the averaged classification report
print("\nAveraged Classification Report (Phi-2_nlp):\n", classification_report(y_test, y_pred_avg_original, digits=4))

# Save the label encoder
joblib.dump(label_encoder_train, '/home/simonettos/comparison/phi-2/paper/nlp/label_encoder_train_descr.joblib')