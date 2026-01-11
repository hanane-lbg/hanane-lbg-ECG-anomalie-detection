import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Bidirectional, Conv1D, MaxPooling1D
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import time
from feature_extraction import X_train_fe, X_test_fe, y_train, y_test

def create_model(model_type, input_shape):
    """
    Create and return a Keras model based on the type.
    
    Parameters:
    - model_type: 'lstm', 'bi_lstm', 'stacked_lstm', 'gru', 'cnn_lstm'
    - input_shape: tuple, shape of input (timesteps, features)
    """
    model = Sequential()
    
    if model_type == 'lstm':
        model.add(LSTM(64, input_shape=input_shape))
    elif model_type == 'bi_lstm':
        model.add(Bidirectional(LSTM(64), input_shape=input_shape))
    elif model_type == 'stacked_lstm':
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64))
    elif model_type == 'gru':
        model.add(GRU(64, input_shape=input_shape))
    elif model_type == 'cnn_lstm':
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(64))
    else:
        raise ValueError("Invalid model_type")
   
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


#Reshape features for sequence models
def reshape_features(X):
    X = np.asarray(X).astype('float32')
    return X.reshape(X.shape[0], X.shape[1], 1)

X_train_fe_reshaped = reshape_features(X_train_fe)
X_test_fe_reshaped  = reshape_features(X_test_fe)


#Function to create models
def create_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(64, input_shape=input_shape))
    elif model_type == 'bi_lstm':
        model.add(Bidirectional(LSTM(64), input_shape=input_shape))
    elif model_type == 'stacked_lstm':
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64))
    elif model_type == 'gru':
        model.add(GRU(64, input_shape=input_shape))
    elif model_type == 'cnn_lstm':
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(64))
    else:
        raise ValueError("Invalid model type")
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train and evaluate all models
model_types = ['lstm', 'bi_lstm', 'stacked_lstm', 'gru', 'cnn_lstm']
histories = {}
results = {}

for m_type in model_types:
    print(f"\nTraining model: {m_type.upper()}")
    model = create_model(m_type, input_shape=(X_train_fe_reshaped.shape[1], 1))
    
    start_time = time.time()
    history = model.fit(
        X_train_fe_reshaped, y_train,
        validation_data=(X_test_fe_reshaped, y_test),
        epochs=50, batch_size=32, verbose=1
    )
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    histories[m_type] = history
    
    # Predictions
    y_pred = (model.predict(X_test_fe_reshaped) > 0.5).astype(int)
    
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[m_type] = {'model': model, 'accuracy': acc, 'auc': auc, 'report': report, 'confusion_matrix': cm}
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"AUC: {auc:.2f}")
    print(report)
    
    # Confusion matrix visualization
    cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'],
                         index=['Actual Negative', 'Actual Positive'])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
    fig.update_xaxes(side='top', title_text='Predicted')
    fig.update_yaxes(title_text='Actual')
    fig.show()

#Plot training history

def plot_history(history, model_name):
    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

for m_type in model_types:
    plot_history(histories[m_type], m_type.upper())


