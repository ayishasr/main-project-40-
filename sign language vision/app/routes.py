from flask import Flask, request, jsonify, render_template, send_file, make_response
from googletrans import Translator  # Google Translate API for language translation
from gtts import gTTS  # Google Text-to-Speech API for text-to-speech conversion
from app import app  # Import the app instance
import os
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="app/sign_language_model.tflite")
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize translator and TTS engine
translator = Translator() 
scaler = joblib.load("app/scaler.pkl")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play_audio', methods=['POST'])
def play_audio():
    data = request.json
    selected_language = data['language']
    input_features = data['input_features']
    labels = ['A', 'B','BAD', 'C', 'D', 'DEAF', 'E', 'F', 'FINE', 'G', 'GOOD', 'GOODBYE', 'H', 'HELLO', 'HUNGRY', 'I', 'J', 'K', 'L', 'M', 'ME', 'N', 'NO', 'O', 'P', 'PLEASE', 'Q', 'R', 'S', 'SORRY', 'T', 'THANK YOU', 'U', 'V', 'W', 'X', 'Y', 'YES', 'YOU', 'Z']

    number_of_features = 21 
    sequence_length = 150 

    # Split the input string, convert to floats, and reshape to match model input
    input_data = np.array([float(value) for value in input_features.split(",")], dtype=np.float32).reshape(-1, number_of_features)
    if input_data.size != sequence_length * number_of_features:
        raise ValueError("Incorrect input length. Expected length is 150 * number_of_features.")
    
    input_data = scaler.transform(input_data)

    input_data = input_data.reshape(1, sequence_length, number_of_features)
    
    print(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get model output and format it as probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])
    gesture = np.argmax(output_data[0])
    predicted_label = labels[gesture]

    recognized_text = predicted_label
    # Translate the recognized text to the selected language
    translated_text = translator.translate(recognized_text, dest=selected_language).text
    # Use gTTS to create an audio file
    tts = gTTS(text=translated_text, lang=selected_language)
    # audio_file_path = f"app/output_{count}.mp3"
    audio_file_path = "app/output.mp3"

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
    
    tts.save(audio_file_path)
    

    print(f'Audio file saved to {audio_file_path}')

    return jsonify({'translated_text': translated_text})

@app.route('/get_audio')
def get_audio():
    print("Sending audio file output.mp3")
    response = send_file("output.mp3", mimetype="audio/mpeg")

    return response


