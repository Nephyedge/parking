import streamlit as st
import cv2
import pickle
import numpy as np
import time
from keras.models import load_model

model = load_model('model_final.h5')
class_dictionary = {0: 'no_car', 1: 'car'}

with open('carposition.pkl', 'rb') as f:
    posList = pickle.load(f)

width, height = 130, 65

@st.cache_resource
def load_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        frames.append(img)
    cap.release()
    return frames

def checkParkingSpace(img):
    spaceCounter = 0
    imgCrops = []

    for pos in posList:
        x, y = pos
        imgCrop = img[y:y + height, x:x + width]
        imgResize = cv2.resize(imgCrop, (48, 48))
        imgNormalized = imgResize / 255.0
        imgCrops.append(imgNormalized)

    imgCrops = np.array(imgCrops)
    predictions = model.predict(imgCrops, verbose=0)

    for i, pos in enumerate(posList):
        x, y = pos
        inID = np.argmax(predictions[i])
        label = class_dictionary[inID]

        if label == 'no_car':
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            textColor = (0,0,0)
        else:
            color = (0, 0, 255)
            thickness = 2
            textColor = (255,255,255)

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        font_scale = 0.5
        text_thickness = 1
        
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        textX = x
        textY = y + height - 5
        cv2.rectangle(img, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 6, textY + 2), color, -1)
        cv2.putText(img, label, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, textColor, text_thickness)

    totalSpaces = len(posList)
    return img, spaceCounter, totalSpaces - spaceCounter

st.title("Parking Space Detection")

frames = load_video_frames('car_test.mp4')
total_frames = len(frames)

if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False

col1, col2 = st.columns([6,1])
with col1:
    frame_idx = st.slider("Frame", 0, total_frames-1, st.session_state.frame_idx, 1)
with col2:
    if st.button("Play" if not st.session_state.playing else "Pause"):
        st.session_state.playing = not st.session_state.playing

# Always update frame index from slider
st.session_state.frame_idx = frame_idx

# Show the current frame and stats
img = frames[st.session_state.frame_idx].copy()
img, free_spaces, occupied_spaces = checkParkingSpace(img)
st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
st.metric("Free Spaces", free_spaces)
st.metric("Occupied Spaces", occupied_spaces)

# If playing, auto-advance the frame index
if st.session_state.playing:
    if st.session_state.frame_idx < total_frames - 1:
        time.sleep(0.07)  # ~14 FPS, adjust as needed
        st.session_state.frame_idx += 1
        st.rerun()
    else:
        st.session_state.playing = False