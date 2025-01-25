import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize variables
coffee_count = 0
last_gesture_time = 0
gesture_cooldown = 3  # Cooldown in seconds to avoid multiple detections
drinking_detected = False
last_drinking_time = 0
drinking_cooldown = 5  # Cooldown for drinking detection

def count_fingers(hand_landmarks):
    """Count extended fingers (excluding thumb)"""
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tip landmarks
    finger_count = 0
    
    # Check if fingers are extended by comparing y coordinates
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1
            
    return finger_count

def calculate_water_needed(coffee_count):
    """Calculate recommended water intake in ml"""
    # Recommend 2 cups (500ml) of water per cup of coffee
    return coffee_count * 500

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check for finger count gesture
            current_time = time.time()
            if current_time - last_gesture_time > gesture_cooldown:
                fingers = count_fingers(hand_landmarks)
                if 1 <= fingers <= 5:  # Valid coffee count range
                    coffee_count = fingers
                    last_gesture_time = current_time
                    
            # Check for drinking gesture (hand near mouth)
            nose_y = hand_landmarks.landmark[9].y  # Middle of palm
            if (hand_landmarks.landmark[4].y < nose_y and  # Thumb position
                current_time - last_drinking_time > drinking_cooldown):
                drinking_detected = True
                last_drinking_time = current_time

    # Display information on screen
    water_needed = calculate_water_needed(coffee_count)
    cv2.putText(image, f'Coffee count: {coffee_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Water needed: {water_needed}ml', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if drinking_detected:
        cv2.putText(image, 'Drinking detected!', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        drinking_detected = False

    # Display the image
    cv2.imshow('Coffee Tracker', image)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
