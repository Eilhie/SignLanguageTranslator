import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained CNN model
model = load_model('D:\Developer\SignLanguageComputerVision\model\CNN.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Preprocess the frame as needed for model input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    input_data = np.expand_dims(resized, axis=0)  # Add batch dimension

    # Make predictions using the model
    prediction = model.predict(input_data)
    predicted_letter = np.argmax(prediction)  # Get the index of the most probable letter

    # Display the prediction on the frame
    cv2.putText(frame, chr(65 + predicted_letter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break


# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
