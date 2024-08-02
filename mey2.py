import cv2
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# إعداد مجلد لحفظ الصور والفيديوهات
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# تحميل نموذج التعرف على الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# بدء التقاط الفيديو من الكاميرا
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# إعداد واجهة رسومية باستخدام Tkinter
def start_recording():
    global out
    if out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = cv2.VideoWriter(os.path.join(output_dir, f"video_{timestamp}.avi"), fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = frame[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(output_dir, f"face_{timestamp}.jpg"), face_img)

        out.write(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def stop_recording():
    global out
    if out is not None:
        out.release()
        out = None
        messagebox.showinfo("Info", "Recording stopped and saved successfully.")

# إعداد نافذة Tkinter
root = tk.Tk()
root.title("Face Recognition System")

start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack()

stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.pack()

root.mainloop()
