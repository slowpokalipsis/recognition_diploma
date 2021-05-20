import face_recognition
from datetime import datetime
import sqlite3
import numpy as np
import pickle
import cv2
import time

t1 = time.time()
conn = sqlite3.connect("db.sqlite3")
cursor = conn.cursor()
cursor.execute("SELECT * FROM FRS")
facecount = cursor.fetchall()
cursor.close()
conn.close()

known_face_encodings =[]
known_face_names = []
known_face_uids = []

for i in range(len(facecount)):
    a = pickle.loads(facecount[i][4])
    known_face_encodings.append(a)
    known_face_names.append(facecount[i][1])
    known_face_uids.append(facecount[i][0])
t2 = time.time()
print(t2-t1)

face_names = []
process_this_frame = 0

print('webcam getting ready')
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 800)
video_capture.set(4, 448)

# vid_cod = cv2.VideoWriter_fourcc(*'MJPG')
# output = cv2.VideoWriter("platon.avi", vid_cod, 10.0, (1366,768))
# capture_duration = 5

print(video_capture.isOpened())
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 448)

prev_frame_time = 0
new_frame_time = 0

while True:
    start_frame = time.time()
    try:
        divideint = 2
        # Grab a single frame of video
        ret, frame = video_capture.read()
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.flip(frame, 1)
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / divideint, fy=1 / divideint)
        

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # Only process every other frame of video to save time
        if process_this_frame == 0:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                #                 if face_distances is None or face_distances.shape[0]==0:
                #                     break
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    uid = known_face_uids[best_match_index]

                face_names.append(name)

        process_this_frame += 1
        if process_this_frame == 3:
            process_this_frame = 0
            
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= divideint
            right *= divideint
            bottom *= divideint
            left *= divideint

            if name != "Unknown":
                #                 cursor = col.find({'uid': f"""{uid}"""})
                usr_time = str(datetime.now())[:-7]
                #                 if usr_time not in cursor[0]['time']:
                #                     col.update_one({'uid': uid}, {'$push': {'time': usr_time}})
            #

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (133, 133, 133), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 23), (right, bottom), (133, 133, 133), cv2.FILLED)
            font = cv2.FONT_ITALIC
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (0, 0, 0), 1)

        font_fps = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, str(fps), (7, 30), font_fps, 1, (100, 255, 0))

        # Display the resulting image
        #         if abctemp > 10:
        cv2.imshow('Video', frame)
        #         abctemp += 1

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break

    except Exception as e:
        print(e)
        print('keke')
        video_capture.release()
        cv2.destroyAllWindows()
        break

video_capture.release()
cv2.destroyAllWindows()



