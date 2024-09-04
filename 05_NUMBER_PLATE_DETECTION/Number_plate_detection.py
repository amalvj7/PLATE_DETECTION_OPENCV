import cv2

####################################
width = 640
height = 480
brightness  = 100
# Update the path to the Haar Cascade XML file
num_plates_dect_lib = cv2.CascadeClassifier('C:/Users/z0050b2z/Downloads/OPEN_CV/05_NUMBER_PLATE_DETECTION/haarcascades/haarcascade_russian_plate_number.xml')
mini_area  = 200
count  = 0
############################

web_can_video = cv2.VideoCapture(0)
web_can_video.set(3, width)
web_can_video.set(4, height)
web_can_video.set(10, brightness)

while True:
    ret, frames = web_can_video.read()
    grey_frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    num_plates = num_plates_dect_lib.detectMultiScale(grey_frames, 1.1, 10)

    for x, y, w, h in num_plates:
        area = w * h
        if area > mini_area:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frames, 'Num_plate', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
            frames_ROI = frames[y: y + h, x: x + w]
            cv2.imshow('window_only_num_plates', frames_ROI)

    cv2.imshow('web_cam_window', frames)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Saved_number_plates/NoPlate_' + str(count) + '.jpg', frames_ROI)
        cv2.rectangle(frames, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(frames, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", frames)
        cv2.waitKey(500)
        count += 1

web_can_video.release()
cv2.destroyAllWindows()
