import cv2
import time
import imutils
import smtplib
from decouple import config
from imutils.video import VideoStream

EMAIL = config('EMAIL')
PASSWORD = config('PASSWORD')
RECEIVER_EMAIL = config('RECEIVER_EMAIL')
SMTP_SERVER = "smtp.gmail.com"
MESSAGE = """\
    Subject: DETECTED MOVEMENT

    Connected cam detected movement."""


def login_to_gmail():
    server = smtplib.SMTP_SSL(SMTP_SERVER)
    server.login(EMAIL, PASSWORD)

    return server


if __name__ == '__main__':
    email_server = login_to_gmail()

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    first_frame = None
    prev_time = time.time()

    while True:
        curr_frame = vs.read()

        if curr_frame is None:
            break

        curr_frame = imutils.resize(curr_frame, width=500)
        grayed_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        grayed_frame = cv2.GaussianBlur(grayed_frame, (21, 21), 0)

        if first_frame is None:
            first_frame = grayed_frame
            continue

        frame_delta = cv2.absdiff(first_frame, grayed_frame)

        thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if time.time() - prev_time > 59:
                prev_time = time.time()
                email_server.sendmail(EMAIL, RECEIVER_EMAIL, MESSAGE)

        cv2.imshow("Security Feed", curr_frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frame_delta)
        key_listener = cv2.waitKey(1) & 0xFF

        if key_listener == ord("q"):
            break

    email_server.quit()
    vs.stop()
    cv2.destroyAllWindows()
