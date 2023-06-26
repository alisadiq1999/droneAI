from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import time
from ultralytics import YOLO
import pafy
from threading import Thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# cap = cv2.VideoCapture(0)  # use 0 for web camera

# ------ Uncomment below and add youtube video link if you want the stream from youtube ---- #
# url = "https://www.youtube.com/watch?v=01guY9qxzWk"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# url = best.url

# ------ RMTP Stream Links --------- #
# url = 'rtmp://46.151.211.6:1935/camera1?Username:root&Password:n#mR6NZ*%N9zus77'  # Camera 1
# url ='rtmp://46.151.211.6:1935/camera2?Username:root&Password:n#mR6NZ*%N9zus77===' # Camera 2

# ------ Camera recordings from drone ---------- #
# url = "./DJI_20230624173006_0001_Z.MP4" # Recording 1
# url = "./DJI_20230624173006_0002_Z.MP4" # Recording 2
url = "./DJI_20230625080645_0001_W copy.mov"  # Day2 - recording 1
# url = "./DJI_20230625080645_0002_W copy.mov"  # Day2 - recording 2
# url = "DJI_20230625092957_0001_W.mov" # Day 2 - recording 3

# ---------------------------------------------- #
# For faster processing, we skip 3 frames, change as needed.
frames_to_skip = 3

# ----------------------------------------------- #
cap = cv2.VideoCapture(url)

final_frame_count = {'person': 0, 'bicycle': 0, 'car': 0, 'bike': 0, 'airplane': 0, 'bus': 0}

# Load a model
model = YOLO("./drone_model.pt")


def main():
    time.sleep(5)
    classes = ["person", "bicycle", "car", "bike", "airplane", "bus"]
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    counter = 0

    # Read video from camera
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, live_frame = cap.read()
        if ret:
            counter += 1

            if counter % frames_to_skip == 0:

                res = model.predict(source=live_frame, imgsz=1920, classes=[0, 2, 3, 5], agnostic_nms=True)

                boxes = res[0].boxes
                class_count = [0, 0, 0, 0, 0, 0]
                for box in boxes:
                    class_id = int(box.cls[0].tolist())
                    class_count[class_id] += 1
                    # print("Detected Class: ", classes[class_id])

                global final_frame_count

                final_frame_count = dict(zip(classes, class_count))
                print("Final Count of Frame: ", final_frame_count)
                res_plotted = res[0].plot()

                ret, buffer = cv2.imencode('.jpg', res_plotted)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')


def update_variable():
    global final_frame_count
    # global variable
    while True:
        socketio.emit('variable_update', {'variable': final_frame_count}, namespace='/')
        time.sleep(1)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('dashboard.html')


# Event handler for client connection
@socketio.on('connect', namespace='/')
def handle_connect():
    # Send the initial variable value to the connected client
    emit('variable_update', {'variable': final_frame_count})


# Start the variable update thread
update_thread = Thread(target=update_variable)
update_thread.start()

if __name__ == '__main__':
    socketio.run(app, port=8000, debug=True)