import cv2

def save_rtmp_stream(url, output_file):
    # Create a VideoCapture object to read the RTMP stream
    cap = cv2.VideoCapture(url)

    # Check if the VideoCapture object was successfully initialized
    if not cap.isOpened():
        print("Failed to open RTMP stream")
        return

    # Get the video codec and create a VideoWriter object to save the stream
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    output = cv2.VideoWriter(output_file, codec, 30.0, (1280, 960))

    while True:
        # Read frames from the RTMP stream
        ret, frame = cap.read()

        if ret:
            # Display the frame (optional)
            cv2.imshow("RTMP Stream", frame)

            # Write the frame to the output file
            output.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    output.release()
    cv2.destroyAllWindows()

# Example usage
url = 'rtmp://46.151.211.6:1935/camera1?Username:root&Password:n#mR6NZ*%N9zus77'  # Replace with your RTMP stream URL
output_file = 'output.avi'  # Replace with the desired output file path
save_rtmp_stream(url, output_file)
