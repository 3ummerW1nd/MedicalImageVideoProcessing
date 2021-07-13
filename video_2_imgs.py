import cv2 as cv

video = "video/original_video.mp4"
images_path = "original_image"
video_capture = cv.VideoCapture(video)
success, image = video_capture.read()
count = 0
while success:
    success, image = video_capture.read()
    count += 1
    if (count < 10):
        cv.imencode('.png', image)[1].tofile(images_path + "/image0%d.png" % count)
    else:
        cv.imencode('.png', image)[1].tofile(images_path + "/image%d.png" % count)
