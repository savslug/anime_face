import os
import shutil
import cv2
from tqdm import tqdm
import glob


def video_2_frames(video_file='./IMG_2140.MOV', image_dir='./mov2image_dir/', image_file='img_%s.png', interval=300):
    # Delete the entire directory tree if it exists.
    # if os.path.exists(image_dir):
        # shutil.rmtree(image_dir)

    # Make the directory if it doesn't exist.
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Video to frames
    i = 0
    hist_last = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?
            break

        # コマの画像を取得

        # ヒストグラムを取得
        hist_now = cv2.calcHist([frame], [0], None, [256], [0, 256])
        # print(hist)
        try:
            dist = cv2.compareHist(hist_now, hist_last, 0)
        except Exception:
            dist = 0
        # print(dist)
        if(abs(dist) < 0.5):
            cv2.imwrite(image_dir + image_file %
                        str(i).zfill(6), frame)  # Save a frame
            print('Save', image_dir + image_file % str(i).zfill(6))

        # 直前のHISTを記録して次のインターバルへ
        hist_last = hist_now
        i += interval

    cap.release()  # When everything done, release the capture


for i in tqdm(glob.glob('D:/Downloads/kf' + '/*')):
    print(i)
    video_2_frames(i, interval=300)

#[Ohys-Raws] Kemono Friends (TX 1280x720 x264 AAC)
