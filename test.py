def extract_faces(images_dir, export_dir='faces'):
    """
    与えられたパスの画像からアニメの顔を抽出
    facesディレクトリに顔部分を出力
    """
    import os
    import cv2
    import glob
    from tqdm import tqdm

    # 特徴量ファイルをもとに分類器を作成
    classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')

    # ディレクトリを作成
    output_dir = export_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in tqdm(glob.glob(images_dir + '/*')):
        # print(image_path)

        # 顔の検出
        image = cv2.imread(image_path)
        # グレースケールで処理を高速化
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_image)

        # print(faces)

        for i, (x, y, w, h) in enumerate(faces):
            # 一人ずつ顔を切り抜く
            face_image = image[y:y + h, x:x + w]
            output_path = os.path.join(
                output_dir, image_path[len(images_dir) + 1:] + '{0}.jpg'.format(i))
            print(output_path)
            cv2.imwrite(output_path, face_image)

        #cv2.imwrite('face.jpg', image)

    return

    for x, y, w, h in faces:
        # 四角を描く
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      color=(0, 0, 255), thickness=3)

    cv2.imwrite('faces.jpg', image)


extract_faces('mov2image_dir')
