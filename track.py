import numpy as np
import cv2
import time
import os

cascade_file = "haarcascade_fullbody.xml"
cascade = cv2.CascadeClassifier(cascade_file)
video = input('見たい動画を入力:')
cap = cv2.VideoCapture(os.path.join('./video/', video))


#Shi-Tomasi(コーナー検出)のコーナー検出のパラメータ検出したいコーナーの個数、
#次に検出するこーなの最低限の質を0~1
#最後に検出される2つのコーナー間の最低限のユークリッド距離を与える
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

#Lucas-Kanade法(opticalflowを推定)
lk_params = dict( winSize=(15,15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#ランダムに色を100個生成
color = np.random.randint(0, 255, (100, 3))

#最初のフレーム処理
end_flag, frame = cap.read()
#グレースケール変換
gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#追跡に向いた特徴(Shi-Tomasi)
feature_prev = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)
#元の配列と同じ形に
mask = np.zeros_like(frame)

while(end_flag):
    #グレースケールに変換
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start = time.time()

    #全身の人を検出
    #minSize:物体がとりうる最小サイズ（これ以下は無視）
    #minNeighbors:物体候補となる矩形は、最低でもこの数だけの近傍矩形を含む
    body = cascade.detectMultiScale(gray_next, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    end = time.time()

    #検出時間を表示
    print("{} : {:4.1f}ms".format("detectTime", (end -start)*1000))


    #opticalflow検出
    #status: 0:検出していない 1:検出
    feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)
    good_prev = feature_prev[status==1]
    good_next = feature_next[status==1]

    #opticalflowを描画
    #toList: Listに変換
    #img, start end color thickness
    for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
        prev_x, prev_y = prev_point.ravel()
        next_x, next_y = next_point.ravel()
        mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(), 2)
        frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    #比と検知数表示用
    human_cnt = 0

    #人を矩形で囲う
    for (x, y, w, h) in body:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        human_cnt += 1

    #人の数表示
    #cv2.putText(img, "Human Cnt:{}".format(int(human_cnt)), (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #ウィンドウに表示
    cv2.imshow('human_view', img)

    #ESCキー
    k = cv2.waitKey(1)
    if k == 27:
        break

    #次のフレーム、ポイントの準備
    gray_prev = gray_next.copy()
    feature_prev = good_next.reshape(-1, 1, 2)
    end_flag, frame = cap.read()

#終了処理
cv2.destroyAllWindows()
cap.release()
