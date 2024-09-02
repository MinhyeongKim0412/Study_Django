#%%
# 라이브러리 주입
import numpy as np
import matplotlib.pyplot as plt
import cv2 # 필요 시 설치 pip install opencv-python
from glob import glob
path = "C:/Image/train/"

# 이미지 객체로 담아주기
img1 = plt.imread('C:/Image/train\\cat.1.jpg') ## numpy array
img2 = plt.imread('C:/Image/train\\cat.2.jpg')
# 이미지 크기 조절 (일원화)
resized_img = cv2.resize(img1,(200,200))
# %%

# 폴더에 어떤 파일들이 들어있는지 출력
glob("C:/Image/train/*")
# 이미지 리스트 만들기
img_list = glob(path + "*")
# 이미지 몇 개 있는지 확인
len(img_list)
# %%

# 사진 크기 확인 ---
print(img1.shape)
print(img2.shape)
print(resized_img.shape)
# %%

# matplot으로 출력
plt.imshow(img1)
plt.imshow(img2)
plt.imshow(resized_img)
# %%

# 다양한 크기로 리사이징해보기
# X = np.zeros((4000,200,200,3))
# X = np.zeros((4000,300,300,3))
X = np.zeros((4000,400,400,3))
# %%

#데이터 읽어들이기
# X = np.zeros((1,400,400,3)) # 하나짜리만 가져와서 400*400 리사이징
# for i in img_list: # 3개만 읽어오려면 [:3]
#     img = cv2.resize(plt.imread(i),(400,400)).reshape(1,400,400,3)
#     if 'cat' in i:
#         print('cat')
#     else:
#         print('dog')
#     X = np.r_[X,img] # row

X = np.zeros((4000,400,400,3)) # 하나짜리만 가져와서 400*400 리사이징
Y = np.zeros(4000)

for idx,val in enumerate(img_list):
    img = cv2.resize(plt.imread(val),(400,400))[:,:,:3].reshape(1,400,400,3)
    X[idx] = img
    
    if 'cat' in val:
        Y[idx]=0
    else:
        Y[idx]=1
# %%
X.shape
# %%
X = X[1:]
# %%
plt.imshow(X[0].astype(np.uint8))
# %%
plt.imshow(X[1].astype(np.uint8))
# %%
plt.imshow(X[2].astype(np.uint8))
# %%
#데이터가 잘 들어왔는지 테스트
plt.imshow(X[2001].astype(np.uint8))
# %%
# dog 라면 1.0이 나올것
print(Y[2001])
plt.imshow(X[2001].astype(np.uint8))
plt.show()
# %%
# cat은 0.0
print(Y[1001])
plt.imshow(X[1001].astype(np.uint8))
plt.show()

# ------------------------------------------------------------------------
# %%
# 데이터를 섞어서 테스트하기 (DL활용), 학습 시키기
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten,Input

from sklearn.model_selection import train_test_split
# %%
# 데이터 정리
# 데이터의 크기 확인
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# 크기가 맞는지 확인한 후 train_test_split을 다시 시도합니다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# X_train = X_train/255.
# X_test = X_test/255.
# %%

# 레이어 설계
model = Sequential()
model.add(Input(shape=(400,400,3)))

#3개짜리를 8개로 늘려보기, 3,3필터
model.add(Conv2D(8,(3,3),activation='relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

#16개로 늘리기, 5,5필터
model.add(Conv2D(16,(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(3,3))

#64개로 늘리기
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(5,5))

#128개로 늘리기, maxpooling 안 태우기, Flatten  사용
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())

#Dense Layer 사용

# 128개 사용
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
#32개 사용
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
#32개 사용
model.add(Dense(1,activation='sigmoid'))

# %%
# 잘 처리됐는지 확인작업
# model.predict(X_train[[1]]).shape
# model.predict(X_train[[1]])[0][:,:,:3].shape
model.predict(X_train[[1]])[0][:,:,:3].astype(np.uint8)
# %%
# 레이어 추가 - BatchNormalization
model.predict(X[[0]]).shape
# %%
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# %%
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
# %%
# 학습시키기
model.fit(X_train,Y_train,epochs=10,batch_size=64,validation_data=(X_test,Y_test))
# %%
plt.imshow(X_train[1000].astype(np.uint8))
# plt.imshow(X_test[10].astype(np.uint8))
# %%
np.where(model.predict(X[[1000]]) > 0.5, "dog","cat")[0][0]
# %%
data = plt.imread("C:/www1/kwlee/static/Cat_November_2010-1a.jpg")
# %%
data = cv2.resize(data,(400,400))
# %%
np.where(model.predict(data.reshape(1,400,400,3)) > 0.5, "dog","cat")
# %%
plt.imshow(data)