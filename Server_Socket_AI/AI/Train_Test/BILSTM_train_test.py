# Colab에 Mecab 설치
# -> 개인 리눅스 or 윈도우사용시 다른 방법 필요
# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
# !bash install_mecab-ko_on_colab190912.sh

# 필요한 모듈 설치
import pickle
import numpy as np
import io
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import SimpleRNN, Dense, LSTM, GRU, Bidirectional, Flatten
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves.urllib.request import urlopen

#필수#
# 데이터 파일 위치 경로로 바꿔주기
datafile = "C:\Users\samsung\Desktop\git\Server_socket_AI\AI\AI_train_test\VP데이터셋\VP_dataset.csv"
# If the training and test sets aren't stored locally, download them.
if not os.path.exists(datafile):
  print("파일 없음")
# 학습데이터 load
total_data = pd.read_csv(datafile,delimiter=',')
# load 데이터 개수 / 구조 확인
print('전체 리뷰 개수 :',len(total_data))
print(total_data[:5])
print(total_data.shape)
# 각 열에 대해서 중복을 제외한 샘플의 수를 카운트
total_data['대화'].nunique(), total_data['VP 여부'].nunique()

# NULL 값 유무를 확인
print(total_data.isnull().values.any()) 

# 훈련 데이터와 테스트 데이터를 3:1 비율로 분리
train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
# 나눈 개숫 확인용
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))

# 훈련/테스트 데이터의 레이블의 분포
print(train_data.groupby('VP 여부').size().reset_index(name = 'count'))
print(test_data.groupby('VP 여부').size().reset_index(name = 'count'))

# 정규 표현식을 사용하여 한글을 제외하고 모두 제거
# 혹시 이 과정에서 빈 샘플이 생기지는 않는지 확인
# 한글과 공백을 제외하고 모두 제거
train_data['대화'] = train_data['대화'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['대화'].replace('', np.nan, inplace=True)
# null 값 확인
print(train_data.isnull().sum())

test_data.drop_duplicates(subset = ['대화'], inplace=True) # 중복 제거
test_data['대화'] = test_data['대화'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['대화'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

# 불용어(의미 없는 용어)를 정의
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']
# 추가 예정
tmp = ['히히', '땡땡땡', '땡땡', '땡']
stopwords = stopwords + tmp
# 전체 불용어 확인용
print(stopwords)

# 형태소 분석기 Mecab을 사용하여 토큰화 작업을 수행
mecab = Mecab() 
# 불용어 제거하고 토큰화
train_data['tokenized'] = train_data['대화'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized'] = test_data['대화'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

# 단어와 길이 분포 계산 후 리스트로 저장
nonVP_words = np.hstack(train_data[train_data['VP 여부'] == 0]['tokenized'].values)
VP_words = np.hstack(train_data[train_data['VP 여부'] == 1]['tokenized'].values)

# Counter()를 사용하여 각 단어에 대한 빈도수를 카운트 후 상위 20개 단어 출력
nonVP_words_count = Counter(nonVP_words)
print(nonVP_words_count.most_common(20))

VP_words_count = Counter(VP_words)
print(VP_words_count.most_common(20))

# label / 데이터 나누기
# 학습 데이터
X_train = train_data['tokenized'].values
y_train = train_data['VP 여부'].values
# 테스트 데이터
X_test= test_data['tokenized'].values
y_test = test_data['VP 여부'].values

# 훈련 데이터에 대해서 단어 집합(vocaburary) 생성
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
# 단어-> 인덱스 부여
tokenizer.fit_on_texts(X_train)
# 텍스트 안의 단어들을 숫자의 시퀀스의 형태로 변환
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# 필요 토큰 객체 저장
import pickle
save_tokenizer = '/content/drive/MyDrive/캡스톤/tokenizer.pickle'
with open(save_tokenizer, 'wb') as handle:
    pickle.dump(tokenizer, handle)

print('대화의 최대 길이 :',max(len(sample) for sample in X_train))
print('대화의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(sample) for sample in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# max_len 이하 길이 비율 구하는 함수
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

# 현재 정한 데이터 셋 기준 200
# 데이터 셋 변할 경우 수정 필요
max_len = 200
below_threshold_len(max_len, X_train)

# 패딩
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": np.array(X_test.data)},
  y=np.array(X_test.target),
  num_epochs=None,
  shuffle=True)

# 필요 토큰 객체 저장
with open(save_tokenizer, 'wb') as handle:
    pickle.dump(tokenizer, handle)

# 학습용 모듈 저장
import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

# 네트워크 구조
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units))) # Bidirectional LSTM을 사용
model.add(Dense(1, activation='sigmoid'))
# 저장 위치
save_model = 'C:\Users\samsung\Desktop\git\Server_socket_AI\AI\best_model.h5'

# 초기 종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# 모델 저장 설정
mc = ModelCheckpoint(save_model, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# 학습 설정
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# 학습 시작(validation 20%)
history = model.fit(X_train, y_train, epochs=32, callbacks=[es, mc], batch_size=256, validation_split=0.2)

# 학습 과정 그래프로 나타내기
plt.figure(figsize = (12,4)) # 그래프 가로세로 비율 (그림(figure)의 크기, (가로, 세로) 인치 단위)
plt.subplot(1,1,1) # 1행 1열 첫 번째 위치
plt.plot(history.history['loss'], 'b--', label = 'loss') # loss 파란색 점선
plt.plot(history.history['accuracy'], 'g--', label = 'Accuracy') # accuracy 는 녹색실선
plt.xlabel('Epoch')
plt.legend()
plt.show()
print('최적화 완료!')

# 테스트 데이터 확인
print("\n============test results============")
loaded_model = load_model(save_model)
print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# 실습용 함수
def VP_predict(new_sentence):
  # new_sectence 정규화
  new_sentence_data = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
   # 토큰화
  new_sentence_data = mecab.morphs(new_sentence_data)
   # 불용어 제거
  new_sentence_data = [word for word in new_sentence_data if not word in stopwords]
   # 정수 인코딩
  encoded = tokenizer.texts_to_sequences([new_sentence_data])
   # 패딩
  pad_new = pad_sequences(encoded, maxlen = max_len)
  score = float(loaded_model.predict(pad_new, verbose=0)) # 예측
  # 결과 출력
  print(new_sentence)
  print("="*40)
  print("=>보이스피싱일 확률 {:.2f}%".format(score * 100))
  if(score > 0.6):
    print("판별 결과 : 보이스피싱입니다.")
  else:
    print("판별 결과 : 보이스피싱이 아닙니다.")
  print()
  print("-"*40)


# 새로운 문장으로 확인
# VP o
VP_predict('본인이 직접 움직일 필요 없으시고요 그냥 그 통장 안에 잔고없는 통장과 연결된 현금카드 갖고 계시는 거 있죠?\n네\n그냥 그 현금인출카드만 저희쪽으로 한 달 동안 빌려주시는 거에요')
VP_predict('어 저는 금융범죄 수사 1팀장을 맡고 있는 신승용 검사라고 합니다.\n메모하세요 자 명의 도용 사건 내용 이해하셨나요?\n자 그럼 이 사건에 대해서는 지금 본인 사건이기 때문에 본인께서 구체적으로 알 권리가 있습니다.\n따라서 본 검사는 이번 사건에 대해서 구체적으로 설명을 해드릴 건데 설명 도중에 이해 못하시는 부분이 있으면 질문해 주시기 바랍니다.\n음 저희 검찰은 김혜선 외 공범 8명 금융범죄 사기단을 검거했습니다.')
VP_predict('면허증 뭐 여권 같은 것도 전혀 없으신 거죠? 재발급 받아 보신 적도 한번도 없으신 거고.')

# VP x
VP_predict('A : 저번에 빌린 10만원 언제 갚을거야?\nB : 까먹고 있었네.\n계좌번호 불러줘 지금 바로 송금할께\nA : 국민은행 ***…\nB : 송금했어. 다음번에 내가 밥 한번 살께')
VP_predict('반갑습니다 상담사 땡땡땡입니다\n예 수고하십니다 저 세탁기가 작동이 안 돼요\n작동이 안 된다면은 뭐 전원은 들어오는데 회전만 안 되는 거에여\n네 전원은 들어와요\n많이 답답하셨겠습니다 고객님 저희 세탁기가 일반 통세탁기세여 드럼세탁기세여\n예 드럼 세탁기여\n드럼이구 혹시 사용하신 지는 일 년이 지나셨어여')