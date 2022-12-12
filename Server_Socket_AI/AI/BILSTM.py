from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from tensorflow.keras.models import load_model

save_model = '/workspace/Server_socket_AI/AI/best_model_2.h5'
save_tokenizer = '/workspace/Server_socket_AI/AI/tokenizer_2.pickle'
loaded_model = load_model(save_model)

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']
tmp = ['히히', '땡땡땡', '땡땡', '땡']
stopwords = stopwords + tmp

# 형태소 분석기 Mecab을 사용하여 토큰화 작업을 수행
mecab = Mecab()

max_len = 200

with open(save_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

def VP_predict(new_sentence):
    new_sentence_data = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence_data = mecab.morphs(new_sentence_data) # 토큰화
    new_sentence_data = [word for word in new_sentence_data if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence_data]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new, verbose=0)) # 예측
    # print("="*40)
    print("=>보이스피싱일 확률 {:.2f}%".format(score * 100))
    if(score > 0.6):
        print("판별 결과 : 보이스피싱입니다.")
    else:
        print("판별 결과 : 보이스피싱이 아닙니다.")
    print()
    print("-"*40)
    return score
