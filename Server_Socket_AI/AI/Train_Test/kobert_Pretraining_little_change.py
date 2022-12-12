# 필요 모듈
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import torch, gc
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

# 학습시간을 줄이기 위해, GPU를 사용
device = torch.device("cuda:0")

# KoBERT 의 tokenizer 객체 생성
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# VP 데이터 셋 불러옴 
data = pd.read_csv('/workspace/ai/VP_dataset.csv')
# 잘 불러왔는지 확인
print(data[:5])

# 대화 데이터와 label을 list로 묶고 각각 합친 list를 data_list로 전체 묶음
data_list = []
for ques, label in zip(data['대화'], data['VP 여부'])  :
    data = []   
    data.append(ques)
    data.append(str(label))

    data_list.append(data)
    
# 입력 데이터셋을 토큰화하기
# 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding 등을 해주는 class 코드
class BERTDataset(Dataset):
    # 초기화 함수
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    # i번째 데이터와 데이터의 label return
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         
    # label 길이
    def __len__(self):
        return (len(self.labels))

# parameter의 경우, 예시 코드에 있는 값들을 동일하게 설정
# Setting parameters
max_len = 64 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64 # 배치 크기
warmup_ratio = 0.1 
num_epochs = 5  # 학습 횟수
max_grad_norm = 1 
log_interval = 200
learning_rate =  5e-5 # 학습률

#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split
# 학습 데이터 : 테스트 데이터 = 75 : 25 비율로 무작위 분리
dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, shuffle=True, random_state=42)

# BERTDataset 클래스를 활용해 tokenization, int encoding, padding 을 진행
tok=tokenizer.tokenize
data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test,0, 1, tok, vocab,  max_len, True, False)

# Torch 형식의 dataset을 만들어주면서, 입력 데이터셋의 처리가 모두 끝
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

# KoBERT 모델 구현
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,#은닉층
                 num_classes=2,   ##클래스 수 조정##
                 dr_rate=None, #dropout 비율
                 params=None): 
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                            
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    
#BERT 모델 불러오기 dropout 50%
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
 
#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
train_history=[] # 학습 과정 기록
test_history=[]  # 테스트 과정 기록
loss_history=[]  # 오차률 기록

# GPU 캐시 초기화
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

# 학습 시작
for e in range(num_epochs):
    train_acc = 0.0 # 학습 정확도
    test_acc = 0.0 # 테스트 정확도

    # 모델 학습
    model.train()
    # with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        loss = loss_fn(out, label)
        # 오차(error) 역전파 확인
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label) # 학습 중 총 정확도 측정
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    train_history.append(train_acc / (batch_id+1))
    
    # 모델 평가
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        # 테스트 정확도 측정
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))

# 모델 저장
PATH = "/workspace/ai/best_model.pth"
torch.save(model, PATH)

# vocab, tok pickle 저장
import pickle 

with open('/workspace/ai/tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle)

with open('/workspace/ai/vocab.pickle', 'wb') as handle:
    pickle.dump(vocab, handle)
    