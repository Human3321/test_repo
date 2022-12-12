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
# 사전 학습된 BERT를 사용할 때는 transformers라는 패키지를 자주 사용

# 학습시간을 줄이기 위해, GPU를 사용
device = torch.device("cuda:0")

from kobert_tokenizer import KoBERTTokenizer
# KoBERT 의 tokenizer 객체 생성
#(이후 우리 데이터셋으로 변경해야함)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# VP 데이터 셋 불러옴 
data = pd.read_csv('/workspace/ai/VP_dataset.csv')
# 잘 불러왔는지 확인
# print(data[:5])

# 데이터와 label을 list로 묶고 각각 합친 list를 data_list로 전체 묶음
data_list = []
for ques, label in zip(data['대화'], data['VP 여부'])  :
    data = []   
    data.append(ques)
    data.append(str(label))

    data_list.append(data)
    
# 입력 데이터셋을 토큰화하기
# 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding 등을 해주는 class 코드
class BERTDataset(Dataset):
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
max_len = 200 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64 # 배치 크기
warmup_ratio = 0.1 
num_epochs = 5  # 학습 횟수
max_grad_norm = 1 
log_interval = 200
learning_rate =  5e-5 # 학습률

#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split
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
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
#BERT 모델 불러오기
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
    
train_history=[]
test_history=[]
loss_history=[]

torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()


for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    # with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        #print(label.shape,out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {} loss{}".format(e+1, train_acc / (batch_id+1), loss.data.cpu().numpy()))
    #train_history.append(train_acc / (batch_id+1))

    model.eval()
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))

PATH = "/workspace/ai/koBERT_M.pth"
torch.save(model.state_dict(), PATH)

loss.data.cpu().numpy()
import pickle 

with open('/workspace/ai/tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle)

with open('/workspace/ai/tok.pickle', 'rb') as handle:
    tok = pickle.load(handle)

with open('/workspace/ai/vocab.pickle', 'wb') as handle:
    pickle.dump(vocab, handle)
    
def VP_predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        prediction = out.cpu().detach().numpy().argmax()
        if prediction == 1:
            return 1
        else:
            return 0
        
# VP o
print(VP_predict('본인이 직접 움직일 필요 없으시고요 그냥 그 통장 안에 잔고없는 통장과 연결된 현금카드 갖고 계시는 거 있죠?\n네\n그냥 그 현금인출카드만 저희쪽으로 한 달 동안 빌려주시는 거에요'))
VP_predict('어 저는 금융범죄 수사 1팀장을 맡고 있는 신승용 검사라고 합니다.\n메모하세요 자 명의 도용 사건 내용 이해하셨나요?\n자 그럼 이 사건에 대해서는 지금 본인 사건이기 때문에 본인께서 구체적으로 알 권리가 있습니다.\n따라서 본 검사는 이번 사건에 대해서 구체적으로 설명을 해드릴 건데 설명 도중에 이해 못하시는 부분이 있으면 질문해 주시기 바랍니다.\n음 저희 검찰은 김혜선 외 공범 8명 금융범죄 사기단을 검거했습니다.')
VP_predict('면허증 뭐 여권 같은 것도 전혀 없으신 거죠? 재발급 받아 보신 적도 한번도 없으신 거고.')

# VP x
VP_predict('A : 저번에 빌린 10만원 언제 갚을거야?\nB : 까먹고 있었네.\n계좌번호 불러줘 지금 바로 송금할께\nA : 국민은행 ***…\nB : 송금했어. 다음번에 내가 밥 한번 살께')
VP_predict('반갑습니다 상담사 땡땡땡입니다\n예 수고하십니다 저 세탁기가 작동이 안 돼요\n작동이 안 된다면은 뭐 전원은 들어오는데 회전만 안 되는 거에여\n네 전원은 들어와요\n많이 답답하셨겠습니다 고객님 저희 세탁기가 일반 통세탁기세여 드럼세탁기세여\n예 드럼 세탁기여\n드럼이구 혹시 사용하신 지는 일 년이 지나셨어여')