import torch
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pickle
from torch import nn
from kobert_tokenizer import KoBERTTokenizer
from kobert import get_pytorch_kobert_model

bertmodel, vocab = get_pytorch_kobert_model()

M_PATH = '/workspace/Server_socket_AI/AI/koBERT_M.pth'
T_PATH = "/workspace/Server_socket_AI/AI/tok.pickle"
V_PATH = "/workspace/Server_socket_AI/AI/vocab.pickle"

device = torch.device('cpu')

max_len = 200 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 64 # 배치 크기

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

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

with open(T_PATH, 'rb') as handle:
    tok = pickle.load(handle)
    
with open(V_PATH, 'rb') as handle:
    vocab = pickle.load(handle)

model = BERTClassifier(bertmodel,  dr_rate=0.5)
model.load_state_dict(torch.load(M_PATH, map_location=device))

def calc_accuracy(X):
    max_vals, max_indices = torch.max(X, 1)
    acc = nn.functional.softmax(X, dim=-1).cpu().numpy()
    return acc[0][max_indices]


def VP_predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data] 

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=4)
    test_acc = 0.0
    model.eval() 
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate((test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            prediction = out.cpu().detach().numpy().argmax()
            test_acc = calc_accuracy(out)
    if prediction == 1:
        print("=>보이스피싱일 확률 {:.2f}%".format(test_acc * 100))
        return "1"
    else:
        print("=>보이스피싱일 확률 {:.2f}%".format((1-test_acc) * 100))
        return "0"

