
import sys, json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

N = 4
max_length = 512
# batch_size = 64
batch_size = 20
accumulation_steps = 8
epoch = 8
lr = 1e-6


## Hint: load tokenizer
## ====== YOUR CODE HERE ==============
# pretrain_path = 'thu-cbert-character'
# pretrain_path = 'bert-base-chinese'
# pretrain_path = 'google/mt5-large'
# pretrain_path = 'ethanyt/guwenbert-large'
# pretrain_path = 'hfl/chinese-roberta-wwm-ext-large'
# pretrain_path = 'hfl/chinese-electra-180g-large-discriminator'
# pretrain_path = 'hfl/chinese-macbert-large'
# pretrain_path = 'ethanyt/guwenbert-base'
pretrain_path = sys.argv[-1]
if 'large' in pretrain_path:
    batch_size = 10
    if 'electra' in pretrain_path:
        epoch = 40
config = AutoConfig.from_pretrained(pretrain_path)
tokenizer = AutoTokenizer.from_pretrained(pretrain_path, config=config)
## ====== END YOUR CODE ===============


## Hint: load dataset
## ====== YOUR CODE HERE ==============
def open_file(fn):
    data = json.load(open(fn, 'r'))
    res = []
    for line in data:
        sentence = ' '.join(line[0])[:400]
        for qa in line[1]:
            ans = -1
            for i, choice in enumerate(qa['choice']):
                if choice == qa['answer']:
                    ans = i
            res.append({'sentence':sentence, 'question':qa['question'], 'choices':qa['choice'], 'answer':ans})
    return res

C3_train = open_file('data/C3/train.json')
C3_valid = open_file('data/C3/valid.json')
C3_test = open_file('data/C3/test_public.json')
ans = json.load(open('data/C3/test_public.json', 'r'))

## ====== END YOUR CODE ===============

def tokenize(dataset, is_test, shuffle=False):
    tokenized_texts, pos = [], [[], [], [], [], []]
    for data in dataset:
        sentence = tokenizer.tokenize("[CLS] " + data['sentence'] + " [SEP] ")
        pos[0].append(len(sentence))
        sentence += ['[unused0]'] +  tokenizer.tokenize(data['question'] + " [SEP] ")
        for k in range(4):
            if len(data['choices']) > k:
                pos[k+1].append(min(len(sentence), max_length - 1))
                sentence += ['[unused1]'] + tokenizer.tokenize(data['choices'][k] + " [SEP]")
            else:
                pos[k+1].append(min(len(sentence) - 1, max_length - 1))
        tokenized_texts.append(sentence)

    print(tokenized_texts[0])

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    out_size = sum([len(sequence) >= max_length for sequence in input_ids])
    print('{} / {} sentences exceeds length limit.'.format(out_size, len(input_ids)))
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")

    attention_masks = [[float(i > 0) for i in sequence] for sequence in input_ids]
    
    if not is_test:
        labels = [data['answer'] for data in dataset]
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(pos[0]), torch.tensor(pos[1]), torch.tensor(pos[2]), torch.tensor(pos[3]), torch.tensor(pos[4]), torch.tensor(labels))
    else: 
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(pos[0]), torch.tensor(pos[1]), torch.tensor(pos[2]), torch.tensor(pos[3]), torch.tensor(pos[4]))
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
    ## ====== END YOUR CODE ===============

C3_train_dataloader = tokenize(C3_train, False, shuffle=True)
C3_valid_dataloader = tokenize(C3_valid, False, shuffle=False)
C3_test_dataloader = tokenize(C3_test, True, shuffle=False)

## Hint: Load SequenceClassification Model
## ====== YOUR CODE HERE ==============
class CCPMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrain_path, config=config)
        self.fc1 = torch.nn.Linear(config.hidden_size * 6, config.hidden_size * 6)
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(config.hidden_size * 6, N)
    def forward(self, b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE, is_test=False):
        if is_test:
            with torch.no_grad():
                if 'electra' in pretrain_path:
                    hidden = self.bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=False)[0]
                else:
                    hidden, _ = self.bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=False)
                cls_pos = torch.zeros(b_posA.size()).long().to('cuda')
                onehot_head = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
                onehot_A = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
                onehot_B = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
                onehot_C = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
                onehot_D = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
                onehot_E = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
                onehot_head = onehot_head.scatter_(1, cls_pos.unsqueeze(-1), 1)
                onehot_A = onehot_A.scatter_(1, b_posA.unsqueeze(-1), 1)
                onehot_B = onehot_B.scatter_(1, b_posB.unsqueeze(-1), 1)
                onehot_C = onehot_C.scatter_(1, b_posC.unsqueeze(-1), 1)
                onehot_D = onehot_D.scatter_(1, b_posD.unsqueeze(-1), 1)
                onehot_E = onehot_E.scatter_(1, b_posE.unsqueeze(-1), 1)
                head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
                tail_A = (onehot_A.unsqueeze(2) * hidden).sum(1)  # (B, H)
                tail_B = (onehot_B.unsqueeze(2) * hidden).sum(1)  # (B, H)
                tail_C = (onehot_C.unsqueeze(2) * hidden).sum(1)  # (B, H)
                tail_D = (onehot_D.unsqueeze(2) * hidden).sum(1)  # (B, H)
                tail_E = (onehot_E.unsqueeze(2) * hidden).sum(1)  # (B, H)
                b_embeds = torch.cat([head_hidden, tail_A, tail_B, tail_C, tail_D, tail_E], 1)  # (B, 6H)
                b_logits = self.fc2(self.dropout(self.fc1(b_embeds))) # (B, N)
        else:
            if 'electra' in pretrain_path:
                hidden = self.bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=False)[0]
            else:
                hidden, _ = self.bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=False)
            cls_pos = torch.zeros(b_posA.size()).long().to('cuda')
            onehot_head = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
            onehot_A = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
            onehot_B = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
            onehot_C = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
            onehot_D = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
            onehot_E = torch.zeros(hidden.size()[:2]).float().to('cuda')  # (B, L)
            onehot_head = onehot_head.scatter_(1, cls_pos.unsqueeze(-1), 1)
            onehot_A = onehot_A.scatter_(1, b_posA.unsqueeze(-1), 1)
            onehot_B = onehot_B.scatter_(1, b_posB.unsqueeze(-1), 1)
            onehot_C = onehot_C.scatter_(1, b_posC.unsqueeze(-1), 1)
            onehot_D = onehot_D.scatter_(1, b_posD.unsqueeze(-1), 1)
            onehot_E = onehot_E.scatter_(1, b_posE.unsqueeze(-1), 1)
            head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
            tail_A = (onehot_A.unsqueeze(2) * hidden).sum(1)  # (B, H)
            tail_B = (onehot_B.unsqueeze(2) * hidden).sum(1)  # (B, H)
            tail_C = (onehot_C.unsqueeze(2) * hidden).sum(1)  # (B, H)
            tail_D = (onehot_D.unsqueeze(2) * hidden).sum(1)  # (B, H)
            tail_E = (onehot_E.unsqueeze(2) * hidden).sum(1)  # (B, H)
            b_embeds = torch.cat([head_hidden, tail_A, tail_B, tail_C, tail_D, tail_E], 1)  # (B, 6H)
            b_logits = self.fc2(self.dropout(self.fc1(b_embeds))) # (B, N)
        return b_logits
model = CCPMModel().cuda()
criterion = torch.nn.CrossEntropyLoss()
## ====== END YOUR CODE ===============

# define training arguments, which can be changed, but are not required.
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
params = model.named_parameters()
optimizer = AdamW([
    { 'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': lr, 'ori_lr': lr },
    { 'params': [p for n, p in params if any(nd in n for nd in no_decay)],  'weight_decay': 0.0, 'lr': lr, 'ori_lr': lr }
], correct_bias=False)

def Train():
    model.train()
    tr_loss, tr_steps = 0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(C3_train_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE, b_labels = batch
        b_logits = model(b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE)
        b_loss = criterion(b_logits, b_labels)
        b_loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        tr_loss += b_loss.item()
        tr_steps += 1
    print("Train loss: {}".format(tr_loss / tr_steps))

def MetricFunc(label, pred):
	return {'Accuracy': accuracy_score(label, pred), 'AUC': roc_auc_score(label, pred), 'Precision':precision_score(label, pred), 'Recall':recall_score(label, pred), 'F1 Score':f1_score(label, pred)}

def Valid():
    model.eval()
    logits, labels = [], []
    for step, batch in enumerate(C3_valid_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE, b_labels = batch
        b_logits = model(b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE, True)
        logits.append(b_logits.cpu())
        labels.append(b_labels.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    labels = torch.cat([_ for _ in labels], dim=0)
    preds = torch.argmax(logits, -1)
    return {'Accuracy': (labels == preds).float().mean()}

def Test(e):
    model.eval()
    logits = []
    for step, batch in enumerate(C3_test_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE = batch
        b_logits = model(b_input_ids, b_input_mask, b_posA, b_posB, b_posC, b_posD, b_posE, True)
        logits.append(b_logits.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    preds = torch.argmax(logits, -1)
    idx = 0
    for i, data in enumerate(ans):
        for j, qa in enumerate(data[1]):
            ans[i][1][j]['answer'] = qa['choice'][preds[idx]]
            idx += 1
    json.dump(ans, open('preds/C3-pred-{}.json'.format(pretrain_path.replace('/', '-')), 'w'), ensure_ascii=False)

def Main():
    best_acc = -1
    for i in range(epoch):
        print('[START] Train epoch {}.'.format(i))
        Train()
        print('[END] Train epoch {}.'.format(i))
        metric = Valid()
        print(metric)
        if metric['Accuracy'] > best_acc:
            best_acc = metric['Accuracy']
            Test(i)
    print('[BEST ACC: {}]'.format(best_acc))

Main()
# print(Valid())