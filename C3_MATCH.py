
import sys, json, jsonlines
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

N = 4
max_length = 512
# batch_size = 64
batch_size = 16
epoch = 4
lr = 1e-6
cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.25)

pretrain_path = sys.argv[-1]
if 'large' in pretrain_path:
    batch_size = 8
config = AutoConfig.from_pretrained(pretrain_path)
tokenizer = AutoTokenizer.from_pretrained(pretrain_path, config=config)

def open_file(fn):
    data = json.load(open(fn, 'r'))
    res = []
    for line in data:
        sentence = ' '.join(line[0])[:400]
        ans = -1
        for qa in line[1]:
            for choice in qa['choice']:
                res.append({'sentence':sentence, 'question':qa['question'], 'choice':choice, 'label':int(choice == qa['answer'])})
            for j in range(N-len(qa['choice'])):
                res.append({'sentence':sentence, 'question':qa['question'], 'choice':'否', 'label':0})
    return res

C3_train = open_file('data/C3/train.json')
C3_valid = open_file('data/C3/valid.json')
C3_test = open_file('data/C3/test_public.json')
ans = json.load(open('data/C3/test_public.json', 'r'))

def tokenize(dataset, is_test, shuffle=False):
    sentences = ["[CLS] " + data['sentence'] + " [SEP] " + data['question'] + " [SEP]" for data in dataset]
    poems = ["[CLS] " + data['choice'] + " [SEP]" for data in dataset]
    tokenized_textsA = [tokenizer.tokenize(sentence) for sentence in sentences]
    tokenized_textsB = [tokenizer.tokenize(poem) for poem in poems]

    print(tokenized_textsA[0], tokenized_textsB[0])
    # exit(0)

    input_idsA = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_textsA]
    out_size = sum([len(sequence) >= 440 for sequence in input_idsA])
    print('{} / {} sentences exceeds length limit.'.format(out_size, len(input_idsA)))
    input_idsB = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_textsB]
    out_size = sum([len(sequence) >= 40 for sequence in input_idsB])
    print('{} / {} poems exceeds length limit.'.format(out_size, len(input_idsB)))
    input_idsA = pad_sequences(input_idsA, maxlen=440, dtype="long", truncating="post", padding="post")
    input_idsB = pad_sequences(input_idsB, maxlen=40, dtype="long", truncating="post", padding="post")

    attention_masksA = [[float(i > 0) for i in sequence] for sequence in input_idsA]
    attention_masksB = [[float(i > 0) for i in sequence] for sequence in input_idsB]
    
    if not is_test:
        labels = [data['label'] for data in dataset]
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_idsA), torch.tensor(attention_masksA), torch.tensor(input_idsB), torch.tensor(attention_masksB), torch.tensor(labels))
    else: 
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_idsA), torch.tensor(attention_masksA), torch.tensor(input_idsB), torch.tensor(attention_masksB))
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
modelA = AutoModel.from_pretrained(pretrain_path, config=config).cuda()
modelB = AutoModel.from_pretrained(pretrain_path, config=config).cuda()
## ====== END YOUR CODE ===============

# define training arguments, which can be changed, but are not required.
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
paramsA, paramsB = modelA.named_parameters(), modelB.named_parameters()
optimizer = AdamW([
    { 'params': [p for n, p in paramsA if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': lr, 'ori_lr': lr },
    { 'params': [p for n, p in paramsA if any(nd in n for nd in no_decay)],  'weight_decay': 0.0, 'lr': lr, 'ori_lr': lr },
    { 'params': [p for n, p in paramsB if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': lr, 'ori_lr': lr },
    { 'params': [p for n, p in paramsB if any(nd in n for nd in no_decay)],  'weight_decay': 0.0, 'lr': lr, 'ori_lr': lr }
], correct_bias=False)


def Train():
    modelA.train(), modelB.train()
    tr_loss, tr_steps = 0, 0
    for step, batch in enumerate(C3_train_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_idsA, b_input_maskA, b_input_idsB, b_input_maskB, b_labels = batch

        optimizer.zero_grad()
        if 'electra' in pretrain_path:
            hiddenA = modelA(b_input_idsA, token_type_ids=None, attention_mask=b_input_maskA, return_dict=False)[0]
            hiddenB = modelB(b_input_idsB, token_type_ids=None, attention_mask=b_input_maskB, return_dict=False)[0]
        else:
            hiddenA, _ = modelA(b_input_idsA, token_type_ids=None, attention_mask=b_input_maskA, return_dict=False)
            hiddenB, _ = modelB(b_input_idsB, token_type_ids=None, attention_mask=b_input_maskB, return_dict=False)
        
        b_loss = cosine_loss(hiddenA[:,0,:], hiddenB[:,0,:], b_labels)
        b_loss.backward()
        optimizer.step()

        tr_loss += b_loss.item()
        tr_steps += 1
    print("Train loss: {}".format(tr_loss / tr_steps))

def MetricFunc(label, pred):
	return {'Accuracy': accuracy_score(label, pred), 'AUC': roc_auc_score(label, pred), 'Precision':precision_score(label, pred), 'Recall':recall_score(label, pred), 'F1 Score':f1_score(label, pred)}

def Valid():
    modelA.eval(), modelB.eval()
    logits, labels = [], []
    for step, batch in enumerate(C3_valid_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_idsA, b_input_maskA, b_input_idsB, b_input_maskB, b_labels = batch
        with torch.no_grad():
            if 'electra' in pretrain_path:
                hiddenA = modelA(b_input_idsA, token_type_ids=None, attention_mask=b_input_maskA, return_dict=False)[0]
                hiddenB = modelB(b_input_idsB, token_type_ids=None, attention_mask=b_input_maskB, return_dict=False)[0]
            else:
                hiddenA, _ = modelA(b_input_idsA, token_type_ids=None, attention_mask=b_input_maskA, return_dict=False)
                hiddenB, _ = modelB(b_input_idsB, token_type_ids=None, attention_mask=b_input_maskB, return_dict=False)
            b_logits = torch.cosine_similarity(hiddenA[:,0,:], hiddenB[:,0,:])
        logits.append(b_logits.cpu())
        labels.append(b_labels.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    labels = torch.cat([_ for _ in labels], dim=0)
    logits = logits.view(-1, N)
    preds = torch.argmax(logits, -1)
    labels = labels.view(-1, N)
    ans = torch.argmax(labels, -1)
    return {'Accuracy': (ans == preds).float().mean()}

def Test():
    modelA.eval(), modelB.eval()
    logits = []
    for step, batch in enumerate(C3_test_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_idsA, b_input_maskA, b_input_idsB, b_input_maskB = batch
        
        with torch.no_grad():
            if 'electra' in pretrain_path:
                hiddenA = modelA(b_input_idsA, token_type_ids=None, attention_mask=b_input_maskA, return_dict=False)[0]
                hiddenB = modelB(b_input_idsB, token_type_ids=None, attention_mask=b_input_maskB, return_dict=False)[0]
            else:
                hiddenA, _ = modelA(b_input_idsA, token_type_ids=None, attention_mask=b_input_maskA, return_dict=False)
                hiddenB, _ = modelB(b_input_idsB, token_type_ids=None, attention_mask=b_input_maskB, return_dict=False)
            b_logits = torch.cosine_similarity(hiddenA[:,0,:], hiddenB[:,0,:])
        logits.append(b_logits.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    logits = logits.view(-1, N)
    preds = torch.argmax(logits, -1)
    idx = 0
    for i, data in enumerate(ans):
        for j, qa in enumerate(data[1]):
            ans[i][1][j]['answer'] = qa['choice'][min(preds[idx], len(qa['choice'])-1)]
            idx += 1
    json.dump(ans, open('preds/C3-pred-{}-March.json'.format(pretrain_path.replace('/', '-')), 'w'), ensure_ascii=False)

def Main():
    best_acc = 0
    for i in range(epoch):
        print('[START] Train epoch {}.'.format(i))
        Train()
        print('[END] Train epoch {}.'.format(i))
        metric = Valid()
        print(metric)
        if metric['Accuracy'] > best_acc:
            best_acc = metric['Accuracy']
            Test()

Main()
print('[END]')
