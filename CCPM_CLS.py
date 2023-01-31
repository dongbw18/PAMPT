
import sys, json, jsonlines
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

N = 4
max_length = 100
batch_size = 64
# batch_size = 40 # for large
epoch = 8
lr = 1e-6


## Hint: load tokenizer
## ====== YOUR CODE HERE ==============
# pretrain_path = 'thu-cbert-character'
# pretrain_path = 'bert-base-chinese'
pretrain_path = sys.argv[-1]
if 'large' in pretrain_path:
    batch_size = 40
config = AutoConfig.from_pretrained(pretrain_path)
tokenizer = AutoTokenizer.from_pretrained(pretrain_path, config=config)
## ====== END YOUR CODE ===============


## Hint: load dataset
## ====== YOUR CODE HERE ==============
def open_file(fn):
    with jsonlines.open(fn) as reader:
        try:
            return [ele for ele in reader]
        except KeyError:
            return

ccpm_train = open_file('data/CCPM/train_01.jsonl')
ccpm_valid = open_file('data/CCPM/valid_01.jsonl')
ccpm_test = open_file('data/CCPM/test_01.jsonl')

## ====== END YOUR CODE ===============

def tokenize(dataset, shuffle=False):
    ## ====== YOUR CODE HERE ==============
    sentences = ["[CLS] " + data['translation'] + " [SEP] " + data['choice'] + " [SEP]" for data in dataset]
    tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]

    print(tokenized_texts[0])
    # exit(0)

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    out_size = sum([len(sequence) >= max_length for sequence in input_ids])
    print('{} / {} sentences exceeds length limit.'.format(out_size, len(input_ids)))
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")

    attention_masks = [[float(i > 0) for i in sequence] for sequence in input_ids]
    
    if 'label' in dataset[0]:
        labels = [data['label'] for data in dataset]
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels))
    else: 
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks))
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
    ## ====== END YOUR CODE ===============

ccpm_train_dataloader = tokenize(ccpm_train, shuffle=True)
ccpm_valid_dataloader = tokenize(ccpm_valid, shuffle=False)
ccpm_test_dataloader = tokenize(ccpm_test, shuffle=False)

## Hint: Load SequenceClassification Model
## ====== YOUR CODE HERE ==============
model = AutoModelForSequenceClassification.from_pretrained(pretrain_path, config=config).cuda()
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
    for step, batch in enumerate(ccpm_train_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        b_loss, b_logits = outputs[0], outputs[1]
        b_loss.backward()
        optimizer.step()

        tr_loss += b_loss.item()
        tr_steps += 1
    print("Train loss: {}".format(tr_loss / tr_steps))

def MetricFunc(label, pred):
	return {'Accuracy': accuracy_score(label, pred), 'AUC': roc_auc_score(label, pred), 'Precision':precision_score(label, pred), 'Recall':recall_score(label, pred), 'F1 Score':f1_score(label, pred)}

# def Valid01():
#     model.eval()
#     logits, labels = [], []
#     for step, batch in enumerate(ccpm_valid_dataloader):
#         batch = tuple(t.to('cuda') for t in batch)
#         b_input_ids, b_input_mask, b_labels = batch
#         with torch.no_grad():
#             outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#             b_logits = outputs[0]
#         logits.append(b_logits.cpu())
#         labels.append(b_labels.cpu())
#     logits = torch.cat([_ for _ in logits], dim=0)
#     labels = torch.cat([_ for _ in labels], dim=0)
#     preds = torch.argmax(logits, -1)
#     return MetricFunc(labels, preds)

def Valid():
    model.eval()
    logits, labels = [], []
    for step, batch in enumerate(ccpm_valid_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logits = outputs[0]
        logits.append(b_logits.cpu())
        labels.append(b_labels.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    labels = torch.cat([_ for _ in labels], dim=0)
    logits = logits[:,1].view(-1, N)
    preds = torch.argmax(logits, -1)
    labels = labels.view(-1, N)
    ans = torch.argmax(labels, -1)
    return {'Accuracy': (ans == preds).float().mean()}

def write_file(fn, raw_datas):
    with jsonlines.open(fn, 'w') as writer:
        writer.write_all(raw_datas)

def Test():
    model.eval()
    logits = []
    for step, batch in enumerate(ccpm_test_dataloader):
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logits = outputs[0]
        logits.append(b_logits.cpu())
    logits = torch.cat([_ for _ in logits], dim=0)
    logits = logits[:,1].view(-1, N)
    preds = torch.argmax(logits, -1)
    ans = [{'answer': _.item()} for _ in preds]
    write_file('preds/pred-{}-CLS-{}.jsonl'.format(pretrain_path.replace('/', '-'), lr), ans)

def Main():
    best_acc = 0
    for i in range(epoch):
        print('[START] Train epoch {}.'.format(i))
        Train()
        print('[END] Train epoch {}.'.format(i))
        # metric = Valid01()
        # print(metric)
        metric = Valid()
        print(metric)
        if metric['Accuracy'] > best_acc:
            best_acc = metric['Accuracy']
            Test()
    print('[BEST ACC: {}]'.format(best_acc))

Main()
# print(Valid())