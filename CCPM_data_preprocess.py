import jsonlines

def open_file(fn):
    with jsonlines.open(fn) as reader:
        try:
            return [ele for ele in reader]
        except KeyError:
            return
def write_file(fn, raw_datas):
    with jsonlines.open(fn, 'w') as writer:
        writer.write_all(raw_datas)

def trans(original_data):
    res = []
    for data in original_data:
        for i, choice in enumerate(data['choices']):
            if 'answer' in data:
                res.append({'translation': data['translation'], 'choice': choice, 'label': 1 if i==data['answer'] else 0})
            else:
                res.append({'translation': data['translation'], 'choice': choice})
    return res

data_path = 'data/CCPM/'
ccpm_train = open_file(data_path + 'train.jsonl')
ccpm_valid = open_file(data_path + 'valid.jsonl')
ccpm_test = open_file(data_path + 'test_public.jsonl')

write_file(data_path + 'train_01.jsonl', trans(ccpm_train))
write_file(data_path + 'valid_01.jsonl', trans(ccpm_valid))
write_file(data_path + 'test_01.jsonl', trans(ccpm_test))