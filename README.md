# Named Entity Recognition as Dependency Parsing (Adapted for [NEREL](https://huggingface.co/datasets/iluvvatar/NEREL))

## Introduction

This repository contains code introduced in the following paper:

**[Named Entity Recognition as Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.577/)**  
Juntao Yu, Bernd Bohnet and Massimo Poesio  
In *Proceedings of the 58th Annual Conference of the Association for Computational Linguistics (ACL)*, 2020

As well as its adaptation for Russian language data for solving nested NER task on [NEREL](https://huggingface.co/datasets/iluvvatar/NEREL) dataset.
This is a modified codebase, that includes a preprocessing script.
For the sake of brevity, I will omit some of the original README parts.

## Environment Setup

* The Biaffine NER code is written in Python 2.7 and Tensorflow 1.0, preprocessing code is written in python 3.8
* Before you can start palying around with the code, you need to create two separate python environments (for Biaffine NER and for preprocessing accordingly) and run `pip install -r requirements.txt` in each of them.
* You also need to download Russian context-independent word [embeddings](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz).
* To reproduce my results faster, you can access all the preprocessed files and filtered vocab (for RAM efficiency) [here]()

## Using a pre-trained model

* All two of the pretrained models are available at may [Google drive](https://essexuniversity.box.com/s/etbae3f57hts3hr79e5ck5z0tppkoasu). 
* Choose the model you want to use and copy it to the `logs/` folder.
* Modifiy the *test_path* accordingly in the `experiments.conf`:
* the *test_path* is the path to *.jsonl* file, each line of the *.jsonlines* file is a batch of sentences and must in the following format:

```json
 {"doc_key": 1, "ners": [[[0, 4, "PROFESSION"], [2, 3, "ORGANIZATION"], [4, 6, "PERSON"], [8, 9, "PROFESSION"], [13, 15, "DATE"], [16, 18, "AGE"], [25, 26, "ORGANIZATION"], [3, 4, "COUNTRY"], [2, 4, "ORGANIZATION"], [10, 11, "PERSON"]], [[0, 5, "ORGANIZATION"], [6, 9, "DATE"], [3, 5, "ORGANIZATION"], [4, 5, "COUNTRY"]], [[0, 4, "FACILITY"], [1, 4, "ORGANIZATION"], [3, 4, "COUNTRY"], [1, 3, "ORGANIZATION"]], [[2, 8, "PROFESSION"], [6, 8, "ORGANIZATION"], [8, 10, "PERSON"], [14, 17, "PROFESSION"], [19, 21, "DATE"], [3, 8, "ORGANIZATION"], [7, 8, "COUNTRY"], [15, 17, "PROFESSION"], [16, 17, "ORGANIZATION"], [10, 11, "EVENT"]], [[0, 1, "PERSON"], [4, 6, "DATE"], [7, 8, "AGE"], [13, 15, "ORGANIZATION"], [18, 20, "ORGANIZATION"], [27, 30, "DATE"], [14, 15, "COUNTRY"], [19, 20, "COUNTRY"]], [[0, 3, "ORGANIZATION"], [4, 7, "DATE"], [8, 11, "DATE"]], [[0, 1, "PERSON"], [3, 6, "ORGANIZATION"], [5, 6, "ORGANIZATION"]], [[1, 4, "ORGANIZATION"], [7, 10, "ORGANIZATION"]], [[0, 3, "ORGANIZATION"]], [[5, 7, "ORGANIZATION"], [1, 2, "ORDINAL"], [4, 7, "PROFESSION"], [7, 9, "PERSON"], [6, 7, "COUNTRY"], [2, 3, "EVENT"]], [[3, 4, "ORGANIZATION"]], [[5, 8, "PROFESSION"], [20, 22, "PROFESSION"], [14, 16, "PERSON"], [18, 22, "ORGANIZATION"], [7, 8, "ORGANIZATION"], [20, 21, "PROFESSION"], [21, 22, "COUNTRY"], [5, 14, "PROFESSION"]], [[4, 6, "PERSON"], [8, 10, "DATE"], [11, 13, "PROFESSION"], [12, 13, "ORGANIZATION"]]], "sentences": [["Глава", "департамента", "ЦБ", "РФ", "Надежда", "Иванова", "получила", "статус", "зампреда", "\n\n", "Иванова", ",", "которой", "13", "июня", "исполнилось", "60", "лет", ",", "всю", "свою", "жизнь", "проработала", "в", "системе", "ЦБ", "."], ["Сводный", "экономический", "департамент", "Банка", "России", "возглавляет", "с", "1995", "года", ".", "\n"], ["Здание", "Центрального", "банка", "РФ", "."], ["Архив", "\n\n", "Директор", "сводного", "экономического", "департамента", "Банка", "России", "Надежда", "Иванова", "назначена", "также", "на", "должность", "заместителя", "председателя", "ЦБ", ",", "сообщил", "в", "четверг", "регулятор", ".", "\n\n\n\n"], ["Иванова", ",", "у", "которой", "13", "июня", "был", "60-летний", "юбилей", ",", "работает", "в", "системе", "Банка", "России", "(", "ранее", "—", "Госбанка", "СССР", ")", "с", "окончания", "института", ",", "то", "есть", "с", "1975", "года", "."], ["Сводный", "экономический", "департамент", "возглавляет", "почти", "20", "лет", "—", "с", "1995", "года", ".", "\n\n"], ["Иванова", "входит", "в", "совет", "директоров", "Центробанка", "."], ["До", "сводного", "экономического", "департамента", "она", "трудилась", "в", "департаменте", "банковского", "надзора", ".", "\n\n"], ["Сводный", "экономический", "департамент", "входит", "в", "блок", "денежно", "-", "кредитной", "политики", ".", "\n\n"], ["Это", "первое", "назначение", "нового", "председателя", "Банка", "России", "Эльвиры", "Набиуллиной", "на", "этом", "посту", "."], ["Раньше", "в", "руководстве", "Центробанка", "преобладали", "мужчины", "."], ["Эксперты", "ждут", "назначения", "на", "пост", "первого", "зампреда", "ЦБ", "по", "вопросам", "денежно", "-", "кредитной", "политики", "Ксении", "Юдаевой", ",", "возглавляющей", "экспертное", "управление", "президента", "РФ", "."], ["Ранее", "этот", "пост", "занимал", "Алексей", "Улюкаев", ",", "который", "в", "понедельник", "стал", "руководителем", "Минэкономразвития", "."]]}  {"doc_key": "batch_01", 
  "ners": [[[0, 0, "PER"], [3, 3, "GPE"], [5, 5, "GPE"]], 
  [[3, 3, "PER"], [10, 14, "ORG"], [20, 20, "GPE"], [20, 25, "GPE"], [22, 22, "GPE"]], 
  []], 
  "sentences": [["Anwar", "arrived", "in", "Shanghai", "from", "Nanjing", "yesterday", "afternoon", "."], 
  ["This", "morning", ",", "Anwar", "attended", "the", "foundation", "laying", "ceremony", "of", "the", "Minhang", "China-Malaysia", "joint-venture", "enterprise", ",", "and", "after", "that", "toured", "Pudong", "'s", "Jingqiao", "export", "processing", "district", "."], 
  ["(", "End", ")"]]}
  ```
  
* Each of the sentences in the batch corresponds to a list of NEs stored under `ners` key, if some sentences do not contain NEs use an empty list `[]` instead.
* Then use `python evaluate.py config_name` to start your evaluation. The NEREL configurations are `nerel` and `nerel_ce16`.

## Training your own model

* Run `python get_char_vocab.py train.jsonl dev.jsonl`to create the character vocabulary.
* Put the address to the resulting file into experiments.con into the `char_vocab_path` field.
* Start training by running `python train.py config_name`

## Other Versions

* [Original repo](https://github.com/juntaoy/biaffine-ner) The repo, this codebase was forked from.
* [Amir Zeldes](https://github.com/amir-zeldes) kindly created a tensorflow 2.0 and python 3 ready version and can be find [here](https://github.com/amir-zeldes/biaffine-ner)


## Task summary
The task was to provide a solution, that solves the NER task on NEREL dataset.

## Detailed reproduction instructions

1. Clone this repo and cd into it
```bash
git clone https://github.com/gregarshinov/biaffine-ner
cd biaffine-ner
```
2. Create 2 separate python environments and fill them with needed packages.
I used conda for this.
```bash
conda create -y -n nerel27 python=2.7
conda activate nerel27
pip install -r requirements.txt

conda create -y -n nerel38 python=3.8
conda activate nerel38
pip install -r preprocessing/requirements.txt
```
3. Download and preprocess the NEREL data
Activate the **nerel38** environment if you have not done it already. Then do:
```bash
mkdir resources
python preprocessing/provide_nerel.py $(pwd)/resources
```
To run this you need an internet connection.
At this point you should have three files under the resources directory: train.jsonl, dev.jsonl and test.jsonl

4. Activate the **nerel27** environment. Prepare the character vocabulary and move it to the resources directory by running:
```bash
python get_char_vocab.py train_dev resources/train.jsonl resources/dev.jsonl
mv char_vocab.train_dev.txt resources/
```

5. Download the fasttext word embeddings and unpack them
```bash
cd resources
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
gzip -dk cc.ru.300.vec.gz
```
6. (Optional) Filter word embeddings file, so we only have to load words, that are used in this experiment for RAM efficiency. Here I will provide some exemplary python code (you have to check your paths), that can be run in jupyter lab:
```python
from pathlib import Path
import json

vocab = set()
for split in ["train", "test", "dev"]:
    with open(f"resources/{split}.jsonl", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            vocab.update({word for sentence in data["sentences"] for word in sentence})

with Path("resources/cc.ru.300.vec").open(encoding='utf-8') as f, Path("resources/cc.ru.300.vec.filtered").open(encoding='utf-8', mode="w") as f2:
    for idx, line in enumerate(f):
        if idx == 0:
            continue
        data = line.strip().split()
        if data[0] in vocab:
            f2.write(line)
```
7. Check the `experiments.conf` file. The `nerel` and `nerel_ce16` should have correct paths (if you followed all the instructions thoroughly it will work as is)
8. Run training. Make sure, that you are in project's root directory `biaffine-ner` and the `nerel27` environment is activated:
```bash
python train.py nerel
```
and then another configuration:
```bash
python train.py nerel_ce16
```
After 40000 steps the process should stop.
Now you have your best checkpoints in the `logs` directory.
9. Evaluate the chosen configuration on the test set.
```bash
python evaluate.py nerel
```
The results will be printed out to stdout.

# Comments and notes

* The NEREL dataset initially has a character span based annotations. To prepare this dataset for this model, one needs to segment each example into sentences and tokens and then map char span anntotations to token span annotations.
* Some of NEREL entities may be "torn apart": one annotation for several char spans. This fact prevents one from using token span annotation based methods. That is why we had to split them into several annotations.
* Spacy's Russian sentence segmentation has its flaws. This may affect the training quality.
* Biaffine NER method allows for nested NER, so we solve this task in particular.
* Original code base did not work well with NEREL without following modifications:
  * `extract_bert_features/data.py` had a bug in the 31st line, that caused Index error.
  * [Original repo](https://github.com/juntaoy/biaffine-ner) version of `biaffine_ner_model.py` does not allow for Languae Model ablaition, while [Amir Zeldes's](https://github.com/amir-zeldes) one does. We had to change it back to TF1 and remove \__future__ imports.
  * `load_char_dict()` function (`util.py` line 44) treated char vocab too loosly: if there were several whitespaces, that occupied several positions in vocab file, that would lead to inconsistency in charcter index. Consequently, no training could be started because of tensor size mismatches. My modifications helped mitigate that.
* We did not employ contextualized word embeddings due to their high computational cost and their taking too long to be computed.

### Experiments

We trained the model during no more than standard 40000 steps variying only character embedding size.

### Results

| model\metric                                                                                   | P     | R     | F1    |
|------------------------------------------------------------------------------------------------|-------|-------|-------|
| This Biaffine NER + fasstext                                                                   | 77.79 | 53.09 | 63.10 |
| This Biaffine NER + fasstext (char emb size doubled)                                           | 77.94 | 54.30 | 64    |
| Biaffine NER + fasstext (reported in [original article](https://arxiv.org/pdf/2108.13112.pdf)) | 78.84 | 71.80 | 75.13 |

### Observations

* It looks like increasing character embedding size improves all the metrics. 
* We did not precisely reproduce the results, reported in [original article](https://arxiv.org/pdf/2108.13112.pdf), but we know, what to try next.

### Room for improvement
1. Train longer than 40000 steps. The loss in both experiments have not reached its plato yet.
2. Use contextualized embeddings from ruBERT.
  * We tried using multilingual BERT cased, but stumbled upon massive computational cost.
  * To effortlessly employ ruBERT the code base needs to be updated to use Tensor Flow 2. (There is a [version](https://github.com/amir-zeldes), that is claimed to be TF2 compliant, but it's not. It is still work-in-progress (I tried using it and ended up rewriting TF calls myself))
3. Perform a grid search on the following hyperparameters: `dropout_rate`, `lexical_droput_rate`, `char_embedding_size`, `filter_size`
4. Write a reverse data format converter to go from token level spans back to char level spans. This will allow assessing the performance of the chosen method objectively with regard to the original task.
5. Think of ways to unify preprocessing and inference code to deliver the pipeline to production. 
* Create two separate microservices: one for preprocessing and postprocessing, another (python 2.7) for model inference. (Drawbacks: the more services we have, the more hassle to maintain and monitor them)
* Rewrite the 2.7 codebase to 3.8 and be happy (Drawbacks: may take significantly more time to implement than just using the given 2.7 codebase as is)
6. Try to use a different sentence segmenter or write it manually to improve the segmentation quality.