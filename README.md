# KPF BERT

- original repo: https://github.com/KPFBERT/kpfbert

## 사용방법

### Load

```python
# from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizerFast, BertModel

model_name_or_path = "jinmang2/kpfbert"
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
# model = AutoModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
model = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
```

### Tokenizer
```python
>>> text = "언론진흥재단 BERT 모델을 공개합니다."
>>> tokenizer.tokenize(text)
['언론', '##진흥', '##재단', 'BE', '##RT', '모델', '##을', '공개', '##합니다', '.']
>>> encoded_input = tokenizer(text)
>>> encoded_input
{'input_ids': [2, 7392, 24220, 16227, 28024, 21924, 7522, 4620, 7247, 15801, 518, 3],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

### Model Inference
```python
>>> import torch
>>> model.eval()
>>> pt_encoded_input = tokenizer(text, return_tensors="pt")
>>> model(**pt_encoded_input, return_dict=False)
(tensor([[[ 0.0700, -3.3552, -1.0075,  ..., -1.0417,  0.4648,  0.0141],
          [ 0.0124, -0.1968, -0.3717,  ...,  1.1253,  0.3683,  0.5172],
          [-0.1394, -0.3788,  0.7651,  ...,  1.0516, -1.5133, -0.5443],
          ...,
          [-0.9051,  0.7249, -0.5570,  ...,  1.0029, -1.7010, -1.0771],
          [-0.1018, -0.0725, -0.1560,  ...,  0.7753, -0.0967,  0.6251],
          [ 0.0700, -3.3551, -1.0078,  ..., -1.0416,  0.4646,  0.0136]]],
          grad_fn=<NativeLayerNormBackward>), None)
```

## 총 5개의 모델에 대해서 평가 작업 수행

* kpfBERT base (https://github.com/KPFBERT/kpfbert)
* KLUE BERT base (https://huggingface.co/klue/bert-base)
* ETRI BERT base (KorBERT, https://aiopen.etri.re.kr/service_dataset.php)
* KoBERT (https://github.com/SKTBrain/KoBERT)
* BERT base multilingual cased (https://huggingface.co/bert-base-multilingual-cased)

### Sequence Classification 성능 측정 결과 비교 (10/22/2021):

| 구분 | NSMC | KLUE-NLI | KLUE-STS |
| :---       |     :---      |     :---      |    :---     |
| 데이터 특징 및 규격 | 영화 리뷰 감점 분석, 학습 150,000 문장, 평가: 50,000문장 | 자연어 추론, 학습: 24,998 문장 평가: 3,000 문장 (dev셋) | 문장 의미적 유사도 측정, 학습: 11,668 문장 평가: 519 문장 (dev셋) |
| 평가방법   | accuracy     | accuracy    | Pearson Correlation    |
| KPF BERT     | 91.29%       | 87.67%    | 92.95%      |
| KLUE BERT     | 90.62%       | 81.33%    | 91.14%      |
| KorBERT Tokenizer | 90.46%      | 80.56%    | 89.85%     |
| KoBERT     | 89.92%       |  79.53%    | 86.17%      |
| BERT base multilingual    | 87.33%       | 73.30%    | 85.66 %    |

### Question Answering 성능 측정 결과 비교 (10/22/2021):

| 구분 | KorQuAD v1 | KLUE-MRC |
| :---       |     :---      |      :---       |
| 데이터 특징 및 규격 | 기계독해, 학습: 60,406 건 평가: 5,774 건 (dev셋) | 기계독해, 학습: 17,554 건 평가: 5,841 건 (dev셋) |
| 평가방법   | Exact Match / F1 | Exact Match / Rouge W |
| KPF BERT     | 86.42% / 94.95% | 69.51 / 75.84% |
| KLUE BERT     | 83.84% / 93.23% | 61.91% / 68.38% |
| KorBERT Tokenizer | 20.11% / 82.00% | 30.56% / 58.59% |
| KoBERT     | 16.85% / 71.36% | 28.56% / 42.06 % |
| BERT base multilingual    | 68.10% / 90.02% | 44.58% / 55.92% |

## KPF BERT 활용 사례

* KPFBERTSUM (https://github.com/KPFBERT/kpfbertsum)
  - KpfBertSum은 Bert 사전학습 모델을 이용한 텍스트 요약 논문 및 모델인 PRESUMM모델을 참조하여 한국어 문장의 요약추출을 구현한 한국어 요약 모델이다.
  - 한국언론진흥재단에서 구축한 방대한 뉴스기사 코퍼스로 학습한 kpfBERT를 이용하여 특히 뉴스기사 요약에 특화된 모델이다.

* YouTube 'BERT란 무엇인가' 설명 링크 https://youtu.be/Pj6563CAnKs
