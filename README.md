# fairseq_tutorial
fairseq tutorial repo
***
## 1. download fairseq
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
## 2. download sample data
```
cd fairseq/examples/translation
bash prepare-wmt14en2de.sh
cd ../..
```
original tutorial say 'bash prepare-iwslt14.sh'
However, It cannot be downloaded anymore. 
So, We use wmt14en2de

## 3. prepare data for train
```
TEXT=examples/translation/wmt14_en_de
python preprocess.py --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0
```

## 4. train model
```
mkdir -p checkpoints/fconv_wmt_en_de
python train.py data-bin/wmt14_en_de \
  --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_de --save-dir checkpoints/fconv_wmt_en_de
```

## 5. test model
```
python generate.py data-bin/wmt14_en_de \
  --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt --beam 5 --remove-bpe
```
