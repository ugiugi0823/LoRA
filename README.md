# LoRA



## Install Requirements
```
git clone https://github.com/ugiugi0823/LoRA.git
```

## By conda
```
conda env create -f environment.yaml
conda activate nd
```
## OR by pip
```
pip install -r requirements.txt
```


## Download model and data
### model weight
```
wget -O animesfw.tgz https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animesfw.tgz
mkdir <your home dir>/LoRA/model <your home dir>/LoRA/dataset
tar -C <your home dir>/LoRA/model -zxvf animesfw.tgz
```
### dataset
```
wget -O mmk.tgz https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/mmk.tgz
tar -C <your home dir>/LoRA/dataset -zxvf mmk.tgz
sed -i 's|img_path:.*|img_path: <your home dir>/LoRA/dataset|' <your home dir>/LoRA/train.yaml
```


## Experiments

Train [LoRA](https://arxiv.org/abs/2106.09685)

```bash
python trainer.py --config experiment/lora.yaml

## extract 
python experiment/extract_lora.py --src last.ckpt
```

Train [LoCon](https://github.com/KohakuBlueleaf/LoCon)

```bash
python trainer.py --config experiment/locon.yaml

## extract 
python experiment/extract_lora.py --src last.ckpt
```

Train [Textual Inversion](https://textual-inversion.github.io)

```bash
python trainer.py --config experiment/textual_inversion.yaml
```

Convert any checkpoint to safetensors
```bash
python scripts/sd_to_safetensors.py --src input.ckpt --dst output.safetensors
```



## Special appreciation
https://github.com/Mikubill/naifu-diffusion


