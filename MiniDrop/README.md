```python
ython cnn/train.py -i /home/mabon/Tiny_power/datasets/power/xmega_unmasked/X1_K1_200k.npz -o /home/mabon/temp333 -m /home/mabon/Tiny_power/models/original/HW/xmega/X1_50k -e 150 -tb 2 -lm hw_model -tn 50000 -pr .99 -aw 1800_2800 -db True
```

```python
python cnn/train.py -i /home/mabon/Tiny_power/datasets/power/xmega_unmasked/X1_K1_200k.npz -o /home/mabon/temp333 -m /home/mabon/Tiny_power/models/custom_prunded/HW/xmega/X1/fpgm/h4 -e 150 -tb 2 -lm hw_model -tn 50000 -pr .5 -aw 1800_2800
```

```python

python cnn/new_train.py -i /home/mabon/Tiny_power/datasets/power/xmega_unmasked/X1_K1_200k.npz -o /home/mabon/temp5 -m /home/mabon/Tiny_power/models/original/HW/xmega/X1_50k -v -e 150 -tb 2 -lm hw_model -aw 1800_2800 -tn 10000 -rp /home/mabon/Tiny_power/Score/Xmega/HW/X1/fpgm/fpgm_idx.csv -CP True -CPP /home/mabon/Tiny_power/Score/CUSTOM/Xmega/HW/X1/fpgm/custum/N4/1-pr.csv 
```



```python
def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_dir', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-cd', '--cross_dev', action='store_true', help='')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='default value is 0')
    parser.add_argument('-tb', '--target_byte', type=int, default=2, help='default value is 2')
    parser.add_argument('-lm', '--network_type', choices={'hw_model', 'ID'}, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scailing'})
    parser.add_argument('-tn', '--max_trace_num', type=int, default=10000, help='')
    parser.add_argument('-sh', '--shifted', type=int, default=0, help='')
    parser.add_argument('-pr', '--pruning_rate', type=float, default=1, help='')
    parser.add_argument('-rp', '--ranks_path', help='')
    parser.add_argument('-CP', '--custom_pruning',default='False', help='')
    parser.add_argument('-CPP', '--custom_pruning_file',default='False', help='')

```
