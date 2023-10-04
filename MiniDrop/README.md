# Structured Pruning Based on Fixed Pruning Rate
Use this script to prune mode based on fppm/l2nor score and a fixed pruning rate.

```python
python cnn/new_train.py -i <input to .npz file for trace> -o < Finetured model output directory>
-m <original model directory for initial weights > -e <epochs to train> -tb <target byte>
-lm <leakage model (hw_model/ID) -tn <number of training traces> -pr <prurning rate>
-aw <attack window> -rp <rank paths (fpgm/l2) -CP <Custom pruinng. Set to false for structured pruning>
```

# Automatic Pruning Based on Fixed Pruning Rate
Use this script to prune mode based on fppm/l2nor score and our alogorith Minidron.


```python

python cnn/new_train.py -i <input to .npz file for trace> -o < Finetured model output directory>
-m <original model directory for initial weights > -e <epochs to train> -tb <target byte>
-lm <leakage model (hw_model/ID) -tn <number of training traces> -aw <attack window> 
-rp <rank paths (fpgm/l2) -CP <Custom pruinng. Set to TRUE for Minidrop>
-CPP <Path to Minidrop Scroes>
```


# Input parameters to new_train.py

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
