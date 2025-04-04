# Reproducing Results
Scripts accompanying the paper ***`confopt`** : A Library for Implementation and Evaluation of Gradient-based One-Shot NAS Methods* ([AutoML'2025](https://2025.automl.cc/))

### Architecture Search on the Benchsuite

There are 3 subspaces in benchsuite (`WIDE`, `DEEP` and `SINGLE_CELL`) and 3 operation sets we utilize (`REGULAR`, `NO_SKIP` and `ALL_SKIP`). 

For example, to run the experiments for subspace `WIDE` with opset `REGULAR`, 

##### DARTS &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "darts" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag wide-regular-darts
```

##### DrNAS &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "drnas" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag "wide-regular-drnas" 
```

##### GDAS &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "gdas" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag "wide-regular-gdas"
```

##### OLES &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "darts" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag "wide-regular-oles" --oles
```

##### PC-DARTS &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "darts" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag "wide-regular-pcdarts" --pcdarts
```

##### SmoothDARTS &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "darts" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag "wide-regular-sdarts" --sdarts "random"
```

##### FairDARTS &#8594;
```bash
python scripts/benchsuite_experiments/supernet_search.py --optimizer "darts" --subspace "wide" --ops "regular" --dataset "cifar10_supernet" --seed 0 --tag "wide-regular-fairdarts" --fairdarts
```

For other benchmarks, one can provide options **`deep`** and **`single_cell`** for `--subspace`, &  **`no_skip`** and **`all_skip`** for `--ops`. If you want to log your run on **WandB**, simply provide `--log_with_wandb` argument with the scripts above.


### Retraining the architecture

Once you obtain searched genotypes, you can train them on the 9 Hyperparameter sets defined in the paper. You can also take a look at the found genotypes in this [link](https://drive.google.com/drive/u/1/folders/1sJrWQcQTfdmsYmm4bLwfSqv_AuCSGwhv). 


For example, if one wants to test the genotype obtained from running **`GDAS`** with **`DEEP`** subspace and **`NO_SKIP`** opset on the Hyperparameter set (HP) **1**, 

```bash 
python scripts/benchsuite_experiments/train_genotype.py --dataset "cifar10_model" --optimizer "gdas" --other "baseline" --subspace "deep" --opset "no_skip" --epochs 300 --seed 0 --hpset 1 --genotypes_folder "genotypes" --tag "retrain_arch"
```

Please use **`--hpset`** (vary from 1-9) for training with the respective hyperparameter set.


> Note: 
> - For Fair-DARTS, SmoothDARTS, OLES, and PC-DARTS, one has to change the **`--other`** option to **`fairdarts`**, **`sdarts`**, **`oles`** and **`pcdarts`** respectively, with **`--optimizer darts`**.
> - Make sure that **`genotypes`** folder exist in the path.