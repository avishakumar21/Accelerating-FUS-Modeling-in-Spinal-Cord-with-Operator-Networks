# Neural Operators for FUS simulations

optimization of ultrasound neural implant placement with deeponet

To run the code 0. have data in data/, split in to train/, val/, test/. Have a result/ folder

1. Check config.json to set hyperparameters, input output directories and device (cpu or gpu)
2. in the root directory:

For CNN-DeepOnet, run

```
 python3 main_cnn.py <a_nane_for_this_trail_NOSPACE>
```


For ViT-DeepOnet, run

```
 python3 main_vit.py <a_nane_for_this_trail_NOSPACE>
```

For FNO,  run
```
 python3 main_fno.py <a_nane_for_this_trail_NOSPACE>
```
