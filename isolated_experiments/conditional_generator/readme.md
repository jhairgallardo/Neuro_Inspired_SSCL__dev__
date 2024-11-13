```bash 
# for transformer-based feature_transfer module
python -m torch.distributed.launch \
--nproc_per_node=4 main.py \
--fixed_color_jitter_order \
--trans_loss \
--num_blocks 1 \
-b 128 \
--epochs 80 \
--lr 1e-3 \
--arch vit \
--dim_feedforward 256 \
--num_layer 2 \
--nhead 4; 
```

```bash
# for resnet-like feature_transfer module
python -m torch.distributed.launch \
--nproc_per_node=4 main.py \
--fixed_color_jitter_order \
--trans_loss \
--num_blocks 1 \
-b 128 \
--epochs 80 \
--lr 1e-3 \
--arch res \
--action_channels 32 \
--num_layer 1; 
```

These two configurations achieve small reconstruction loss while keeping the reconstruction module small. 
I found the transformer-based feature transfer module works better. 
It not only achieves smaller loss on random-crop but also provides better reconstruction images (visually).

- random flip is automatically disabled, unless use `--random_flip`
- `--fixed_color_jitter_order` will simply the color jitter action code (remove 4 digits representing order of operation)
- model definition: 
  - `main.py` line 180-195; which defines: 
    - Feature Transfer Model; `FeatureTransModelNew` or `ActionConditionedTransformer` in `model.py`. Which takes the anchor embedding and action code to prediction the augmented embedding.
    - Decoder (ie, unconditional generator); `ResNet18Dec` in `model.py`. Which takes the embedding and generate image.
- model training:
  - `train_one_epoch` in `main.py` (line 339-386); note all 4 losses are used, which showed highest performance in exp. 
- dataset definition:
  - `aug_with_action_code` and `ActionCodeDataset` in `action_code.py`. Which output original image, augmented image, corresponding action code, and label.