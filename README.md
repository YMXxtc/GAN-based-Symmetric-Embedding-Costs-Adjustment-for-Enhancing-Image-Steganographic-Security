# GAN-based-Symmetric-Embedding-Costs-Adjustment-for-Enhancing-Image-Steganographic-Security

## 训练 training
1. run **example_04bpp_szu.m** in HILL to genearate original cost map and original probability map 
   
matlab -nosplash -nodesktop -r "example_04bpp_szu"

2. run **stegan_stc_23_szu_getC1cost_hill.py** in src to embed the first sub-image
   
python stegan_stc_23_szu_getC1cost_hill.py --config 'train result' --datacover 'path to save dataset' --costroot 'path to cost' --root 'root to save'

3. run **main.py** in src to train the GAN model
   
python main.py --dataroot 'path to cover with the first sub-image being embedded' --coverroot 'path to original cover dataset' --pmaproot 'path to original probability map' --outf 'folder to output' --config 'model_name'


## 测试 testing
1. run **example_04bpp.m** in HILL to genearate original cost map and original probability map
   
matlab -nosplash -nodesktop -r "example_04bpp"

2. run **stegan_stc_23_boss_getC1cost_hill.py** in src to embed the first sub-image
   
python stegan_stc_23_boss_getC1cost_hill.py --config 'train result' --datacover 'path to save dataset' --costroot 'path to cost' --root 'root to save'

3. run **stegan_stc_23_1102_ResUNet1DBconf3_learnp_C2adjust_hill.py** to embed the second sub-image
   
python stegan_stc_23_1102_ResUNet1DBconf3_learnp_C2adjust_hill.py --netG 'path to netG' --config --datacover 'path to cover with the first sub-image being embedded' --coverroot 'path to original cover dataset' --pmaproot 'path to original probability map' --root 'folder to output' --config 'name'