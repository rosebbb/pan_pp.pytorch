1. Run inference:
    a. python inference.py 
        Config: Modify `args.config = 'config/pan/pan_r18_accessmathtestontrain_finetune.py'`
        Model: Modify `args.checkpoint = '/data/Projects/pan_pp.pytorch/checkpoints/pan_r18_accessmath_finetune/pan_r18_accessmath_finetune.pth.tar'`


2. Add datasets;
    1. Add .py file for this dataset in `dataset/pan` or `dataset/pan_pp` of `dataset/psenet`
    2. Add config file in in `config/pan` or `config/pan_pp` of `config/psenet`
    3. Modify `dataset/pan/__init__.py`, add `from .pan_accessmath import PAN_ACCESSMATH`

 
