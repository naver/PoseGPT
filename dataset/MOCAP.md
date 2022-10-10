# Preprocessing of MOCAP data

## AMASS

Preprocessing for SMPLH/SMPLX format
```
# Debug
python dataset/preprocessing/amass.py "prepare_annots(debug=False,type='smplx',split='train',num_workers=0,n_max=1000)"

# Jobs
type=( "smplh" "smplx" )
split=( "train" "val" "test")
for x in "${type[@]}"
do
  for y in "${split[@]}"
  do
    sbatch -p cpu bash.sh dataset/preprocessing/amass.py "prepare_annots(type=${x}, split=${y}, debug=False, seq_len=64, min_seq_len=16, overlap_len=0)"    
  done 
done
```
That will create data with the following structure:
```
/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/<type>/amass/<split>/<name>/pose.pt
                                                                                               valid.pt    
```


## BABEL

Trimmed sequences preprocessing in SMPLH/SMPLX format.
```
# Debug
python dataset/preprocessing/babel.py "prepare_annots_trimmed(type='smplx',debug=False,num_workers=0,n_max=100,save_into_array=True)"
python dataset/preprocessing/babel.py "prepare_annots_trimmed(type='smplx',debug=False,num_workers=0,n_max=1000,save_into_array=False,seq_len=900)"
python dataset/preprocessing/babel.py "prepare_annots_trimmed(type='smpl',debug=False,num_workers=0,n_max=1000,save_into_array=False,seq_len=900)"

# Jobs for creating list of tensor
type=( "smplx" "smpl" )
type=( "smpl" )
split=( "train" "val" )
for x in "${type[@]}"
do
  for y in "${split[@]}"
  do
    sbatch -p cpu bash.sh dataset/preprocessing/babel.py "prepare_annots_trimmed(type='${x}', split='${y}', debug=False, seq_len=900, min_seq_len=16, overlap_len=0, save_into_array=False)"  
  done
done

```

That will create data with the following structure:
```
/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/<type>/babel_trimmed/<split>_<n_actions>/<name>/pose.pt
                                                                                                                   valid.pt
                                                                                                                   idx.pt
                                                                                                                   action.pt    
```

Untrimmed sequences preprocessing in SMPLH/SMPLX format.
```
# Debug
python dataset/preprocessing/babel.py "prepare_annots_untrimmed(type='smplx',debug=False,num_workers=0,n_max=1000)"

# Jobs
type=( 'smplx' )
split=( 'train' 'val' )
for x in "${type[@]}"
do
  for y in "${split[@]}"
  do
    echo ''
    sbatch -p cpu bash.sh dataset/preprocessing/babel.py "prepare_annots_untrimmed(type='${x}', split='${y}', debug=False, seq_len=900, min_seq_len=16, overlap_len=0, num_workers=8)"  
  done 
done

```

## GRAB
Preparing SMPLH/SMPLX format
```
# Debug
ipython dataset/preprocessing/grab.py -- "prepare_annots(debug=True)"

# Jobs
sbatch -p cpu bash.sh dataset/preprocessing/grab.py "prepare_annots(split='train', debug=False, seq_len=10000, min_seq_len=16, overlap_len=0, handle_object=False)"
sbatch -p cpu bash.sh dataset/preprocessing/grab.py "prepare_annots(split='val', debug=False, seq_len=10000, min_seq_len=16, overlap_len=0, handle_object=False)"
sbatch -p cpu bash.sh dataset/preprocessing/grab.py "prepare_annots(split='test', debug=False, seq_len=10000, min_seq_len=16, overlap_len=0, handle_object=False)"
sbatch -p cpu bash.sh dataset/preprocessing/grab.py "prepare_annots(split='test', debug=False, seq_len=10000, min_seq_len=16, overlap_len=0, n_max=5, handle_object=False)"

# Debug rotation
ipython dataset/preprocessing/grab.py -- "rotate_scene()"

```

## HumanAct12

Preprocessing data
```
ipython dataset/preprocessing/humanact12.py -- "prepare_annots()"
```

## Mocap dataloader
```
# Debug
ipython dataset/mocap.py -- "test_mocap_dataset()"
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smpl/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')"
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smplx/babel_untrimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')"
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smplx/grab/test/seqLen10000_fps30_overlap0_minSeqLen16')"

# Teaser figure
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smpl/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16', seq_len=110, n_max=2, ry=np.pi, freq=20, color_start=[114,188,212], color_end=[114,188,212])" && mv img.jpg entire_blue.jpg && 
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smpl/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16', seq_len=110, n_max=2, ry=np.pi, freq=20)" && mv img.jpg entire.jpg && 
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smpl/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16', seq_len=110, n_max=2, ry=np.pi, freq=20, end_rendering=55)" && mv img.jpg start.jpg && 
ipython dataset/mocap.py -- "test_mocap_dataset(data_dir='data/smpl/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16', seq_len=110, n_max=2, ry=np.pi, freq=20, start_rendering=55)" && mv img.jpg end.jpg
```

## Glove

```
ipython dataset/preprocessing/babel.py -- "glove()"
```


