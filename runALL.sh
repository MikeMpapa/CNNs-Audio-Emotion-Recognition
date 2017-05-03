#!/bin/bash


#DATA ROOT FOLDER
data='EmotionFinalDataset'
#data='EmotionFinalDataset_original'
#data='EmotionFinalDataset_Balanced'
#data='EmotionFinalDataset_original_balanced'


#CNN PARAMETERS
model_structure_id=15
input_size=227
net="Structures/Emotion_Gray_$model_structure_id.prototxt"
prefix='FinalTest_15_64batch_5000_augmented_c17c27c35c45c53'

maxiter='5000'

snapshot='--snapshot 10000'
test_interval='--test_interval 500'
step_size='--stepsize 600'
batch_size='--batch_size 64'
base_lr=' --base_lr 0.001'
display='--display 250'
input_size_s='--input_size '

#test_iter='--test_iter 60'
#init_type='--init_type fin'
#init='--init InitializationModels/SM_gray_suffle_noaug_200x200_smallCNN_iter_4500.caffemodel'
#lr_type='--type AdaDelta'
solver_mode='--solver_mode GPU'

declare -a dataset_list=("Emovo" "Savee" "German" "Movies")

for i in {0..3} 
do
    python trainCNN.py $net $data/${dataset_list[$i],,}_specs/train/ $data/${dataset_list[$i],,}_specs/'test'/ ${dataset_list[$i]}_$prefix $maxiter $init $init_type $test_iter $base_lr $solver_mode $batch_size $snapshot $test_interval $step_size $lr_type $display $input_size_s$input_size
    for j in {0..3} 
    do
        if [ "${dataset_list[$j]}" == "${dataset_list[$i]}" ]; then
            totest='test'
            #totest='train'
        else
            totest='train'
        fi
        python ClassifyWavGrayCORRECT.py evaluate $data/${dataset_list[$j],,}_wavs/${totest}/ ${dataset_list[$i]}_${prefix}_iter_${maxiter}.caffemodel  cnn 0 "" $model_structure_id $input_size
    done
done

#TODO
#batch=1
#bigger cnn
#ims 200x200
