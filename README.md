### Publication

@article{papakostas2017deep,
  title={Deep visual attributes vs. hand-crafted audio features on multidomain speech emotion recognition},
  author={Papakostas, Michalis and Spyrou, Evaggelos and Giannakopoulos, Theodoros and Siantikos, Giorgos and Sgouropoulos, Dimitrios and Mylonas, Phivos and Makedon, Fillia},
  journal={Computation},
  volume={5},
  number={2},
  pages={26},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}


# CNNs-AUDIO-EMOTION-RECOGNITION

**Train**: 

python trainCNN.py <path_to_model_architecture> <path_to_training_images> <path_to_validation_images> <output_model_name> <number_of_iterations> 

*For more training options run : python trainCNN.py -h
  
Example:

python trainCNN.py Structures/Emotion_Gray_14.prototxt EmotionFinalDataset/savee_specs_byspeaker/s1/train/ EmotionFinalDataset/savee_specs_byspeaker/s1/test/ BySpeakerS1 5000 --base_lr 0.001 --solver_mode GPU --batch_size 64 --snapshot 10000 --test_interval 500 --stepsize 600 --display 250 --input_size 250

**Test**: 

python ClassifyWavGrayCORRECT.py evaluate <path_to_test_wavs> <trained_model_name>.caffemodel <classification_method> <color_images:0, grayscale:1> "" <model_id> rectangular_image_size


Example:

python ClassifyWavGrayCORRECT.py evaluate EmotionFinalDataset/savee_wavs_byspeaker/s4/test/ BySpeakerS4_iter_5000.caffemodel cnn 0 "" Emovo 250


Dependencies:
1) Caffe-Depp Learning library
3) [PyAudioAnalysis Library](https://github.com/tyiannak/pyAudioAnalysis)
2) Open-CV and Python 2.7


### Publication

@article{papakostas2017deep,
  title={Deep visual attributes vs. hand-crafted audio features on multidomain speech emotion recognition},
  author={Papakostas, Michalis and Spyrou, Evaggelos and Giannakopoulos, Theodoros and Siantikos, Giorgos and Sgouropoulos, Dimitrios and Mylonas, Phivos and Makedon, Fillia},
  journal={Computation},
  volume={5},
  number={2},
  pages={26},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}

