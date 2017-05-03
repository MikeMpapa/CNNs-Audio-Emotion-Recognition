
from scipy.io import loadmat
from pyAudioAnalysis import audioBasicIO as io, audioFeatureExtraction as aF, audioSegmentation
import hmmlearn.hmm
import sys, os, glob, csv
import cPickle
import random
import string
import numpy as np, scipy, matplotlib, Image
import matplotlib.pyplot as plt
import time
import cv2
#Load Caffe library
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def initialize_transformer(input_size,SingleFrame_net):
  ''' 
  shape = (input_size, input_size) #input_frames x num_of_chanels x im_Height x im_Width , 10 is required for oversampling
  transformer = caffe.io.Transformer({'data': shape}) #set data input size
  channel_mean = np.zeros((1))
  channel_mean[0] = 103.939
  print channel_mean.shape,"MEAN", shape
  transformer.set_mean('data',channel_mean) #set data mean from training
  transformer.set_raw_scale('data', 255) #define that the input is 0-255
  '''

  # load input and configure preprocessing
  channel_mean = np.zeros((3,input_size,input_size))
  channel_mean[:,:,:] = 103.939
  print SingleFrame_net.blobs['data'].data.shape,"ppppppppp"
  transformer = caffe.io.Transformer({'data': SingleFrame_net.blobs['data'].data.shape})
  transformer.set_mean('data', channel_mean)
  transformer.set_transpose('data', (2,0,1))
  transformer.set_channel_swap('data', (2,1,0))
  transformer.set_raw_scale('data', 255.0)
 
  return transformer

def loadCNN(caffeModelName, model_structure_id, input_size):




    singleFrame_model = 'Structures/Emotion_Gray_'+str(model_structure_id)+'_deploy.prototxt'
    
    SingleFrame_net =  caffe.Net(singleFrame_model, caffeModelName, caffe.TEST)
                
    SingleFrame_net.blobs['data'].reshape(1,3,input_size,input_size)

    #INitialize input image transformer
    input_transformer = initialize_transformer(input_size,SingleFrame_net)

    classNamesFileName = caffeModelName
    classNamesFileName = classNamesFileName[0: classNamesFileName.find("_iter_")] + "_classNames"    
    classNamesAll = cPickle.load(open(classNamesFileName, 'rb'))
    classNamesAll = [c.lower() for c in classNamesAll]
    return SingleFrame_net, input_transformer.mean, input_transformer, classNamesAll



def computePreRec(CM, classNames):
    numOfClasses = CM.shape[0]
    if len(classNames) != numOfClasses:
        print "Error in computePreRec! Confusion matrix and classNames list must be of the same size!"
        return
    Precision = []
    Recall = []
    F1 = []    
    for i, c in enumerate(classNames):
        Precision.append(CM[i,i] / (np.sum(CM[:,i])+0.00001)) 
        Recall.append(CM[i,i] / (np.sum(CM[i,:])+0.00001))
        F1.append( 2 * Precision[-1] * Recall[-1] / (Precision[-1] + Recall[-1]+0.00001))
    return Recall, Precision, F1



def singleFrame_classify_video(signal, SingleFrame_net, transformer, with_smoothing, classNamesCNN,input_size):
    input_images = []
    output_classes = []
    input_im = caffe.io.load_image(signal.replace(".wav",".png")) 
    print input_im.shape
    #input_im = input_im[:,:,0] 
    #input_im = np.expand_dims(input_im, axis=3)    
    #input_images.append(input_im)
    #clip_input = caffe.io.oversample(input_images,[input_size,input_size])
    #Grayscale = rgb2gray(input_im)
    '''
    print Grayscale.shape
    Grayscale = np.expand_dims(Grayscale, axis=3)    
    #Grayscale = 
    input_images.append(Grayscale)
    print len(input_images)
    #clip_input = caffe.io.oversample(input_images,[input_size,input_size])
    print np.asarray(input_images).shape
    #caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32) #initialize input matrix
    #print caffe_in.shape
    caffe_in = transformer.preprocess('data',Grayscale) # transform input data appropriatelly and add to input matrix        
    out = net.forward_all(data=caffe_in)
    print out
    output_predictions= np.mean(out['probs'],0) #predict labels       
    iMAX = output_predictions.argmax()
    prediction = classNamesCNN[iMAX]
    print prediction
    '''
    SingleFrame_net.blobs['data'].reshape(1,3,input_size,input_size)
    #im = caffe.io.load_image('examples/images/cat.jpg')
    SingleFrame_net.blobs['data'].data[...] = transformer.preprocess('data', input_im)

#compute
    out = SingleFrame_net.forward()
    output_predictions = np.zeros((len(input_images),5))
    output_predictions= np.asarray(out['probs']) #predict labels   
    #print output_predictions    
    iMAX = output_predictions.argmax()
# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
    prediction = classNamesCNN[iMAX]
    #print prediction
    output_classes.append(prediction)


    #sys.exit()

    #os.remove(signal.replace(".wav",".png"))    
    #Initialize predictions matrix                
   # output_predictions = np.zeros((len(input_images),5))
   # output_classes = []
    #print [method for method in dir(net) if callable(getattr(net, method))]    
    #out = net.forward_all(data=input_im) #feed input to the network
    
    #print out.shape
    #output_predictions= np.mean(out['probs'],0) #predict labels       
    #iMAX = output_predictions.argmax()
    #prediction = classNamesCNN[iMAX]
    #output_classes.append(prediction)
    '''
    for i in range(0,len(input_images)):        
        # print "Classifying Spectrogram: ",i+1         
        #clip_input = input_images[i:min(i+batch_size, len(input_images))] #get every image -- batch_size==1
        #clip_input = caffe.io.oversample(clip_input,[input_size,input_size]) #make it 227x227        
        caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32) #initialize input matrix
        for ix, inputs in enumerate(clip_input):
            caffe_in[ix] = transformer.preprocess('data',inputs) # transform input data appropriatelly and add to input matrix        
        net.blobs['data'].reshape(caffe_in.shape[0], caffe_in.shape[1], caffe_in.shape[2], caffe_in.shape[3]) #make input caffe readable        
        out = net.forward_all(data=caffe_in) #feed input to the network
        output_predictions[i:i+batch_size] = np.mean(out['probs'].reshape(10,caffe_in.shape[0]/10,5),0) #predict labels        
        
        #Store predicted Labels without smoothing        
        iMAX = output_predictions[i:i+batch_size].argmax(axis=1)[0]
        prediction = classNamesCNN[iMAX]
        output_classes.append(prediction)
        #print "Predicted Label for file -->  ", signal.upper() ,":",    prediction
    '''


    return output_classes, output_predictions



def mtCNN_classification(signal, Fs, mtWin, mtStep, SingleFrame_net, channel_mean, input_transformer, classNamesCNN, input_size):
    mtWin2 = int(mtWin * Fs)
    mtStep2 = int(mtStep * Fs)
    stWin = 0.040
    stStep = 0.005    
    N = len(signal)
    curPos = 0
    count = 0
    fileNames = []
    flagsInd = []
    Ps = []
    randomString = (''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5)))
    while (curPos < N): 
        N1 = curPos
        N2 = curPos + mtWin2 + stStep*Fs
        if N2 > N:
            N2 = N
        xtemp = signal[int(N1):int(N2)]    
        #print xtemp.shape
        #print xtemp.shape[0]            # get mid-term segment        
        if xtemp.shape[0] < 8000:
            curPos += mtStep2               
            count += 1  
            continue 
        specgram, TimeAxis, FreqAxis = aF.stSpectogram(xtemp, Fs, round(Fs * stWin), round(Fs * stStep), False)     # compute spectrogram
        specgram = cv2.resize(specgram,(input_size, input_size), interpolation = cv2.INTER_LINEAR)     
        #specgram = scipy.misc.imresize(specgram, float(input_size) / float(specgram.shape[0]), interp='bilinear')        # resize to 227 x 227
        if specgram.shape[0] != specgram.shape[1]:  
            break
        #print specgram.shape
        #specgram = scipy.misc.imresize(specgram, float(input_size) / float(specgram.shape[0]), interp='bilinear')        # resize to 227 x 227
        #print specgram.shape
        
       # imSpec = Image.fromarray(np.uint8(matplotlib.cm.jet(specgram)*255))                                         # create image        
        curFileName = randomString + "temp_{0:d}.png".format(count)
        fileNames.append(curFileName)    
        #imSpec = rgb2gray(np.uint8(matplotlib.cm.jet(specgram)*255))
        imSpec = Image.fromarray(np.uint8(matplotlib.cm.jet(specgram)*255))
        scipy.misc.imsave(curFileName, imSpec)
        T1 = time.time()
        output_classes, outputP = singleFrame_classify_video(curFileName, SingleFrame_net, input_transformer, False, classNamesCNN,input_size)        
        T2 = time.time()
        os.remove(curFileName)    
        #print T2 - T1
        #flagsInd.append(classNamesCNN.index(output_classes[0]))        
        Ps.append(np.copy(outputP[0]))
        #print flagsInd[-1]
        curPos += mtStep2               
        count += 1      
    return np.array(flagsInd), classNamesCNN, np.array(Ps)




def evaluateEmotion(fileName, modelName, input_size, SingleFrame_net, channel_mean, input_transformer, classNamesCNN,  method = "svm", postProcess = 0, postProcessModelName = "", PLOT = False):        
        GTlabel = fileName.split('/')[-2].lower()
        if method == "cnn":
              WIDTH_SEC = 2.0    
              [Fs, x] = io.readAudioFile(fileName)
              x = io.stereo2mono(x)
              [flagsInd, classesAll, CNNprobs] = mtCNN_classification(x, Fs, WIDTH_SEC, 1.0, SingleFrame_net, channel_mean, input_transformer, classNamesCNN, input_size)            
        print CNNprobs
        CNNprobs = np.mean(CNNprobs, axis=0)
        print CNNprobs
        #sys.exit()
        PredLabelInd = np.argmax(CNNprobs)
        PredLabel = classesAll[PredLabelInd]
      
        GTlablInd = classesAll.index(GTlabel)
             
        print GTlabel,"---",PredLabel

        return GTlabel,GTlablInd,PredLabel ,PredLabelInd, classesAll





def main(argv):    
    if argv[1] == "evaluate":
        model_structure_id = argv[7]
        input_size = int(sys.argv[8]) #CNNinput 
        
        SingleFrame_net, channel_mean, input_transformer, classNamesCNN= loadCNN(argv[3],model_structure_id, input_size)                                    # load the CNN
        if os.path.isfile(argv[2]):  
            CM, classesAll = evaluateEmotion(argv[2], argv[3], argv[4], int(argv[5]), argv[6], True)
            print CM
        elif os.path.isdir(argv[2]):    
           
            dirsToClassify = [os.path.join(argv[2], d) for d in os.listdir(argv[2]) if os.path.isdir(os.path.join(argv[2], d))]
            print dirsToClassify
            CM = np.zeros((len(dirsToClassify),len(dirsToClassify)))
            for d in dirsToClassify: 

             types = ('*.wav', )
             wavFilesList = []
             for files in types:
                 wavFilesList.extend(glob.glob(os.path.join(d, files)))    
             wavFilesList = sorted(wavFilesList)                 
             modelName = argv[3]
             method = argv[4]
             postProcess = int(argv[5])                        
             postProcessModelName = argv[6]

             for ifile, wavFile in enumerate(wavFilesList):    
                 print "{0:s}, {1:d} file of {2:d}".format(wavFile, ifile+1, len(wavFilesList))
                 gtLabel, gtLabel_id, predLabel, predLabel_id, classesAll  = evaluateEmotion(wavFile, modelName, input_size, SingleFrame_net, channel_mean, input_transformer, classNamesCNN ,method, postProcess, postProcessModelName)
                 CM[gtLabel_id,predLabel_id]+=1      
                 
             

        
            [RecAll, PreAll, F1All] = computePreRec(CM, classesAll)        
            print CM    
            CM = CM / np.sum(CM)
            print CM
            print "Based on overall CM"
            print "{0:s}\t{1:s}\t{2:s}\t{3:s}".format("", "Rec", "Pre", "F1")
            for ic, c in enumerate(classesAll):
                print "{0:s}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(c, 100*RecAll[ic], 100*PreAll[ic], 100*F1All[ic])                            
            
            print "Average (duration-irrelevant)"
            print "mean Accuracy %.2f" % np.trace(CM)
            print "mean Recall %.2f" % np.asarray(RecAll).mean()
            print "mean Precision %.2f" % np.asarray(PreAll).mean()
            print "mean F1 %.2f" % np.asarray(F1All).mean()
            
            #SAVE RESULT
            testDataset = argv[2].split('/')[-3].split('_')[0].upper()
            np.save('Results/CM_'+('_').join(( modelName.split('_')[0:-2] ))+'_on'+testDataset, CM)
            f1 = np.asarray(F1All).mean()
            results = 'Results/F1_'+('_').join(( modelName.split('_')[1:-2] ))
            if os.path.exists(results+'.npy'):
                finalF1 = np.load(results+'.npy')
                next_index = np.where(finalF1==-1)
                f1_row = next_index[0][0]
                f1_col = next_index[1][0] 
                finalF1[f1_row,f1_col] = f1
                np.save(results, finalF1)

            else:    
                finalF1 =  np.zeros((4,4))-1
                finalF1[0,0] = f1
                np.save(results, finalF1)
           
if __name__ == '__main__':
    main(sys.argv)