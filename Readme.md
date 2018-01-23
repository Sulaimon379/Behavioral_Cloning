The aim of the project is to develop a deep learning based algorithm to simulate an autonomous driving vehicle. Data was collected by driving the simulator. The data is collected in a csv file which contains the link to images (center, left, right), the steering angles, throttle, 	brake, and speed data are collected. 
The data collection is the most important part of the project, since an inaccurate data will lead to an inaccurate drive. I drove the vehicle using the direction keys of my keyboard so it was a little challenging driving accurately. After each attempt I fed the the dataset to the model and tested the simulator in autonomous mode. 

The following transformations were performed on the images before being fed to the model:
Cropping image: The view from all three cameras shows some part of the hood of the vehicle, the road ahead and the horizon above. The model will only need the view of road without other distractions such as the hood and horizon so I cropped them out.
Resize the image: The images were resized to 64x64 which is the input image shape for the model.
Flipping the images: The dataset contained more of images of the datasets turning to left and very few images turning right. This could leading to a bias towards the left. So I simulated the right turn images by creating flipped the left turn images with their left turning steering angles.
Brightness augmentation: There different lighting conditions throughout the track. By augmenting the brightness of the images, we can generate images with different lighting conditions as training data for the model so that the model  can also learn these different conditions.
A python generator creates new training batches by applying these transformations with the corrected steering angles.

Model:

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_2 (Lambda)                (None, 64, 64, 3)     0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 32, 32, 32)    896         lambda_2[0][0]                   
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 32, 32, 32)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 31, 31, 32)    0           activation_4[0][0]               
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 16, 16, 64)    18496       maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
relu2 (Activation)               (None, 16, 16, 64)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 8, 8, 64)      0           relu2[0][0]                      
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 8, 8, 128)     73856       maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 8, 8, 128)     0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_6 (MaxPooling2D)    (None, 4, 4, 128)     0           activation_5[0][0]               
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 2048)          0           maxpooling2d_6[0][0]             
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 2048)          0           flatten_2[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 128)           262272      dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 128)           0           dense_4[0][0]                    
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 128)           0           activation_6[0][0]               
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 128)           16512       dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1)             129         dense_5[0][0]                    
====================================================================================================

Optimization:
Optimizer - Adam Optimizer
No. of epochs – 1
Samples per epoch - 20,000
Loss – Mean squared error

The initial drives in autonomous mode were continuously going off the track in the sharp turns. Every drive which had occurrences of the vehicle going off the track a lot led to the vehicle driving off the road in autonomous mode. After multiple attempts, and my driving had improved, I decided to only collect data on the areas with difficulties which were the sharp left turn just after bridge and the sharp right turn. I recorded data of the vehicle turning out of the turns so that the model can learn what to do when the vehicle finds itself in those situations. After adding more data of the vehicle turning in more acute angles at those troubled areas to the datasets, the vehicle was able to complete the drive over the track multiple laps.
