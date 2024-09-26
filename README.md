## **Streetstatistics-noGUI version**
Containerized (docker) version of StreetStatistics. Includes a python script as an example of how to interact.

## **How to use:** 
There are 2 dockers, one for capturing frames and the other for processing them. 
These are standalone and can be used interchangably. 

Capturing frames docker is a docker that runs continuously until stopped. 
It takes the following inputs: 
-- url, to a youtube live stream
-- path, to a folder used as a storing place (or queue) for the frames
-- frames_to_skip, amount of frames to skip during processing 
-- duration, time in seconds before the connection is closed. Default 43200

Using these it creates a connection and starts collecting frames from the youtube link.

Model docker is a docker that runs when it is called.
It takes the following inputs:
-- frame_path, path to the frame you wish to process
-- points, Points data for the structure
-- model_path, path to the yolo model you wish to use. 

The model docker processes data and outputs to a predefined folder, it returns the processed frame, stats, and two objects in the form of a picklefile. 
The processed frame and stats can be read. The picklefiles are used in the next iteration to maintain a tracking state, the user does not need to concern themselves with the files. (But they may valuable hold information)


## **Example script:**
main.py is set up as following:
It starts by collecting frames, once there are more than {x} frames in the designated folder it sends a signal to the docker to stop. 
Once it has stopped the model docker will start processing the frames, the outputs are read and printed at the end of its run. 

## **StreetStatistics:**
StreetStatistics was made with a user-friendly GUI, check out the repo: 
https://github.com/JazzPauw/streetstatistics
 
