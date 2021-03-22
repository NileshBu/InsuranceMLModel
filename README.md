# InsuranceMLModel

Insurance ML Model is the folder which consist of the ML Model Prepartion Material ( including Dataset, Data Cleaning & Preparation Code, Traning ML Modelling Code, Metrics Analysis Code). It also consist of the code which is used to Expose the trained ML model as the REST API using Flask.

FOLDER INTRODUCTION:

1.) dataMiningAssignment.py --> This file consists of a python code for the Data Cleaning & Preparation Code, Traning ML Modelling Code, Metrics Analysis Code)

2.) app.py. --> This file consists of a python code which is used Expose the trained ML model created in the above (dataMiningAssignment.py file)as the REST API using Flask.

3.)final_model.sav --> This file consists of the Trained ML Model( from dataMiningAssignment.py file). The model is saved using Pickle in this file. This save model is then used in (app.py file) for exposing it(Trained ML Model) as an API using Flask.

4.)Dockerfile --> This is the docker file which is used to create a docker image.

5.)python_requirements.txt --> This is the file which consist of list of libraries which needs to be downloaded in the docker image; this is used by Dockerfile.

6.)carInsurance_train.csv --> This is the dataset which is used to train the ML model.

BASIC GUIDE TO USE 
Download or Clone all the Files available above in one same folder(ITS IS VERY IMPORTANT THAT ALL THE FILES ARE IN SAME FOLDER).


1.) If you want to check the data cleaning,data preparation, and ML Model Training Codebase, then please open dataMiningAssignment.py file in any of the your desirable tool. I used Anaconda(Spyder) and run it. Running it will create the get the unprocessed and uncleaned dataset, then will clean and prepare the dataset, and the ML model will be trained and saved. 

2.) If you want to get the trained ML model exposed as the REST API , then  please open dataMiningAssignment.py  file in any of the your desirable tool. I used Anaconda(Spyder). Run this file and the REST API will be exposed at local port 5000. And the Endpoint can be accessed via Postman using http://localhost:5000/api/ link and sending the data in the body . The format of JSON  data to be sent in the body is as mentioned belows:

{"Id":3,
   "Age":29,
   "Job":"management",
   "Marital":"single",
   "Education":"tertiary",
   "Default":0,
   "Balance":637,
   "HHInsurance":1,
   "CarLoan":0,
   "Communication":"cellular",
   "LastContactDay":3,
   "LastContactMonth":"jun",
   "NoOfContacts":1,
   "DaysPassed":119,
   "PrevAttempts":1,
   "Outcome":"failure",
   "CallStart":"16:30:24",
   "CallEnd":"16:36:04"}
   
   
   a.) After sending the above post request with the expected JSON data in the body ; the Rest API will then return the results as predicted using Trained Random Forest Model from within.
   
   
   b.) The output will be as follows: "[1]" --> representing SUCCESS and "[0]" --> representing FAILURE.
   
3.) Another easier way to run the ML model exposed as the REST API is by using Docker. In this case then there is no need to run open or run any of the files. I have created an Docker image and pushed it onto Docker Hub which can be pulled using --> docker pull bukane/prediction-api:latest
 In order to run it directly without pulling the image is just use command: ---> docker run -p 5000:5000 bukane/ prediction-api
 
 This will then run the ML model exposed as the REST API in local at port 5000 and it can be aceesed using http://localhost:5000/api/  and sending the expected JSON in the request.


