#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import sys
import csv
import pickle
import math
import numpy as np
import pandas as pd
from random import seed
from random import randint
from nltk.stem import PorterStemmer
from IPython.display import display, HTML


# In[2]:


StopWords=open("Stopword-List.txt")
StopWords=StopWords.readlines()
ps = PorterStemmer()


# # FILTERING OF DATA

# In[3]:


def GenerateTokensBySpace(File):
    Tokens=[]
    for Word in File:
        Token=""
        for Character in Word:
            if Character!=" ":
                Token+=Character
            else:
                Tokens.append(Token)
                Token=""
    return Tokens
def RemovingDots(Tokens):
    Result=[]
    for Token in Tokens:
        if Token.count(".")>=2: #For Initials like U.S.A
            Result.append(Token.replace(".",""))
        else:
            SplitByDot=re.split("\.",Token) # For Words Like Thousands.So
            for Word in SplitByDot:
                if Word!="":
                    Result.append(Word)
    return Result

def RemovingContractions(Tokens):
    Result=[]
    for Token in Tokens:
        Word=Token.replace("?","").replace(":","").replace(",","").replace('"',"")
        Word=re.split(r"n't",Word)
        if len(Word)>1:
            Word[1]="not"
        if len(Word)<2:
            Word=re.split(r"'s",Word[0])
            if len(Word)>1:
                Word[1]="is"
        if len(Word)<2:
            Word=re.split(r"'re",Word[0])
            if len(Word)>1:
                Word[1]="are"
        if len(Word)<2:
            Word=re.split(r"'m",Word[0])
            if len(Word)>1:
                Word[1]="am"
        if len(Word)<2:
            Word=re.split(r"'ll",Word[0])
            if len(Word)>1:
                Word[1]="will"
        if len(Word)<2:
            Word=re.split(r"'ve",Word[0])
            if len(Word)>1:
                Word[1]="have"
        if len(Word)<2:
            Word=re.split(r"'d",Word[0])
            if len(Word)>1:
                Word[1]="had"
        for W in Word:
            if W!="":
                Result.append(W)
    return Result

def LOWERCASECONVERTOR(Tokens):
    Result=[]
    for Token in Tokens:
        Result.append(Token.lower())
    return Result

def RemovingBraces(Tokens): #[]
    Result=[]
    for Token in Tokens:
        Words=re.split(r"\[(\w+)\]",Token)
        for Word in Words:
            if Word!="":
                Result.append(Word)
    return Result

def RemovingHypens(Tokens):
    Result=[]
    for Token in Tokens:
        Words=re.split(r"\-",Token)
        for Word in Words:
            if Word!="":
                Result.append(Word)
    return Result

def PorterStemming(Tokens):
    Result=[]
    for Token in Tokens:
        Result.append(ps.stem(Token))
    return Result

def GeneratingStopWordsList(File):
    StopWordList=[]
    for word in StopWords:
        word=re.split("\\n",word)
        if word[0]!="":
            StopWordList.append(word[0].replace(" ",""))
    return StopWordList


def RemovingStopWords(Tokens,StopWordList):
    Result=[]
    for Token in Tokens:
        if Token not in StopWordList:
            Result.append(Token)
    return Result

def FinalFilter(SortedKeys):
    
    NewKeys=[]
    
    for i in SortedKeys:
        i=i.replace("'","").replace(";","").replace(")","").replace("(","").replace("[","").replace("]","").replace("Ã¢Â","").replace("Ã¢Â","")
        NewKeys.append(i)
    while "" in NewKeys:
        NewKeys.remove("")
        
    return NewKeys


# ## Generating Dictionary Keys

# In[9]:


def GeneratePostingList(directory):
    
    Dictionary={}
    
    IndexOfFile=0

    Folders=next(os.walk(directory))[1]

    for Folder in Folders:
    
        Files=next(os.walk(directory+Folder))[2]
    
        for FileName in Files:
        
            File=open(directory+Folder+'/'+FileName)
        
            Speech=File.readlines()
        
            # Filtering Data
        
            StopWordList=GeneratingStopWordsList(StopWords)
        
            Tokens=GenerateTokensBySpace(Speech)
        
            Tokens=RemovingContractions(Tokens)
        
            Tokens=RemovingDots(Tokens)
        
            Tokens=LOWERCASECONVERTOR(Tokens)
        
            Tokens=RemovingBraces(Tokens)
        
            Tokens=RemovingHypens(Tokens)
        
            Tokens=FinalFilter(Tokens)
        
            Tokens=RemovingStopWords(Tokens,StopWordList)
        
            Tokens=PorterStemming(Tokens)
        
        
            for i in range(0,len(Tokens)):
                
                Dictionary.setdefault(Tokens[i],{})
        
            IndexOfFile+=1
        
    return Dictionary


# In[10]:


Dictionary=GeneratePostingList('Train/bbcsport/')
SortedKeys=sorted(Dictionary)


# # Train Data set Information

# In[13]:


def GetTotalDocuments(directory):
    TotalTrainDoc=0
    
    Folders=next(os.walk(directory))[1]

    for Folder in Folders:
    
        Files=next(os.walk(directory+Folder))[2]
    
        for FileName in Files:
            
            TotalTrainDoc+=1
    
    return TotalTrainDoc

def GetTotalFolder(directory):
    
    return len(next(os.walk(directory))[1])
  
def GetTotalFilesInClass(ClassName,directory):
    
    Files=next(os.walk(directory+ClassName))[2]
    
    return len(Files)


# ## Printing Train Data Set Information

# In[14]:


print("*"*124)
print("Total Train Folders          : ",GetTotalFolder('Train/bbcsport/'))
print("Total Train Documents        : ",GetTotalDocuments('Train/bbcsport/'))
print("Documents In Athletics Class : ",GetTotalFilesInClass('athletics','Train/bbcsport/'))
print("Documents In Cricket Class   : ",GetTotalFilesInClass('cricket','Train/bbcsport/'))
print("Documents In Football Class  : ",GetTotalFilesInClass('football','Train/bbcsport/'))
print("Documents In Rugby Class     : ",GetTotalFilesInClass('rugby','Train/bbcsport/'))
print("Documents In Tennis Class    : ",GetTotalFilesInClass('tennis','Train/bbcsport/'))
print("*"*124)
TotalTrainDocuments=GetTotalDocuments('Train/bbcsport/')


# # Generating Train Data Set Files

# In[16]:


def GeneratingTrainCsvFile(Dictionary):
    
    SortedKeys=sorted(Dictionary) 
    
    MatrixForTrainDataset=[[0 for i in range(0,len(SortedKeys)+1)]for j in range(0,TotalTrainDocuments)]

    TermIdfMatrix=[[0 for i in range(0,len(SortedKeys)+1)]for j in range(0,TotalTrainDocuments)]
    
    IndexOfFile=0

    Folders=next(os.walk('Train/bbcsport/'))[1]

    for Folder in Folders:
    
        Files=next(os.walk('Train/bbcsport/'+Folder))[2]
    
        for FileName in Files:
        
            File=open("Train/bbcsport/"+Folder+'/'+FileName)
        
            Speech=File.readlines()
        
            # Filtering Data
        
            StopWordList=GeneratingStopWordsList(StopWords)
        
            Tokens=GenerateTokensBySpace(Speech)
        
            Tokens=RemovingContractions(Tokens)
        
            Tokens=RemovingDots(Tokens)
        
            Tokens=LOWERCASECONVERTOR(Tokens)
        
            Tokens=RemovingBraces(Tokens)
        
            Tokens=RemovingHypens(Tokens)
            
            Tokens=FinalFilter(Tokens)
        
            Tokens=RemovingStopWords(Tokens,StopWordList)
        
            Tokens=PorterStemming(Tokens)
        
        
            for i in range(0,len(Tokens)):
            
                MatrixForTrainDataset[IndexOfFile][SortedKeys.index(Tokens[i])]+=1
            
                TermIdfMatrix[IndexOfFile][SortedKeys.index(Tokens[i])]=1
        
            MatrixForTrainDataset[IndexOfFile][len(MatrixForTrainDataset[IndexOfFile])-1]=Folder
        
            TermIdfMatrix[IndexOfFile][len(MatrixForTrainDataset[IndexOfFile])-1]=Folder
        
            IndexOfFile+=1
    
    
    for column in range(0,len(TermIdfMatrix[0])-1):
        
        NoOfDocumentForTerm=0
    
        for row in range(0,len(TermIdfMatrix)):
            
            NoOfDocumentForTerm+=TermIdfMatrix[row][column]
    
    
        for row in range(0,len(TermIdfMatrix)):
            
            if MatrixForTrainDataset[row][column]!=0:
                
                MatrixForTrainDataset[row][column]*=(math.log2(TotalTrainDocuments/NoOfDocumentForTerm))
    
    
    with open('traindata.p', 'wb') as fp:
        pickle.dump(MatrixForTrainDataset,fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('IdfScores.p', 'wb') as fp:
        pickle.dump(TermIdfMatrix,fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    return TermIdfMatrix,MatrixForTrainDataset


# In[17]:


try:
    print("*"*124)
    
    File = open("IdfScores.p","rb")
    TermIdfMatrix=pickle.load(File)
    
    File = open("traindata.p","rb")
    MatrixForTrainDataset=pickle.load(File)
    
    print("File Found")
    
    print("Printing Train Data .................")
    
    columns=SortedKeys.copy()
    
    columns.append("Class")
    
    File=pd.DataFrame(MatrixForTrainDataset,columns=columns)
    
    display(File)
    
    print("*"*124)
    
except:
    
    print("File Not Found")
    
    print("Creating Files .................")
    
    TermIdfMatrix,MatrixForTrainDataset=GeneratingTrainCsvFile(Dictionary)
    
    columns=SortedKeys.copy()
    
    columns.append("Class")
    
    File=pd.DataFrame(MatrixForTrainDataset,columns=columns)
    
    display(File)
    
    print("Files Created !")


# # Test Data Information

# ### Printing Test Data Information

# In[18]:


print("*"*124)
print("Total Test Folders          : ",GetTotalFolder('Test/'))
print("Total Test Documents        : ",GetTotalDocuments('Test/'))
print("Documents In Athletics Class : ",GetTotalFilesInClass('athletics','Test/'))
print("Documents In Cricket Class   : ",GetTotalFilesInClass('cricket','Test/'))
print("Documents In Football Class  : ",GetTotalFilesInClass('football','Test/'))
print("Documents In Rugby Class     : ",GetTotalFilesInClass('rugby','Test/'))
print("Documents In Tennis Class    : ",GetTotalFilesInClass('tennis','Test/'))
print("*"*124)
TotalTestDocuments=GetTotalDocuments('Test/')


# In[22]:


def GeneratingTestCsvFile(Dictionary,TermIdfMatrix):
    
    SortedKeys=sorted(Dictionary)
    
    MatrixForTestDataset=[[0 for i in range(0,len(SortedKeys)+1)]for j in range(0,TotalTestDocuments)]
    
    IndexOfFile=0

    Folders=next(os.walk('Test/'))[1]

    for Folder in Folders:
    
        Files=next(os.walk('Test/'+Folder))[2]
    
        for FileName in Files:
        
            File=open("Test/"+Folder+'/'+FileName)
        
            Speech=File.readlines()
        
            # Filtering Data
        
            StopWordList=GeneratingStopWordsList(StopWords)
        
            Tokens=GenerateTokensBySpace(Speech)
        
            Tokens=RemovingContractions(Tokens)
        
            Tokens=RemovingDots(Tokens)
        
            Tokens=LOWERCASECONVERTOR(Tokens)
        
            Tokens=RemovingBraces(Tokens)
        
            Tokens=RemovingHypens(Tokens)
            
            Tokens=FinalFilter(Tokens)
        
            Tokens=RemovingStopWords(Tokens,StopWordList)
        
            Tokens=PorterStemming(Tokens)
        
        
            for i in range(0,len(Tokens)):
                try:
                    MatrixForTestDataset[IndexOfFile][SortedKeys.index(Tokens[i])]+=1
                except:
                    pass
                
        
            MatrixForTestDataset[IndexOfFile][len(MatrixForTestDataset[IndexOfFile])-1]=Folder
        
            IndexOfFile+=1
      
    
    for column in range(0,len(TermIdfMatrix[0])-1):
    
        NoOfDocumentForTerm=0
    
        for row in range(0,len(TermIdfMatrix)):
        
            NoOfDocumentForTerm+=TermIdfMatrix[row][column]
    
    
        for row in range(0,len(MatrixForTestDataset)):
        
            if MatrixForTestDataset[row][column]!=0:
            
                MatrixForTestDataset[row][column]*=(math.log2(TotalTrainDocuments/NoOfDocumentForTerm))
    
    
    
    with open('testdata.p', 'wb') as fp:
        
        pickle.dump(MatrixForTestDataset,fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return MatrixForTestDataset
        


# In[23]:


try:
    print("*"*124)
    
    File = open("testdata.p","rb")
    
    MatrixForTestDataset=pickle.load(File)
    
    print("File Found")
    
    print("Printing Test Data .................")
    
    columns=SortedKeys.copy()
    
    columns.append("Class")
    
    File=pd.DataFrame(MatrixForTestDataset,columns=columns)
    
    display(File)
    
    print("*"*124)
    
except:
    
    print("File Not Found")
    
    print("Creating Files .................")
    
    MatrixForTestDataset=GeneratingTestCsvFile(Dictionary,TermIdfMatrix)
    
    columns=SortedKeys.copy()
    
    columns.append("Class")
    
    File=pd.DataFrame(MatrixForTestDataset,columns=columns)
    
    display(File)
    
    print("Files Created !")


# In[24]:


def GeneratingTestOutputCsvFile():
    
    print("*"*124)
    
    print("Generating Matrix To Be Saved in File")
    
    TrainDocIndex=0

    Output=[]

    Folders=next(os.walk('Test/'))[1]

    for Folder in Folders:
    
        Files=next(os.walk('Test/'+Folder))[2]
    
        for FileName in Files:
            
            l=[]
            
            l.append("DOCUMENT "+str(TrainDocIndex))
            
            l.append(Folder)
            
            Output.append(l)
        
            TrainDocIndex+=1
        
    
    Matrix=np.array(Output)
    
    print("Matrix Generated")
    
    columns=['Actual Output']
    
    columns.insert(0,' DOCUMENT INDEX ')
    
    print("Creating File................")
    
    File=pd.DataFrame(Matrix,columns=columns)
    
    File.to_csv("Test output.csv",index=False)
    
    print("File Created ")
    
    print("Printing Test Data Set ")
    
    display(File)
    
    print("*"*124)


# In[25]:


GeneratingTestOutputCsvFile()


# # Loading Files

# In[28]:


def LoadFiles():
    
    output=pd.read_csv('Test output.csv')
    
    with open('traindata.p', 'rb') as fp:
        MatrixForTrainDataset=pickle.load(fp)
    with open('testdata.p', 'rb') as fp:
        MatrixForTestDataset=pickle.load(fp)
    
    
#     return train,test,output,MatrixForTestDataset,MatrixForTrainDataset

    return output,MatrixForTestDataset,MatrixForTrainDataset


# In[29]:


# train,test,output,MatrixForTestDataset,MatrixForTrainDataset=LoadFiles()
output,MatrixForTestDataset,MatrixForTrainDataset=LoadFiles()


# # Generating Prediction File

# In[30]:


def MakingPredictionFile(MatrixForTestDataset,MatrixForTrainDataset):
    
    index=0
    PredictCorrect=0
    PredictWrong=0
    ActualClass="athletics"
    TotalClassDocuments=0

    PredictedResult=[]

    for Test in range(0,len(MatrixForTestDataset)):
    
    
        Result={}
    
    
        ActualClass=MatrixForTestDataset[Test][len(MatrixForTestDataset[0])-1]
    
    
        Distance1=sys.maxsize
        Distance2=sys.maxsize
        Distance3=sys.maxsize
    
        Class1=""
        Class2=""
        Class3=""
    
        DocTest=MatrixForTestDataset[Test][1:len(MatrixForTestDataset[Test])-1]
    
        for Train in range(0,len(MatrixForTrainDataset)):
        
        
            ClassTrain=MatrixForTrainDataset[Train][len(MatrixForTrainDataset[0])-1]
        
            DocTrain=MatrixForTrainDataset[Train][1:len(MatrixForTrainDataset[Train])-1]
        
            distance = math.sqrt(sum([(float(a) - float(b)) ** 2 for a, b in zip(DocTest,DocTrain)]))
        
#         Result.setdefault(distance,ClassTrain)
    
    
            if distance<Distance1 and distance<Distance2 and distance<Distance3:
            
                Distance1=distance
                Class1=ClassTrain
            
            elif distance<Distance2 and distance<Distance3:
            
                Distance2=distance
                Class2=ClassTrain
            
            elif distance<Distance3:
            
                Distance3=distance
                Class3=ClassTrain
      
    
#     print([b for a, b in sorted(Result.items())][0:3])
    
        Result.setdefault(Distance3,Class3)
        Result.setdefault(Distance2,Class2)
        Result.setdefault(Distance1,Class1)
    
        index=0
    
        DistanceSortedResult={}
    
    
        for i in sorted(Result):
        
            if Result[i] in DistanceSortedResult.keys():
            
                j=DistanceSortedResult[Result[i]]
                j+=1
                DistanceSortedResult[Result[i]]=j
            
            else:
                DistanceSortedResult.setdefault(Result[i],1)
            index+=1
            if index==3:
                break
    
        PredictedClass=""
        for key,value in DistanceSortedResult.items():
            PredictedClass=key
            break

        PredictedResult.append(PredictedClass)
    
        if PredictedClass==ActualClass:
            PredictCorrect+=1
        else:
            PredictWrong+=1
        
        TotalClassDocuments+=1
    
    Prediction=[]
    docno=0
    for i in PredictedResult:
        l=[]
        doc="Document "+str(docno)
        l.append(doc)
        l.append(i)
        Prediction.append(l)
        docno+=1
    pd.DataFrame(Prediction,columns=["Document Index","Predicted Ouptut"]).to_csv("Predicted Output.csv",index=False)


# In[31]:


try:
    print("*"*124)
    
    print("Prediction File Found")
    
    displayTrainDataset=pd.read_csv("Predicted Output.csv")
    
    print("Printing Output Data")
    
    display(displayTrainDataset)
    
    print("*"*124)
    
except:
    print("*"*124)
    print("File Not Found")
    print("Creating Files .................")
    MakingPredictionFile(MatrixForTestDataset,MatrixForTrainDataset)
    print("Files Created !")
    
    displayTrainDataset=pd.read_csv("Predicted Output.csv")
    
    print("Printing Output Data")
    
    display(displayTrainDataset)
    
    print("*"*124)
    
    


# ## Prediction

# In[32]:


TestOutput=pd.read_csv('Test output.csv')
PredictedOutput=pd.read_csv("Predicted Output.csv")

index=0
PredictCorrect=0
PredictWrong=0
ActualClass="athletics"
TotalClassDocuments=0
TotalPredictedCorrect=0

print(">"*123)
for i,y in zip(TestOutput['Actual Output'],PredictedOutput['Predicted Ouptut']):
    
    PreviousClass=ActualClass
    
    ActualClass=i
    
    AfterClass=ActualClass
    
    if PreviousClass!=AfterClass:
        
        print("-"*123)
        print(" "*50,PreviousClass.capitalize())
        print("Total Test Documents:   ",TotalClassDocuments)
        print("Predicted right:   ",PredictCorrect)
        print("Predicted wrong:   ",PredictWrong)
        print(PreviousClass.capitalize()," Class Accuracy:   ",(PredictCorrect/TotalClassDocuments)*100)
        print("-"*123)
        PredictCorrect=0
        PredictWrong=0
        TotalClassDocuments=0
    
    if i==y:
        TotalPredictedCorrect+=1
        PredictCorrect+=1
    else:
        PredictWrong+=1
        
    TotalClassDocuments+=1

print("-"*123)
print(" "*50,PreviousClass.capitalize())
print("Total Test Documents:   ",TotalClassDocuments)
print("Predicted right:   ",PredictCorrect)
print("Predicted wrong:   ",PredictWrong)
print(PreviousClass.capitalize()," Class Accuracy:   ",(PredictCorrect/TotalClassDocuments)*100)
print("-"*123)
print("Total Accuracy :          ",(TotalPredictedCorrect/len(MatrixForTestDataset))*100)
print(">"*123)


# # K-Means Clustering

# In[36]:


def ClassDistributionOfTerms():
    
    Dictionary={}
    
    Folders=next(os.walk('bbcsport/'))[1]

    for Folder in Folders:
    
        Files=next(os.walk('bbcsport/'+Folder))[2]
    
        for FileName in Files:
        
            File=open("bbcsport/"+Folder+'/'+FileName)
        
            Speech=File.readlines()
        
            # Filtering Data
        
            StopWordList=GeneratingStopWordsList(StopWords)
        
            Tokens=GenerateTokensBySpace(Speech)
        
            Tokens=RemovingContractions(Tokens)
        
            Tokens=RemovingDots(Tokens)
        
            Tokens=LOWERCASECONVERTOR(Tokens)
        
            Tokens=RemovingBraces(Tokens)
        
            Tokens=RemovingHypens(Tokens)
            
            Tokens=FinalFilter(Tokens)
        
            Tokens=RemovingStopWords(Tokens,StopWordList)
        
            Tokens=PorterStemming(Tokens)
        
        
            for i in range(0,len(Tokens)):
                
                if Tokens[i] not in Dictionary:
                    
                    Dictionary.setdefault(Tokens[i],{})
                    
                    Dictionary[Tokens[i]].setdefault(Folder,1)
                
                else:
                    
                    if Folder in Dictionary[Tokens[i]]:
                        
                        Quantity = Dictionary[Tokens[i]][Folder]
                        
                        Quantity+=1
                        
                        Dictionary[Tokens[i]][Folder]=Quantity
                    
                    else:
                        
                        Dictionary[Tokens[i]].setdefault(Folder,1)
    
    return Dictionary


# In[37]:


def GenerateTotalDatasetFile(FilterPercentage):
    
    ClassDistribution=ClassDistributionOfTerms()
    
    QtyTerms=0
    
    NewDictionary={}
    
    
    
    
    
    for Term in ClassDistribution:    
        
        for Class in ClassDistribution[Term]:
                
            if ClassDistribution[Term][Class]>=3:
                
                QtyTerms+=1
        
                NewDictionary.setdefault(Term)
        
    SortedKeys=sorted(NewDictionary)
    
    
    Folders=next(os.walk('bbcsport/'))[1]
    TotalDocuments=len(next(os.walk('bbcsport/'+Folders[0]))[2])
    TotalDocuments+=len(next(os.walk('bbcsport/'+Folders[1]))[2])
    TotalDocuments+=len(next(os.walk('bbcsport/'+Folders[2]))[2])
    TotalDocuments+=len(next(os.walk('bbcsport/'+Folders[3]))[2])
    TotalDocuments+=len(next(os.walk('bbcsport/'+Folders[4]))[2])

    
    
    MatrixForTrainDataset=[[0 for i in range(0,len(SortedKeys)+1)]for j in range(0,TotalDocuments)]

    TermIdfMatrix=[[0 for i in range(0,len(SortedKeys)+1)]for j in range(0,TotalDocuments)]
    
    IndexOfFile=0

    Folders=next(os.walk('bbcsport/'))[1]

    for Folder in Folders:
    
        Files=next(os.walk('bbcsport/'+Folder))[2]
    
        for FileName in Files:
        
            File=open("bbcsport/"+Folder+'/'+FileName)
        
            Speech=File.readlines()
        
            # Filtering Data
        
            StopWordList=GeneratingStopWordsList(StopWords)
        
            Tokens=GenerateTokensBySpace(Speech)
        
            Tokens=RemovingContractions(Tokens)
        
            Tokens=RemovingDots(Tokens)
        
            Tokens=LOWERCASECONVERTOR(Tokens)
        
            Tokens=RemovingBraces(Tokens)
        
            Tokens=RemovingHypens(Tokens)
            
            Tokens=FinalFilter(Tokens)
        
            Tokens=RemovingStopWords(Tokens,StopWordList)
        
            Tokens=PorterStemming(Tokens)
        
        
            for i in range(0,len(Tokens)):
                
                try:
            
                    MatrixForTrainDataset[IndexOfFile][SortedKeys.index(Tokens[i])]+=1
            
                    TermIdfMatrix[IndexOfFile][SortedKeys.index(Tokens[i])]=1
                
                except:
                    pass
         

        
            MatrixForTrainDataset[IndexOfFile][len(MatrixForTrainDataset[IndexOfFile])-1]=Folder
        
        
            TermIdfMatrix[IndexOfFile][len(MatrixForTrainDataset[IndexOfFile])-1]=Folder
        
            IndexOfFile+=1
    
    
    for column in range(0,len(TermIdfMatrix[0])-1):
        
        NoOfDocumentForTerm=0
    
        for row in range(0,len(TermIdfMatrix)):
            
            NoOfDocumentForTerm+=TermIdfMatrix[row][column]
            
    
        for row in range(0,len(TermIdfMatrix)):
            
            if MatrixForTrainDataset[row][column]!=0:
                
                MatrixForTrainDataset[row][column]*=(math.log2(TotalDocuments/NoOfDocumentForTerm))
    
    
    with open('Kmeansdataset.p', 'wb') as fp:
        pickle.dump(MatrixForTrainDataset,fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return MatrixForTrainDataset,SortedKeys


# In[38]:


try:
    print("*"*124)
    File = open("Kmeansdataset.p","rb")
    MatrixForTrainDataset = pickle.load(File)
    print("Total data Set File Found")
    print("*"*124)
except:
    
    print("File Not Found")
    
    print("Creating Files .................")
    
    MatrixForTrainDataset,Dictionary = GenerateTotalDatasetFile()
    
    SortedKeys=sorted(Dictionary)
    
    print("Printing Train Data .................")
    
    columns=SortedKeys.copy()
    
    columns.append("Class")
    
    File=pd.DataFrame(MatrixForTrainDataset,columns=columns)
    
    display(File)
    
    print("Files Created !")


# In[51]:


def SetDocuments(D1,D2,D3,D4,D5):
    
    NewMatrixForTrainDataset=np.delete(MatrixForTrainDataset,(D1,D2,D3,D4,D5),axis=0)
    
    D1=MatrixForTrainDataset[D1]
    D2=MatrixForTrainDataset[D2]
    D3=MatrixForTrainDataset[D3]
    D4=MatrixForTrainDataset[D4]
    D5=MatrixForTrainDataset[D5]
    
    return MatrixForTrainDataset,NewMatrixForTrainDataset,D1,D2,D3,D4,D5


# In[52]:


def DocumentCluster(D1,D2,D3,D4,D5):
    
    DocumentCluster={}
    
    DocumentCluster.setdefault(0,[])
    DocumentCluster.setdefault(1,[])
    DocumentCluster.setdefault(2,[])
    DocumentCluster.setdefault(3,[])
    DocumentCluster.setdefault(4,[])
    
    
    MatrixForTrainDataset,NewMatrixForTrainDataset,D1,D2,D3,D4,D5=SetDocuments(D1,D2,D3,D4,D5)
    
    RandomFiveDocuments=[D1,D2,D3,D4,D5]
    
    print("Calculating Distance ...............................")
    
    for i in range(0,len(NewMatrixForTrainDataset)):
    
        DocTrain=NewMatrixForTrainDataset[i][0:len(NewMatrixForTrainDataset[0])-1]
    
        MiniMumDistance={}
    
        for j in range(0,len(RandomFiveDocuments)):
        
            DocTest=RandomFiveDocuments[j][0:len(RandomFiveDocuments[0])-1]
            
#             print(DocTrain)
            
            DocTestMagnitude=math.sqrt(sum([float(a)**2 for a in DocTest]))
            
            DocTrainMagnitude=math.sqrt(sum([float(a)**2 for a in DocTrain]))
            
            Magnitude=DocTestMagnitude*DocTrainMagnitude
            
            distance=sum([(float(a)*float(b)) for a, b in zip(DocTest,DocTrain)])
            
            if Magnitude==0:
                
                distance=0
            
            else:
                
                distance=distance/Magnitude
        
            MiniMumDistance.setdefault(distance,j)
    
        DocumentCluster[[b for a, b in sorted(MiniMumDistance.items(),reverse=True)][0]].append(i)
    
    
    
    print("Calculating Mean .................................................")
    
    
    DocumentClusterMean={}
    
    DocumentClusterMean.setdefault(0,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
    DocumentClusterMean.setdefault(1,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
    DocumentClusterMean.setdefault(2,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
    DocumentClusterMean.setdefault(3,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
    DocumentClusterMean.setdefault(4,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
    
    
    for Cluster in range(0,len(DocumentCluster)):
    
        for Doc in DocumentCluster[Cluster]:
        
            MatrixOne=DocumentClusterMean[Cluster]
        
            MatrixTwo=MatrixForTrainDataset[Doc][0:len(MatrixForTrainDataset[0])-1]
        
            DocumentClusterMean[Cluster]=[x+y for x,y in zip(MatrixOne,MatrixTwo)]
        
        try:
            
            DocumentClusterMean[Cluster] = [float(x) / float(len(DocumentCluster[Cluster]))  for x in DocumentClusterMean[Cluster] ]
        
        except:
            
            pass
    
    DocumentClusterAfterMean={}
    
    DocumentClusterAfterMean.setdefault(0,[])
    
    DocumentClusterAfterMean.setdefault(1,[])
    
    DocumentClusterAfterMean.setdefault(2,[])
    
    DocumentClusterAfterMean.setdefault(3,[])
    
    DocumentClusterAfterMean.setdefault(4,[])
    
    print("Calculating Document Cluster After Calculating Mean ................................")
    
    for i in range(0,len(MatrixForTrainDataset)):
    
        DocTrain=MatrixForTrainDataset[i][0:len(MatrixForTrainDataset[0])-1]
    
        MiniMumDistance={}
    
        for j in range(0,len(DocumentClusterMean)):
        
            DocTest=DocumentClusterMean[j]
            
            DocTestMagnitude=math.sqrt(sum([float(a)**2 for a in DocTest]))
            
            DocTrainMagnitude=math.sqrt(sum([float(a)**2 for a in DocTrain]))
            
            Magnitude=DocTestMagnitude*DocTrainMagnitude
            
            distance=sum([(float(a)*float(b)) for a, b in zip(DocTest,DocTrain)])
            
            if Magnitude==0:
                
                distance=0
            
            else:
                
                distance=distance/Magnitude
        
            MiniMumDistance.setdefault(distance,j)
    
        DocumentClusterAfterMean[[b for a, b in sorted(MiniMumDistance.items(),reverse=True)][0]].append(i)
    
    
    print("Converging Cluster...............................")
    
    Iterations=0
    
    while DocumentCluster[0]!=DocumentClusterAfterMean[0] or DocumentCluster[1]!=DocumentClusterAfterMean[1] or DocumentCluster[2]!=DocumentClusterAfterMean[2] or DocumentCluster[3]!=DocumentClusterAfterMean[3] or DocumentCluster[4]!=DocumentClusterAfterMean[4]:
        
        Iterations+=1
        
        print("Iterations : ",Iterations)
    
        DocumentCluster=DocumentClusterAfterMean
    
    
        DocumentClusterMean={}
        
        DocumentClusterMean.setdefault(0,[0 for i in range(0,len(MatrixForTrainDataset[0]))])

        DocumentClusterMean.setdefault(1,[0 for i in range(0,len(MatrixForTrainDataset[0]))])

        DocumentClusterMean.setdefault(2,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
        DocumentClusterMean.setdefault(3,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
        DocumentClusterMean.setdefault(4,[0 for i in range(0,len(MatrixForTrainDataset[0]))])
    
        for Cluster in range(0,len(DocumentClusterAfterMean)):
        
            for Doc in DocumentClusterAfterMean[Cluster]:
            
                MatrixOne=DocumentClusterMean[Cluster]
            
            
#                 print(Doc,len(MatrixForTrainDataset[0])-1)
                
                MatrixTwo=MatrixForTrainDataset[Doc][0:len(MatrixForTrainDataset[0])-1]
            
                DocumentClusterMean[Cluster]=[x+y for x,y in zip(MatrixOne,MatrixTwo)]
     
            try:
                DocumentClusterMean[Cluster] = [float(x) / float(len(DocumentClusterAfterMean[Cluster]))  for x in DocumentClusterMean[Cluster] ]
            except:
                pass

    
        DocumentClusterAfterMean={}
    
        DocumentClusterAfterMean.setdefault(0,[])
    
        DocumentClusterAfterMean.setdefault(1,[])
    
        DocumentClusterAfterMean.setdefault(2,[])
    
        DocumentClusterAfterMean.setdefault(3,[])
    
        DocumentClusterAfterMean.setdefault(4,[])
    
    
        for i in range(0,len(MatrixForTrainDataset)):
    
            DocTrain=MatrixForTrainDataset[i][0:len(MatrixForTrainDataset[0])-1]
    
            MiniMumDistance={}
    
            for j in range(0,len(DocumentClusterMean)):
        
                DocTest=DocumentClusterMean[j]
                
                DocTestMagnitude=math.sqrt(sum([float(a)**2 for a in DocTest]))
            
                DocTrainMagnitude=math.sqrt(sum([float(a)**2 for a in DocTrain]))
            
                Magnitude=DocTestMagnitude*DocTrainMagnitude
            
                distance=sum([(float(a)*float(b)) for a, b in zip(DocTest,DocTrain)])
                
                if Magnitude==0:
                    
                    distance=0
                
                else:
                    
                    distance=distance/Magnitude
        
                MiniMumDistance.setdefault(distance,j)
    
            DocumentClusterAfterMean[[b for a, b in sorted(MiniMumDistance.items(),reverse=True)][0]].append(i)
    
    return DocumentCluster,DocumentClusterAfterMean


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


# In[63]:



try:
    print("*"*124)
    File = open("Cluster.p","rb")
    d = pickle.load(File)
    print("Cluster Found")
    print("*"*124)
except:
    
    Documents=np.random.randint(low = 1, high = len(MatrixForTrainDataset)-1, size = 5) 
    d,d2=DocumentCluster(Documents[0],Documents[1],Documents[2],Documents[3],Documents[4])
    
    with open('Cluster.p', 'wb') as fp:
        pickle.dump(d,fp, protocol=pickle.HIGHEST_PROTOCOL)


# In[64]:


AthleticsClassDocuments=[]
TennisClassDocuments=[]
RugbyClassDocuments=[]
FootballClassDocuments=[]
CricketClassDocuments=[]


# In[65]:


for i in range(0,len(MatrixForTrainDataset)):
    if MatrixForTrainDataset[i][len(MatrixForTrainDataset[0])-1]=="athletics":
        AthleticsClassDocuments.append(i)
    if MatrixForTrainDataset[i][len(MatrixForTrainDataset[0])-1]=="cricket":
        CricketClassDocuments.append(i)
    if MatrixForTrainDataset[i][len(MatrixForTrainDataset[0])-1]=="rugby":
        RugbyClassDocuments.append(i)
    if MatrixForTrainDataset[i][len(MatrixForTrainDataset[0])-1]=="football":
        FootballClassDocuments.append(i)
    if MatrixForTrainDataset[i][len(MatrixForTrainDataset[0])-1]=="tennis":
        TennisClassDocuments.append(i)


# In[67]:


print(">"*123)
Accuracy=0
for i in range(0,5):
    R={}
    R.setdefault(len(intersection(AthleticsClassDocuments,d[i])),'Athletics')
    R.setdefault(len(intersection(FootballClassDocuments,d[i])),'Football')
    R.setdefault(len(intersection(TennisClassDocuments,d[i])),'Tennis')
    R.setdefault(len(intersection(RugbyClassDocuments,d[i])),'Rugby')
    R.setdefault(len(intersection(CricketClassDocuments,d[i])),'Cricket')
    print("Cluster :",i)
    print("Cluster Length : ",len(d[i]))
    print("Athletics Class Length : ",len(intersection(AthleticsClassDocuments,d[i])))
    print("Football Class Length : ",len(intersection(FootballClassDocuments,d[i])))
    print("Rugby Class Length : ",len(intersection(RugbyClassDocuments,d[i])))
    print("Tennis Class Length : ",len(intersection(TennisClassDocuments,d[i])))
    print("Cricket Class Length : ",len(intersection(CricketClassDocuments,d[i])))
    print("Cluster : ",i," belongs to : ",[b for a,b in sorted(R.items(),reverse=True)][0])
    Accuracy+=[a for a,b in sorted(R.items(),reverse=True)][0]
    print("\n")
print("Purity Of Cluster is : ",(Accuracy/len(MatrixForTrainDataset))*100)
print(">"*123)


# In[ ]:




