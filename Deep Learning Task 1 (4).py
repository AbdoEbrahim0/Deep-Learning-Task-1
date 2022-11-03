#!/usr/bin/env python
# coding: utf-8

# In[505]:


import numpy as np  ## for leable encoding
import matplotlib.pyplot as plt
import seaborn as sns  #plotting feature
import pandas as pd  ##reading data
from sklearn import preprocessing  ## for leable encoding


# In[506]:


## Pandas recognized both empty cells and “NA” as a missing value. Unfortunately, the other types weren’t recognized
##https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b
missing_values = ["n/a", "na", "--", "", "NA"] 
peng = pd.read_csv(
    "D:/Fourth Year/1st Term/DeepLearnining/fcis2023/Labs/penguins.csv",
    na_values=missing_values)


# In[507]:


import seaborn as sns

sns.pairplot(peng, hue="species", palette='bright')


# In[508]:


peng.info()


# In[509]:


print(peng[:50])
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(peng[50:100])
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(peng[100:150])


print("species sum :", peng["species"].isnull().sum())
print("bill_length_mm :", peng["bill_length_mm"].isnull().sum())
print("bill_depth_mm :", peng["bill_depth_mm"].isnull().sum())
print("flipper_length_mm :", peng["flipper_length_mm"].isnull().sum())
print("gender sum :", peng["gender"].isnull().sum())
print("body_mass_g sum :", peng["body_mass_g"].isnull().sum())
## Pandas recognized both empty cells and “NA” as a missing value. Unfortunately, the other types weren’t recognized


# In[510]:


##can see null values now
print("species sum :", peng["species"].isnull().sum())
print("bill_length_mm :", peng["bill_length_mm"].isnull().sum())
print("bill_depth_mm :", peng["bill_depth_mm"].isnull().sum())
print("flipper_length_mm :", peng["flipper_length_mm"].isnull().sum())
print("gender sum :", peng["gender"].isnull().sum())
print("body_mass_g sum :", peng["body_mass_g"].isnull().sum())


# In[511]:


##fill N/A with most occurence
peng = peng.fillna({"gender": peng["gender"].mode()[0]})


# In[512]:


peng


# In[513]:


## now we dont have any missing values in gender column
print("gender sum :", peng["gender"].isnull().sum())


# In[514]:


print(peng['gender'].unique())
print(peng['species'].unique())


# In[515]:


from sklearn import preprocessing  ## for leable encoding

label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'gender'.
peng['gender'] = label_encoder.fit_transform(peng['gender'])
##peng['species'] = label_encoder.fit_transform(peng['species'])

print(['male', 'female'])
print(['Adelie', 'Gentoo', 'Chinstrap'])
print(peng['gender'].unique())
print(peng['species'].unique())


# In[516]:


print(peng['gender'].unique())


# In[517]:


peng


# In[518]:


featuresNames = peng.columns
print(featuresNames)  ##type of this is object not array
classesNames = ['Adelie', 'Gentoo', 'Chinstrap']
featuresNames = [
    'species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
    'gender', 'body_mass_g'
]


# In[519]:


peng


# In[520]:


import seaborn as sns

sns.pairplot(peng, hue="species", palette='bright')


# In[521]:


print(peng[:50])
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(peng[50:100])
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(peng[100:150])


# In[522]:


#150*0.8= 120 for training  120/3 (no.classes)= 40 train for every class
#150*0.7= 105 for training 105/3 (no.classes)= 35 train for every class
#150*0.6= 90 for training 90/3 (no.classes)= 30 train for every class

C1_train=peng.loc[0:29]
C2_train=peng.loc[50:79]
C3_train=peng.loc[100:129]

C1_train
print(">>>>>>C1_train:", C1_train.count())  ## to ensure that i had 30 row for every class
print(">>>>>>C2_train:",C2_train.count())
print(">>>>>>C3_train:",C3_train.count())


# In[523]:


C1_train  ##monitor your data


# In[524]:


C2_train ##monitor your data


# In[525]:


C3_train ##monitor your data


# In[526]:


##make all trained data together
C_train_Concat=pd.concat([C1_train,C2_train,C3_train])
C_train_Concat.count()


# In[527]:


##this step to subtract training data from original data to get Test Data
C_test_Concat=peng.merge(C_train_Concat ,how='left',indicator=True) 
print(C_test_Concat.to_string())


# In[528]:


##this step to subtract training data from original data to get Test Data

C_test_Concat.count()  


# In[529]:


AllDataFrame=C_test_Concat
print(AllDataFrame.to_string())


# In[530]:


##this step to subtract training data from original data to get Test Data

C_test_Concat=C_test_Concat[C_test_Concat['_merge']=='left_only']


# In[531]:


C_test_Concat.count()  ### to ensure that i had 20 row for every class total (60)


# In[532]:


print(C_test_Concat.to_string())  ##moitor testing data


# In[533]:


#SelectedTraining=None
#SelectedTesting=None
def Selected2ClassesTrainig(C1,C2):  ## take 2 strings deepened on which classes you select
    SelectedTraining=pd.concat([ C_train_Concat[C_train_Concat.species==C1]  ,C_train_Concat[C_train_Concat.species==C2]    ])
    return SelectedTraining
    #print(SelectedTraining.to_string())

SelectedTraining = Selected2ClassesTrainig('Gentoo', 'Chinstrap')


# In[534]:



print('############################# Training Seclected Classes#############################\n')
print(SelectedTraining.to_string())


# In[535]:


def Selected2ClassesTesting(C1,C2):  ## take 2 strings deepened on which classes you select
    SelectedTesting=pd.concat([ C_test_Concat[C_test_Concat.species==C1]  ,C_test_Concat[C_test_Concat.species==C2]    ])
    return SelectedTesting
    #print(SelectedTesting.to_string())
    
SelectedTesting= Selected2ClassesTesting('Gentoo', 'Chinstrap')


# In[536]:



print('############################## Testing Seclected Classes#############################')
print(SelectedTesting.to_string())


# In[537]:


'''
##use this to drop rows of specisf class with specif value like if i wan t to drop whole class from my dataset like  Chinstrap
print(peng[peng.species!='Chinstrap'].count())
print(peng[peng.species!='Chinstrap'].to_string())

'''

## remove ! to only show seleced class then i merge 2 together
##print(peng[peng.species=='Chinstrap'].count())
##print(peng[peng.species=='Chinstrap'].to_string())



##convert selecte 2 classes to 1 and -1
pengOfOnesTraining=None
pengOfOnesTesting=None



def Ones(C1,C2):  ## take 2 strings deepened on which classes you select
    mapping = {C1: 1, C2: -1}
    pengOfOnesTraining=SelectedTraining.replace({'species': mapping})
    pengOfOnesTesting=SelectedTesting.replace({'species': mapping})
    return pengOfOnesTraining ,  pengOfOnesTesting
    #print(pengOfOnesTraining.to_string())
    #print(pengOfOnesTesting.to_string())

pengOfOnesTraining ,pengOfOnesTesting = Ones('Gentoo', 'Chinstrap')


# In[538]:


print('#############################Final Training Seclected Classes#############################\n')    

print(pengOfOnesTraining.to_string())


# In[539]:


print('#############################Final Testing Seclected Classes#############################\n')

print(pengOfOnesTesting.to_string())


# In[540]:


##suffele final DataTraining after last preprocessing 
pengOfOnesTraining = pengOfOnesTraining.sample(frac = 1)
pengOfOnesTraining


# In[541]:


##suffele final DatatTest after last preprocessing 
pengOfOnesTesting = pengOfOnesTesting.sample(frac = 1)
pengOfOnesTesting


# In[ ]:





# In[ ]:





# In[542]:


## must shuffle here this Next dataframe  "" pengOfOnesTesting""


# In[543]:


## only show 2 selected fearture
##t = pengOfOnesTraining[['bill_depth_mm', 'bill_length_mm']]
##print(t.to_string())
#df = df.loc[:, ['col2', 'col6']]
## 
sp=None
SelectedTwoFeatureTraining=None
SelectedTwoFeatureTesting=None
def Selected2Feature(F1,F2):  ## take 2 strings deepened on which Features you select
    #SelectedTwoFeatureTraining = SelectedTwoFeatureTraining.loc[:, [F1, F2]]
    #SelectedTwoFeatureTesting = SelectedTwoFeatureTesting.loc[:, [F1, F2]]
    sp=pengOfOnesTraining['species']
    SelectedTwoFeatureTraining= pengOfOnesTraining[[F1, F2]]
    ##relize we used pengOfOnesTraining
    SelectedTwoFeatureTraining= SelectedTwoFeatureTraining.assign(species=pengOfOnesTraining['species'])
    SelectedTwoFeatureTesting = pengOfOnesTesting[[F1,F2]]
        ##relize we used pengOfOnesTesting
    SelectedTwoFeatureTesting= SelectedTwoFeatureTesting.assign(species=pengOfOnesTesting['species'])

    return SelectedTwoFeatureTraining , SelectedTwoFeatureTesting
    #print(SelectedTraining.to_string())



# In[544]:


sp=pengOfOnesTraining[['species']]
sp


# In[545]:


SelectedTwoFeatureTraining , SelectedTwoFeatureTesting = Selected2Feature('bill_length_mm', 'bill_depth_mm')


# In[546]:


print(SelectedTwoFeatureTraining.count())
SelectedTwoFeatureTraining


# In[547]:


###use this code if u want to move specis clolumn to first not last 
#SelectedTwoFeatureTraining
first_column = SelectedTwoFeatureTraining.pop('species')
  
# insert column using insert(position,column_name,
# first_column) function
SelectedTwoFeatureTraining.insert(0, 'species', first_column)


# In[548]:


print('#############################Final SelectedTwoFeatureTraining#############################\n')
print(SelectedTwoFeatureTraining.count())

SelectedTwoFeatureTraining


# In[549]:


'''
#C_train_Concat=pd.concat([C1_train,C2_train,C3_train])
s=pengOfOnesTraining['species']
SelectedTwoFeatureTraining = pd.concat([s,SelectedTwoFeatureTraining])
SelectedTwoFeatureTesting

'''


# In[550]:


print(SelectedTwoFeatureTesting.count())

SelectedTwoFeatureTesting


# In[551]:


###use this code if u want to move specis clolumn to first not last 
#SelectedTwoFeatureTraining
first_column = SelectedTwoFeatureTesting.pop('species')
  
# insert column using insert(position,column_name,
# first_column) function
SelectedTwoFeatureTesting.insert(0, 'species', first_column)


# In[552]:


print('#############################Final SelectedTwoFeatureTesting#############################\n')
print(SelectedTwoFeatureTesting.count())

SelectedTwoFeatureTesting


# In[553]:


##متنساش تغير ال ترين سامبلز ل 30 والتيست ل 20


# In[554]:


##متنساش تعمل راندوم لل ارصفوف 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




