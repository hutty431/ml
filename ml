


[1200208001062_ML_Practical.pdf](https://github.com/hutty431/ml/files/11432844/1200208001062_ML_Practical.pdf)


AIM: Study python ecosystem for machine learning: Python,Scipy, Scikit learn. 
 
 
Python: 
 
•	The Python ecosystem is growing and may become the dominant platform for machine learning. 
 
•	The Primarily rationale for adopting Python for machine learning is because it is general purpose programming language that you can use both for research and development and in production. 
 
•	Python is a general purpose interpreted programming language. It is easy to learn and use primarily because the language focuses on readability. 
 
•	It’s a dynamic language and very suited to interactive development and quick prototyping with the power to support the development of large applications. 
 
•	Python is widely used for machine learning and data science because of the excellent Library support and because it is a general purpose programming language (unlike R or Matlab). 
 
•	It means that you can perform your research and development (figuring out what models to use) in the same programming language that you use in operations. 
 
 
Scipy: 
 
•	SciPy is an ecosystem of Python libraries for mathematics, science and engineering. 
•	It is an add-on to Python as it is further need for machine learning. 
 
•	The SciPy ecosystem is comprised of the following core modules relevant to machine learning: 
 
 
➔	NumPy: 
A foundation for SciPy that allows you to efficiently work with data in arrays. 
➔	Matplotlib: 
Allows you to create 2D charts and plots from data ➔ pandas: 
Tools and data structures to organize and analyze your data. 
 
 
•	To be effective at machine learning in Python you must install and become familiar with SciPy. Specifically: 
 
➔	You will use Pandas to load explore and better understand your data. 
➔	You will use Matplotlib (and wrappers of Matplotlib in other frameworks) to create plots and charts of your data. 
➔	You will prepare your data as NumPy arrays for modeling in machine learning algorithms. 
 
•	There are many ways to install SciPy. 
 
For example two popular ways are to use package management on your platform (e.g. yum on RedHat or macports on OS X) or use a Python package management tool like pip. 
 
 
Scikit learn: 
 
 
•	The scikit-learn library is how you can develop and practice machine learning in python. 
 
•	It is built upon and requires the SciPy ecosystem. 
 
•	The name “scikit” suggests that it is a SciPy plugin or toolkit. 

•	The focus of the library is machine learning algorithms for classification, regression, clustering and more. 
 
•	It also provides tools for related tasks such as evaluating models, tuning parameters and pre-processing data. 
 
•	Scikit-learn is open source and commercially usable under the BSD license. 
 
•	This means that you can learn about machine learning, develop models and put them into operations all with the same ecosystem and code. 
 
•	This is a powerful reason to use scikit-learn. 
 
•	If you are using Anaconda distribution, then no need to install Scikit-learn separately as it is already installed with it. 
 
•	You just need to use the package into your Python script. 
 
  
REFERENCE: 
 
 
https://machinelearningmastery.com/python-ecosystem-machine-learning/ 

















Practical-2

AIM: Study of pre-processing methods. Write a program to find following statistics from a given dataset. Mean, Mode, Median, Variance, Standard Deviation, Quartiles, Interquartile range. 
 
 
1.	Mean: 
 
import numpy as np a=[120,65,28,798,45,65,24] mean=np.mean(a) 
print(mean) 
 
Output: 
 
![image](https://github.com/hutty431/ml/assets/132896743/241c84e4-7f86-4895-9344-41a457814414)

 
 
 
2.	Mode: 
 
import statistics as stat a=[120,65,28,798,45,65,24] mode=stat.mode(a) print(mode) 
 
Output: 
![image](https://github.com/hutty431/ml/assets/132896743/ad0c9e0a-8bd3-4232-9851-952a390c3249)
 
 
1.	Median: 
 
import numpy as np a=[120,65,28,798,45,65,24] np.sort(a) median=np.median(a) print(median) 
 
 
Output: 
![image](https://github.com/hutty431/ml/assets/132896743/a169f6e7-a971-4efb-9a26-39507ce93d57)



2.	Variance: 
 
 
import statistics as stat a=[120,65,28,798,45,65,24] var=stat.variance(a) print(var) 
 
Output: 
 
 
 
1.	Standard Deviation: 
 
import statistics as stat a=[120,65,28,798,45,65,24] std=stat.stdev(a) 
print(std) 
 
 
Output: 
 
 
 
 
 
 
 
 
2.	Interquartile range: 
 
import 	numpy 	as 	np a=[120,65,28,798,45,65,24] q1=np.percentile(a,25) q3=np.percentile(a,75) q1,q3 iqr=q3-q1 iqr 
 
 
Output: 
 
 
 
 











Practical-3 
 
AIM: Study and implement PCA(Principal Component Analysis) in python. 
 
 
PCA(Principal Component Analysis): 
 
•	Like the term PCA suggests: 
 
➢	Principal: Reflecting importance 
➢	Component: A part of something 
➢	Analysis: Analyzing something 
 
•	So together it means to find or analyze the most important parts of some entity. 
 
•	In ML entity is data and the job of PCA is to extract the most important features from the data. 
 
•	PCA or Principal Component Analysis is an age-old Machine Learning algorithm and its main use has been for dimensionality reduction. 
 
•	PCA is a mathematical technique that allows you to engineer new features from your given dataset such that the new features will be smaller in dimensions but will be able to represent the original features so these reduced features (the new ones) can be passed to a Machine Learning model to and still get reasonable results whilst drastically reducing complexity. 
 
•	There is one clarification needed here, PCA does not drop any data as most people falsely believe, it creates a linear combination of the given data such that the resultant data is a very close (if not exact) representation of the original data. 
 
•	PCA is a statistical technique to convert high dimensional data to low dimensional data by selecting the most important features that capture maximum information about the dataset. 
 
•	The features are selected on the basis of variance that they cause in the output. 

IMPLEMENTATION: 
 
 
Step-1: Import Libraries 
 
 
import pandas as pd import numpy as np import matplotlib.pyplot as plt %matplotlib inline from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
 
Step-2: Loading data 
 
 
from sklearn.datasets import load_breast_cancer data=load_breast_cancer() 
data.keys() 
 
# Check the output classes 
print(data['target_names']) 
 
# Check the input attributes print(data['feature_names']) 
 
Output: 
 
 
 
 
Step-3: Apply PCA 
 
 
# construct a dataframe using pandas 
df1=pd.DataFrame(data['data'],columns=data['feature_names']) # Scale data before applying PCA scaling=StandardScaler() 
 
# Use fit and transform method scaling.fit(df1) 
Scaled_data=scaling.transform(df1) 
 
# Set the n_components=3 principal=PCA(n_components=3) principal.fit(Scaled_data) 
x=principal.transform(Scaled_data) 
 
# Check the dimensions of data after PCA 
print(x.shape) 
 
Output: 
 
(569,3) 
 
Step-4: Check components 
# Check the values of eigen vectors # prodeced by principal components 
principal.components_ 
 
Output: 
 
 
 
 
 
Step-5: Plot the components(Visualization) 
plt.figure(figsize=(10,10)) 

plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma') plt.xlabel('pc1') plt.ylabel('pc2') 
 
Output: 
 
 
 
 
 
 
For three principal components, we need to plot a 3d graph. x[:,0] signifies the first principal component. Similarly, x[:,1] and x[:,2] represent the second and the third principal component. 
 
 
# import relevant libraries for 3d graph from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(figsize=(10,10)) 
 
# choose projection 3D for creating a 3D graph 
axis = fig.add_subplot(111, projection='3d') 
 
 
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3 axis.scatter(x[:,0],x[:,1],x[:,2], 
c=data['target'],cmap='plasma') axis.set_xlabel("PC1", fontsize=10) axis.set_ylabel("PC2", fontsize=10) axis.set_zlabel("PC3", fontsize=10) 
 
 
 
 
 
Step-6: Calculate variance ratio 
 
 
# check how much variance is explained by each principal component print(principal.explained_variance_ratio_) 
 
Output: 
 
 
 
[0.44272026, 0.18971182, 0.09393163]  






PRACTICAL 4

AIM : Study and implement simple linear regression. 
• Simple linear regression is a method used to model the relationship between a dependent variable and an independent variable. In this method, we assume that there is a linear relationship between the two variables, and we try to estimate the equation of the line that best fits the data. 
 
Implementation of Simple linear regression: 

   
 





PRACTICAL 5
 
AIM : Write a program to demonstrate the working of the decision tree-based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample. 
 
 
  

  
 
  
PRACTICAL:6

AIM: Write a program to implement the Naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets. 
 
from sklearn import preprocessing 
 
#Generating the Gaussian Naive Bayes model from sklearn.naive_bayes import GaussianNB 
 
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast',' Sunny','Sunny', 
'Rainy','Sunny','Overcast','Overcast','Rainy'] 
humidity=['High','High','High','Medium','Low','Low','Low','Medium','Low', 
'Medium','Medium','Medium','High','Medium'] 
 
batfirst=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes',' Yes','No'] 
 
le = preprocessing.LabelEncoder() weather_encoded=le.fit_transform(weather) hum_encoded=le.fit_transform(humidity) label=le.fit_transform(batfirst) print(weather_encoded,hum_encoded,label) 
 
 
 
 
features=list(zip(weather_encoded,hum_encoded)) 
model = GaussianNB() model.fit(features,label) 


 
 
print("Enter Weather and Humidtity conditions : ") w,h=map(int, input().split()) 
 
 
 
predicted= model.predict([[w,h]]) print(predicted) 
 
 















 
PRACTICAL 7 
 
AIM : Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Compute the accuracy of the classifier, considering few test data sets. 
 
 
 
 
 
 

PRACTICAL:8 
 
AIM: Write a program to implement SVM algorithm to classify the iris data set. Compute the accuracy of the classifier, considering few test data sets. 
 from sklearn import datasets from sklearn.model_selection import train_test_split from sklearn import svm 
from sklearn import metrics 
 cancer = datasets.load_breast_cancer() 
 
print("Features: ", cancer.feature_names) print("Labels: ", cancer.target_names) 
 
 
 
 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) 



clf = svm.SVC(kernel='linear') # Linear Kernel print(y_train.shape) 
 
clf.fit(X_train, y_train) 
 
 
y_pred = clf.predict(X_test) 
 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) print("Precision:",metrics.precision_score(y_test, y_pred)) print("Recall:",metrics.recall_score(y_test, y_pred)) 
 
 
 
 
 



PRACTICAL:9

Write a program to implement and check Sklearn’s K-Fold, Shuffled K-fold, Repeated K-Fold and Leave-One-Out validation technique for appropriate classification algorithm and dataset.

 
 
 
 





PRACTICAL:10

Write a program to implement k-Means clustering algorithm for a sample training data set stored as a .CSV file.
 
 
 
            
