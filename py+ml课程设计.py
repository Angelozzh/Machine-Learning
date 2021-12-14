import pandas as pd
import numpy as np

#引入训练集
train = pd.read_csv('C:/Pythonworkfile/titanic/train.csv')
test = pd.read_csv('C:/Pythonworkfile/titanic/test.csv')

#Python数据分析部分
#数据统计
train.info()
train.Age.mean()
train.Age.min()
train.Age.max()
train.Age.var()
train.Age.skew()
train.Age.kurt()
train.Fare.max()
train['Survived'].corr(train['Age'])
train['Survived'].corr(train['Pclass'])
train['Survived'].corr(train['SibSp'])
train['Survived'].corr(train['Parch'])
train['Survived'].corr(train['Fare'])
train['Cabin'].value_counts()
train['Embarked'].value_counts()

#填充缺失值
train['Age'] = train.Age.fillna(train.Age.mean())    
train['Cabin'] = train.Cabin.fillna('NaN')    
train['Embarked'] = train.Embarked.fillna('S')    
train.info()
test['Age'] = test.Age.fillna(test.Age.mean()) 
test.info()
#数据可视化
import matplotlib.pyplot as plt
plt.rc("font",family="SimHei") #正常显示汉字而不是方格
#柱状图展示训练集中男女中生存人数对比
sex = train.groupby('Sex')['Survived'].sum()
sex.plot.bar()
plt.xticks(range(2),['女性','男性'],rotation=0)
plt.ylabel('存活人数')
plt.show()
#柱状图展示训练集中男女生存与死亡人数的比较
train.groupby(['Sex','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.xticks(range(2),['女性','男性'],rotation=0)
plt.title('存活/死亡人数比较')
plt.ylabel('总人数')
plt.show()
#柱状图展示训练集中不同等级客舱生存与死亡人数的比较
train.groupby(['Pclass','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.xticks(rotation=0)
plt.title('存活/死亡人数比较')
plt.ylabel('总人数')
plt.show()
#密度图展示训练集中不同年龄的存活情况
train.Age[train.Survived == 1].plot(kind='kde')
train.Age[train.Survived == 0].plot(kind='kde')
plt.xlabel("年龄")
plt.ylabel("密度") 
plt.title("乘客年龄分布")
plt.legend(('存活', '死亡'),loc='best')
plt.show()
#柱状图展示训练集中兄弟姐妹数量不同生存与死亡人数的比较
train.groupby(['SibSp','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.xticks(rotation=0)
plt.title('存活/死亡人数比较')
plt.xlabel('兄弟姐妹个数')
plt.ylabel('总人数')
plt.legend(('死亡', '存活'),loc='best')
plt.show()
#柱状图展示训练集中父母子女数量不同生存与死亡人数的比较
train.groupby(['Parch','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.xticks(rotation=0)
plt.title('存活/死亡人数比较')
plt.xlabel('父母子女个数')
plt.ylabel('总人数')
plt.legend(('死亡', '存活'),loc='best')
plt.show()
#密度图展示训练集中不同票价的存活情况
train.Fare[train.Survived == 1].plot(kind='kde')
train.Fare[train.Survived == 0].plot(kind='kde')
plt.xlabel("票价")
plt.ylabel("密度") 
plt.title("乘客年龄分布")
plt.legend(('存活', '死亡'),loc='best')
plt.show()
#柱状图展示训练集中不同登船港口生存与死亡人数的比较
train.groupby(['Embarked','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.xticks(rotation=0)
plt.title('存活/死亡人数比较')
plt.xlabel('登船港口')
plt.ylabel('总人数')
plt.legend(('死亡', '存活'),loc='best')
plt.show()
#散点图展示船舱等级与票价之间的关系
plt.scatter(train.Pclass, train.Fare)
plt.xlabel('船舱等级（1为最高等）')
plt.ylabel("票价")                         
plt.grid(b=True, which='major', axis='y') 
plt.title("船舱等级-票价关系")
plt.show()
#箱型图查看存活与死亡人员的年龄情况
train.Age[train.Survived == 1].plot(kind='box')
plt.title('存活人员Age箱型图')
plt.ylabel('年龄')
plt.show()
train.Age[train.Survived == 0].plot(kind='box')
plt.title('死亡人员Age箱型图')
plt.ylabel('年龄')
plt.show()


#机器学习部分 使用决策树CART算法
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV,cross_val_score 
from sklearn.model_selection import train_test_split

#去除不需要的列
train.drop(["Name","Ticket","Fare","Cabin","Embarked"],inplace = True,axis=1)
test.drop(["Name","Ticket","Fare","Cabin","Embarked"],inplace = True,axis=1)

#决策树只能识别数值，需要对字符串进行类型转换
train["Sex"] = (train["Sex"] == "male").astype("int")
test["Sex"] = (test["Sex"] == "male").astype("int")

#调参 寻找最佳参数
#分割数据集用于交叉验证
x = train.iloc[:,train.columns != "Survived"]
y = train.iloc[:,train.columns == "Survived"]
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size = 0.3)

#网格搜索最佳参数
parameters = {"min_samples_leaf":[*range(1,50,5)],"min_impurity_decrease":[*np.linspace(0,0.5,50)]}
clf = DecisionTreeClassifier(random_state=25,max_depth=3)
GS = GridSearchCV(clf,parameters,cv=10)
GS = GS.fit(Xtrain,Ytrain)
GS.best_params_

#按最优参数生成决策树
model = DecisionTreeClassifier(max_depth=20,min_impurity_decrease=0.0, min_samples_leaf=1)
model.fit(Xtrain, Ytrain)
prediction = model.predict(test)
result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':prediction.astype(np.int32)})
result.to_csv("DecisionTree_predictions.csv", index=False)

