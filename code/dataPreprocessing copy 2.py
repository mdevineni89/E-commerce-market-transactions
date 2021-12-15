import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import datetime
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px


dataset='Online Retail .csv'
df=pd.read_csv(dataset,dtype={'CustomerID':'Int64'})
print(df.info())

#checking the total number of missing values
print(df.isnull().sum().sort_values(ascending=False))
#
# '''We see that there are some missing values for Customers ID and Description.
# The rows with any of these missing values will therefore be removed. '''
# #droping the rows having missing values
df=df.dropna(axis=0)
#
# #understanding the data in a more descriptive manner
print(df.describe().round(2))
# '''
# Quantity has negative values
# Unit Price has zero values,
# we have some odd and irregular values in the 'UnitPrice' and 'Quantity' columns
# that we will find and remove to prevent them from negatively affect our analysis.
# In the 'StockCode' variable we can see that some of the transaction are not
# actually products, but they are some costs or fees regarding to the post or
# bank or other transactions that we don't really need in our data'''
#
df.drop(df[df['Quantity'] < 0].index, inplace = True)
df.drop(df[df['UnitPrice'] == 0].index, inplace = True)
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'],format='%m/%d/%y %H:%M')

#Fixing Duplicate CustomerIDs
for i,v in df.groupby('CustomerID')['Country'].unique().items():
    if len(v)>1:
        df.Country[df['CustomerID'] == i] = df.Country[df['CustomerID'] == i].mode()[0]
print(df['Country'].unique())

df = df[(np.abs(st.zscore(df['UnitPrice']))<3) & (np.abs(st.zscore(df['Quantity']))<5)]

q1_quantity, q2_quantity, q3_quantity = df['Quantity'].quantile([0.25,0.5,0.75])
iqr_quantity = q3_quantity - q1_quantity
lower_limit_quantity = q1_quantity - 1.5*iqr_quantity
upper_limit_quantity = q3_quantity + 1.5*iqr_quantity
print(f'Q1 and Q3 of Quantity is {q1_quantity:.2f} & {q3_quantity:.2f} .')
print(f'IQR for the Quantity {iqr_quantity:.2f} .')
print(f'Any Quantity < {lower_limit_quantity:.2f} and Quantity > {upper_limit_quantity:.2f} is an outlier')

sns.set_style('darkgrid')
sns.boxplot(y=df['Quantity'])
plt.title('Boxplot of Quantity')
plt.show()
df= df[(df['Quantity'] >= lower_limit_quantity) & (df['Quantity'] <= upper_limit_quantity)]
sns.set_style('darkgrid')
sns.boxplot(y=df['Quantity'])
plt.title('Boxplot of quantity after removing outliers')
plt.show()

q1_UnitPrice, q2_UnitPrice, q3_UnitPrice = df['UnitPrice'].quantile([0.25,0.5,0.75])
iqr_UnitPrice = q3_UnitPrice - q1_UnitPrice
lower_limit_UnitPrice = q1_UnitPrice - 1.5*iqr_UnitPrice
upper_limit_UnitPrice = q3_UnitPrice + 1.5*iqr_UnitPrice
print(f'Q1 and Q3 of UnitPrice is {q1_UnitPrice:.2f} & {q3_UnitPrice:.2f} .')
print(f'IQR for the UnitPrice {iqr_UnitPrice:.2f} .')
print(f'Any UnitPrice < {lower_limit_UnitPrice:.2f} and UnitPrice > {upper_limit_UnitPrice:.2f} is an outlier')

sns.set_style('darkgrid')
sns.boxplot(y=df['UnitPrice'])
plt.title('Boxplot of UnitPrice')
plt.show()
df = df[(df['UnitPrice'] >= lower_limit_UnitPrice) & (df['UnitPrice'] <= upper_limit_UnitPrice)]
sns.set_style('darkgrid')
sns.boxplot(y=df['UnitPrice'])
plt.title('Boxplot of UnitPrice after removing outliers')
plt.show()

q1_UnitPrice2, q2_UnitPrice2, q3_UnitPrice2 = df['UnitPrice'].quantile([0.25,0.5,0.75])
iqr_UnitPrice2 = q3_UnitPrice2 - q1_UnitPrice2
lower_limit_UnitPrice2 = q1_UnitPrice2 - 1.5*iqr_UnitPrice2
upper_limit_UnitPrice2 = q3_UnitPrice2 + 1.5*iqr_UnitPrice2
print(f'Q1 and Q3 of UnitPrice is {q1_UnitPrice2:.2f} & {q3_UnitPrice2:.2f} .')
print(f'IQR for the UnitPrice {iqr_UnitPrice2:.2f} .')
print(f'Any UnitPrice < {lower_limit_UnitPrice2:.2f} and UnitPrice > {upper_limit_UnitPrice2:.2f} is an outlier')

df = df[(df['UnitPrice'] >= lower_limit_UnitPrice2) & (df['UnitPrice'] <= upper_limit_UnitPrice2)]
sns.set_style('darkgrid')
sns.boxplot(y=df['UnitPrice'])
plt.title('Boxplot of UnitPrice after removing outliers')
plt.show()






#PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
features=['Quantity','UnitPrice']
x_f=df[features].values
X=StandardScaler().fit_transform(x_f)
H_pca = np.matmul(x_f.T,x_f)
_,d_PCA,_ = np.linalg.svd(H_pca)
print(f'Transformed X singular values: {d_PCA}')
print(f'Transformed X condition number: {np.linalg.cond(x_f)}')

sns.heatmap(df.corr(),annot=True)
plt.title("correlation coefficient between features - orginal feature space")
plt.show()
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print("explained variance ratio:",pca.explained_variance_ratio_)
print("shape:",X_pca.shape)
plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1,1), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1,1))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
df1=pd.DataFrame(X_pca).corr()
a,b = X_pca.shape
col = []
for i in range(b):
    col.append(f'Principle column {i+1}')

# sns.heatmap(df1,annot=True,xticklabels=col,yticklabels=col)
# plt.title("correlation coefficient")
# plt.show()
df_PCA = pd.DataFrame(data=X_pca, columns=col)
print("old one:",df.head().to_string)
print("new one:",df_PCA.head().to_string)

kstest_Quantity = st.kstest(df['Quantity'],'norm')
kstest_UnitPrice = st.kstest(df['UnitPrice'],'norm')

print(f"K-S test: statistics={kstest_UnitPrice[0]}, p-value={kstest_UnitPrice[1]}")
print(f"K-S test:  dataset looks {'Normal' if kstest_UnitPrice[1] > 0.01 else 'Non-Normal'}")
print(f"K-S test: statistics={kstest_Quantity[0]}, p-value={kstest_Quantity[1]}")
print(f"K-S test: looks {'Normal' if kstest_Quantity[1] > 0.01 else 'Non-Normal'}")


shapiro_test_Quantity = st.shapiro(df['Quantity'])
shapiro_test_UnitPrice = st.shapiro(df['UnitPrice'])

print(f"Shapiro test: statistics={shapiro_test_Quantity[0]:.5f}, p-value={shapiro_test_Quantity[1]:.5f}")
print(f"Shapiro test: dataset looks {'Normal' if shapiro_test_Quantity[1] > 0.01 else 'Non-Normal'}")
print(f"Shapiro test: statistics={shapiro_test_UnitPrice[0]:.5f}, p-value={shapiro_test_UnitPrice[1]:.5f}")
print(f"Shapiro test: dataset looks {'Normal' if shapiro_test_UnitPrice[1] > 0.01 else 'Non-Normal'}")


da_test_Quantity = st.normaltest(df['Quantity'])
da_test_UnitPrice = st.normaltest(df['UnitPrice'])

print(f"da_k_squared test: statistics={da_test_UnitPrice[0]:.5f}, p-value={da_test_UnitPrice[1]:.5f}")
print(f"da_k_squared test: dataset looks {'Normal' if da_test_UnitPrice[1] > 0.01 else 'Non-Normal'}")
print(f"da_k_squared test: statistics={da_test_Quantity[0]:.5f}, p-value={da_test_Quantity[1]:.5f}")
print(f"da_k_squared test: dataset looks {'Normal' if da_test_Quantity[1] > 0.01 else 'Non-Normal'}")


#adding features
europe = ['Germany','Spain', 'France', 'Italy', 'Netherlands', 'Norway', 'Sweden','Czech Republic', 'Finland',
          'Denmark', 'Switzerland', 'United Kingdom', 'Poland', 'Greece','Austria', 'Belgium','Portugal', 'Lithuania', 'Iceland','EIRE',
          'Channel Islands','Cyprus','European Community','Malta']
asia = ['Bahrain', 'United Arab Emirates','Saudi Arabia','Lebanon','Japan','Singapore','Israel']
other=['Australia','Canada','Unspecified','Brazil','USA','RSA']

conti = []
for i in df['Country']:
    if i in europe:
        conti.append('Europe')
    elif i in asia:
        conti.append('Asia')

    elif i in other:
        conti.append('Other')

df['Continent']=conti
df['FinalPrice'] = df['Quantity']*df['UnitPrice']
df['InvoiceMonth'] = df['InvoiceDate'].apply(lambda x: x.strftime('%B'))
df['Day of week'] = df['InvoiceDate'].dt.day_name()
print(df.head())
print(np.max(df['Quantity']))
print(np.min(df['Quantity']))
print(df['Quantity'])

#================================================



# df.to_csv(r'/Users/sunny/Desktop/ecommerce project/Online-Retail-cleaned.csv', date_format = '%Y-%m-%d %H:%M', index = False)










