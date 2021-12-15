import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


new_dataset="Online-Retail-cleaned.csv"
df=pd.read_csv(new_dataset)

df_numeric=df[['UnitPrice','Quantity','FinalPrice']]
print(f'mean :{df_numeric.mean()}')
print(f'median :{df_numeric.median()}')
print(f'std :{df_numeric.std()}')
print(f'var :{df_numeric.var()}')



new_df = df[['CustomerID','InvoiceNo','StockCode','Quantity','FinalPrice','InvoiceDate','Country']]
avg_quan = new_df[['Quantity','FinalPrice','Country','InvoiceNo','CustomerID']]
# Top 10 customers sales overall countries sorted by FinalPrice
avg_sum = avg_quan.groupby(['Country','CustomerID']).sum()
avg_sum.sort_values('FinalPrice',ascending=False).head(10)
# average of paid amount by each customer at each country ordered by number of invoices.
avg_cus = df[['Quantity','FinalPrice','Country','InvoiceNo']].copy()

x = avg_cus.groupby(['Country','InvoiceNo']).sum()

x['Ones']=1
y = x.groupby('Country').sum()
y['AVG'] = y['FinalPrice'] / y['Ones']
y.sort_values(['Ones','AVG'],ascending=False,inplace=True)
y.head()
y['FinalPrice'].sum() / y['Quantity'].sum()
y['AVG'].plot(kind='bar',figsize=(10,5),title='Average amount paid by the customer over all countries')
plt.ylabel('AVG')
plt.xlabel('Country')
plt.show()


#top 20 products by quantity and finalprice
sns.set_style('whitegrid')
Top20Quan = df.groupby('Description')['Quantity'].agg('sum').sort_values(ascending=False)[0:20]
Top20Price = df.groupby('Description')['FinalPrice'].agg('sum').sort_values(ascending=False)[0:20]

#creating the subplot
fig,axs = plt.subplots(nrows=2, ncols=1, figsize = (10,15))
plt.subplots_adjust(hspace = 0.3)
fig.suptitle('Best Selling Products by Amount and Value', fontsize=25, x = 0.6, y = 0.98)
sns.barplot(x=Top20Quan.values, y=Top20Quan.index, ax= axs[0]).set(xlabel='Total amount of sales')
axs[0].set_title('By Amount', size=16, fontweight = 'bold')
sns.barplot(x=Top20Price.values, y=Top20Price.index, ax= axs[1]).set(xlabel='Total value of sales')
axs[1].set_title('By Value', size=16, fontweight = 'bold')
plt.show()



#plotting the qunatity vs unitprice
Corr = sns.jointplot(x="Quantity", y="UnitPrice", data = df[df.FinalPrice>0], height = 7)
Corr.fig.suptitle("UnitPrice and Quantity Comparison", fontsize = 15, y = 0.98)
plt.show()
#
# #creating the pie chart
df.groupby('Day of week')['FinalPrice'].sum().plot(kind = 'pie', autopct = '%.2f%%', figsize=(7,7)).set(ylabel='')
plt.title('Percantages of Sales Value by Day of Week', fontsize = 15)
plt.show()
#
heatmap_df = df.pivot_table(index = 'InvoiceMonth',columns = 'Day of week', values = 'FinalPrice', aggfunc='sum')
print(heatmap_df)
plt.figure(figsize = (10,6))
sns.heatmap(heatmap_df, cmap = 'vlag').set(xlabel='', ylabel='')
plt.title('Sales Value per Month and Day of Week', fontsize = 15)
plt.show()

df1=df.copy()
df_teams_countries_customers = df1.groupby(by="Country").agg({'FinalPrice':'count'}).reset_index().sort_values(by='FinalPrice', ascending=False)
ax = df_teams_countries_customers.plot.bar(x='Country')
import geopandas as gpd

df_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

print(f"{type(df_world)}, {df_world.geometry.name}")

print(df_world.head())

print(df_world.geometry.geom_type.value_counts())
df_world.plot(figsize=(10,6))
df_world_teams = df_world.merge(df_teams_countries_customers, how="left", left_on=['name'], right_on=['Country'])
print("Type of DataFrame : ", type(df_world_teams), df_world_teams.shape[0])
df_world_teams.head()
ax = df_world["geometry"].boundary.plot(figsize=(20,16))
df_world_teams.plot( column="FinalPrice", ax=ax, cmap='OrRd',
                     legend=True, legend_kwds={"label": "Customers", "orientation":"horizontal"})
ax.set_title("final price at countries")
plt.show()









