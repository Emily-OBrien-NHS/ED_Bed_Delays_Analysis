import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sqlalchemy import create_engine
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
os.chdir('ED/Bed Delays Correlations')

#Create the engine
cl3_engine = create_engine('mssql+pyodbc://@cl3-data/DataWarehouse?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
#sql query
query = """
select dischargedate
, sum(datediff(mi,BedRequestedDateTime, DischargeDateTime)) as BedDelayMins
,(sum(case when is4hourbreach = 'n' then 1 else 0 end)*100.0)/count(*) as FourHourPerf
,avg(datediff(mi,arrivaldatetime,dischargedatetime)) as MeanTimeInDept
,avg(timetoinitialassessment) as MeanTimetoTriage
,avg(timetotreatment) as MeanTimetoTreatment
,avg(datediff(mi,arrivaldatetime, managementplandatetime)) as MeanTimeToManagementPlan
,avg(datediff(mi,arrivaldatetime, departurereadydatetime)) as MeanTimetoCRtP
from DataWarehouse.ed.vw_EDAttendance
where DischargeDate between '01-APR-2023' and '31-MAR-2024'
and IsNewAttendance = 'y'
and DischargeDate is not NULL
group by dischargedate
order by DischargeDate desc
"""
#Read query into dataframe
df = pd.read_sql(query, cl3_engine)
#Close the connection
cl3_engine.dispose()

#Remove outlier time to management plan and erroneous negatives
value_cols = ['BedDelayMins', 'FourHourPerf', 'MeanTimeInDept', 'MeanTimetoTriage', 'MeanTimetoTreatment',
                 'MeanTimeToManagementPlan', 'MeanTimetoCRtP']

df = df.loc[(np.abs(stats.zscore(df['MeanTimeToManagementPlan'])) < 3)]
df = df.loc[(df[value_cols] > 0).all(axis=1)]

norm_df = df.copy()
norm_df[value_cols] = (norm_df[value_cols] - norm_df[value_cols].min()) / (norm_df[value_cols].max() - norm_df[value_cols].min())

#Correalton heatmap
cor_mat = df[['BedDelayMins', 'FourHourPerf', 'MeanTimeInDept', 'MeanTimetoTriage', 'MeanTimetoTreatment',
              'MeanTimeToManagementPlan', 'MeanTimetoCRtP']].corr(method='pearson')
fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(cor_mat, robust=True, annot=True, fmt='.2f', square=True, ax=ax)
plt.title('Pearson Coefficients')
plt.savefig('Correlation Matrix.png', bbox_inches='tight')

norm_cor_mat = df[['BedDelayMins', 'FourHourPerf', 'MeanTimeInDept', 'MeanTimetoTriage', 'MeanTimetoTreatment',
              'MeanTimeToManagementPlan', 'MeanTimetoCRtP']].corr(method='pearson')
fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(norm_cor_mat, robust=True, annot=True, fmt='.2f', square=True, ax=ax)
plt.title('Normalised Pearson Coefficients')
plt.savefig('Normalised Correlation Matrix.png', bbox_inches='tight')

#Function to fit a line of best fit to each relationship and save the plot
def model_fit_and_plot(df, x_column, x_name, y_column, y_name, mod_degree):
    #Define x and y data
    x = df[x_column]
    x_plot = np.linspace(min(x), max(x), 1000)
    y = df[y_column]
    #If degree is higher than 1, then will require polynomial transform
    if mod_degree > 1:
        #Transform data
        poly = PolynomialFeatures(degree=mod_degree, include_bias=False)
        x_trans = poly.fit_transform(np.array(x).reshape(-1,1))
        #Transform an array ready for plotting
        x_plot_trans = poly.fit_transform(x_plot.reshape(-1,1))
    else:
        x_trans = np.array(x).reshape(-1,1)
        x_plot_trans = x_plot.reshape(-1,1)
        y = np.array(y).reshape(-1,1)

    #Fit the model
    model = LinearRegression()
    model.fit(x_trans, y)
    #Create the y_plot data
    y_plot = model.predict(x_plot_trans)
    #Create and save the plot
    title = x_name + ' against ' + y_name
    plt.figure()
    plt.title(title)
    plt.scatter(x, y)
    plt.plot(x_plot, y_plot, c='red')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(title + '.png')

model_fit_and_plot(df, 'BedDelayMins', 'Bed Delay (mins)', 'FourHourPerf', 'Four Hour Performance', 3)
model_fit_and_plot(df, 'BedDelayMins', 'Bed Delay (mins)', 'MeanTimeInDept', 'Mean Time in Department', 2)
model_fit_and_plot(df, 'BedDelayMins', 'Bed Delay (mins)', 'MeanTimetoTriage', 'Mean Time to Triage', 2)
model_fit_and_plot(df, 'BedDelayMins', 'Bed Delay (mins)', 'MeanTimetoTreatment', 'Mean Time to Treatment', 1)
model_fit_and_plot(df, 'BedDelayMins', 'Bed Delay (mins)', 'MeanTimeToManagementPlan', 'Mean Time to Management Plan', 1)
model_fit_and_plot(df, 'BedDelayMins', 'Bed Delay (mins)', 'MeanTimetoCRtP', 'Mean Time to CRtP', 1)

model_fit_and_plot(df, 'MeanTimetoTreatment', 'Mean Time to Treatment', 'MeanTimeToManagementPlan', 'Mean Time to Management Plan', 1)
model_fit_and_plot(df, 'MeanTimetoCRtP', 'Mean Time to CRtp', 'MeanTimeToManagementPlan', 'Mean Time to Management Plan', 1)
model_fit_and_plot(df, 'MeanTimetoCRtP', 'Mean Time to CRtp', 'MeanTimetoTreatment', 'Mean Time to Treatment', 1)