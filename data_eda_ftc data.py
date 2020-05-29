import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from datetime import datetime, timedelta
import statsmodels.api as sm
file_path = '/home/iris/PycharmProjects/week1/combined_csv.csv'

df = pd.read_csv(file_path, parse_dates = True, encoding = 'utf-8-sig')
df.head()
print(df.columns)
print(df.shape)
df.Recorded_Message_Or_Robocall.value_counts()
df.groupby(['Recorded_Message_Or_Robocall'])['Subject'].value_counts()

df['Violation_Date'] = pd.to_datetime(df['Violation_Date'])
df.sort_values('Violation_Date', inplace=True)
# new data frame with split value columns
df['year'] = df['Violation_Date'].dt.year
df['month'] = df['Violation_Date'].dt.month
df['day'] = df['Violation_Date'].dt.day
df['hour'] =df['Violation_Date'].dt.hour
df['minute'] = df['Violation_Date'].dt.minute

Robocall = df[df['Recorded_Message_Or_Robocall']=='Y']
Nuisance = df[df['Recorded_Message_Or_Robocall']=='N']
Robocall.set_index(['day'])
Nuisance.set_index(['day'])
robo_date = df['day']
robo_count = Robocall.groupby(['day', 'hour']).size().reset_index(name='count')

fml = 'count~(day) + (hour)'
m1 = sm.GLM.from_formula(fml, family = sm.families.Poisson(), data=robo_count)
r1 = m1.fit(scale="X2")
print(r1.summary())

fml = 'count~(day) * (hour)'
m1 = sm.GLM.from_formula(fml, family = sm.families.Poisson(), data=robo_count)
r1 = m1.fit(scale="X2")
print(r1.summary())

robo_count2 = pd.DataFrame()
robo_count2['hour'] = pd.to_datetime(robo_count['hour'])
robo_count2.sort_values('hour', inplace=True)

sns.kdeplot(data=Robocall['hour'], shade=True)
plt.clf()

sns.kdeplot(data=Robocall['day'], shade=True)


