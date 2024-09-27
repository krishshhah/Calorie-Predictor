import talib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import random
import statsmodels.api as sm
#from sklearn.trees import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor

random.seed(123) #ensures same random numbers generated everytime

days_of_week = [i % 7 + 1 for i in range(364)]

# Generate a list of 52 random integers
random_ints_monday = random.sample(range(2894, 3452), 52)
random_ints_tuesday = random.sample(range(3141, 3781), 52)
random_ints_wednesday = random.sample(range(2983, 3342), 52)
random_ints_thursday = random.sample(range(3341, 4059), 52)
random_ints_friday = random.sample(range(2874, 3983), 52)
random_ints_saturday = random.sample(range(3421, 3823), 52)
random_ints_sunday = random.sample(range(2999, 3412), 52)


# Create a list of 364 zeros to represent calories for each day of the year
all_calories = [0] * 364

# Assign random_ints_tuesday to all Tuesdays in all_calories
for i in range(len(days_of_week)):
    if days_of_week[i] == 1:
        all_calories[i] = random_ints_monday.pop(0)
    if days_of_week[i] == 2:
        all_calories[i] = random_ints_tuesday.pop(0)
    if days_of_week[i] == 3:
        all_calories[i] = random_ints_wednesday.pop(0)
    if days_of_week[i] == 4:
        all_calories[i] = random_ints_thursday.pop(0)
    if days_of_week[i] == 5:
        all_calories[i] = random_ints_friday.pop(0)
    if days_of_week[i] == 6:
        all_calories[i] = random_ints_saturday.pop(0)
    if days_of_week[i] == 7:
        all_calories[i] = random_ints_sunday.pop(0)

d = {'day' : days_of_week, 'calories_burned' : all_calories}
df = pd.DataFrame(d)

dates = []
for x in range (0, 364):
    now = datetime.datetime(2023, 10, 24)
    new_date = now - datetime.timedelta(days = (364-x))
    dates.append(new_date)

df['date'] = dates
df.set_index('date', inplace = True)
df = df.drop('day', axis = 1)

df['10d_calories_pct'] = df['calories_burned'].pct_change(10)
df['10d_future_calories'] = df['calories_burned'].shift(-10)
df['10d_future_calories_pct'] = df['10d_future_calories'].pct_change(10)

# df['ma50'] = talib.SMA(df['calories_burned'].values.astype(float), timeperiod = 50)
# df['rsi50'] = talib.RSI(df['calories_burned'].values.astype(float), timeperiod = 50)

feature_names = ['10d_calories_pct'] #current percent change

for n in [14, 30, 50, 200]: # calculates various RSI and SMA for different time periods
    df['ma' + str(n)] = talib.SMA(df['calories_burned'].values.astype(float), timeperiod = n)/ df['calories_burned']
    df['rsi' + str(n)] = talib.RSI(df['calories_burned'].values.astype(float), timeperiod = n)
    
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
    
df = df.dropna()
features = df[feature_names]
targets = df['10d_future_calories_pct'] #future percent change 
feature_and_target_cols = ['10d_future_calories_pct'] + feature_names
feat_targ_df = df[feature_and_target_cols]

corr = feat_targ_df.corr()
sns.heatmap(corr, annot= True)
plt.yticks(rotation=0, size = 14); plt.xticks(rotation=90, size = 14)  # fix ticklabel directions and size
plt.show()  # show the plot

# Add a constant to the features
linear_features = sm.add_constant(features)

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * features.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
#print(linear_features.shape, train_features.shape, test_features.shape)

model = sm.OLS(train_targets, train_features)
results = model.fit()  # fit the model
# print(results.summary())

# examine pvalues
# Features with p <= 0.05 are typically considered significantly different from 0
#print(results.pvalues)

# Make predictions from our model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, test_targets, alpha=0.2, color='r', label='test')

# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

# Set the axis labels and show the plot
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  # show the legend
plt.show()

# plt.scatter(df['ma50'], df['10d_future_calories_pct'])
# plt.show()


days_of_week = pd.get_dummies(df.index.dayofweek,
                              prefix='weekday',
                              drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = df.index

# Join the dataframe with the days of week dataframe
df = pd.concat([df, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend(['weekday_' + str(i) for i in range(1, 5)])
df.dropna(inplace=True)  # drop missing values in-place
print(df.head())

tue = df[df['weekday_1'] == 1]
wed = df[df['weekday_2'] == 1]
thur = df[df['weekday_3'] == 1]
fri = df[df['weekday_4'] == 1]
sat = df[df['weekday_5'] == 1]
sun = df[df['weekday_6'] == 1]

