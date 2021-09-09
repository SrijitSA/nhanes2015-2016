import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm

plt.style.use("seaborn")
sns.set_theme(context='paper', font_scale=1.4)
pd.set_option('display.max_columns', 30)
sns.set()

###Import the nhanes dataset
floc = "C:/Users/sriji/Documents/Coursera/R Programming/data/"
fname = input("Enter filename:")
fhand = floc + fname + ".csv"

try:
    nhanes = pd.read_csv(fhand)
except:
    print(f"File '{fname}.csv' could not be located in the folder: {floc}"
          f"\nPlease recheck the filename and try again.")
    quit()

print(nhanes.head())
print(nhanes.shape)
print("\n")

## Drop columns that have more than 10% missing values and ones that are unwarranted for analyzing the outcome
nhanes.dropna(thresh=len(nhanes) * 0.9, axis='columns', inplace=True)
nhanes.drop(labels=["SDMVPSU", "SDMVSTRA", "WTINT2YR", "DMDHHSIZ"], axis=1, inplace=True)
print("Number of missing values:", nhanes.isna().sum())
print("\n")

## Rename columns
nhanes.rename(columns={'SEQN': 'id', 'ALQ101': 'alc_drink_year', 'SMQ020': 'smoke_100_cig', 'RIAGENDR': 'gender',
                       'RIDAGEYR': 'age',
                       'RIDRETH1': 'race', 'DMDCITZN': 'citizenship', 'DMDEDUC2': 'edu_lvl',
                       'DMDMARTL': 'marital_status', 'BPXSY1': 'sys_blood_pres', 'BPXDI1': 'dias_blood_pres',
                       'BPXSY2': 'sys_blood_pres2',
                       'BPXDI2': 'dias_blood_pres2', 'BMXWT': 'weight', 'BMXHT': 'height', 'BMXBMI': 'bmi',
                       'BMXLEG': 'leg_length',
                       'BMXARML': 'arm_length', 'BMXARMC': 'arm_circum', 'BMXWAIST': 'waist_circum'}, inplace=True)
print(nhanes.columns)
print("\n")

### Check the uniqueness and value counts for categorical data elements
print(nhanes["alc_drink_year"].value_counts())
print(nhanes["smoke_100_cig"].value_counts())
print(nhanes["gender"].value_counts())
print(nhanes["race"].value_counts())
print(nhanes["citizenship"].value_counts())
print(nhanes["edu_lvl"].value_counts())
print(nhanes["marital_status"].value_counts())
print("\n")

### Replace some of the column data for categorical variables for ease of interpretation.
nhanes["alc_drink_year"].replace({2: 0, 7: np.nan, 9: np.nan}, inplace=True)
print(nhanes["alc_drink_year"].value_counts())
nhanes["smoke_100_cig"].replace({2: 0, 7: np.nan, 9: np.nan}, inplace=True)
print(nhanes["smoke_100_cig"].value_counts())
nhanes["gender_str"] = np.where(nhanes["gender"] == 1, "M", "F")
print(nhanes["gender_str"].value_counts())
nhanes["race"].replace({1: "MexicanAmerican", 2: "OtherHispanic", 3: "Caucasian", 4: "AfricanAmerican", 5: "OtherRace"},
                       inplace=True)
print(nhanes["race"].value_counts())
nhanes["citizenship"].replace({2: 0, 7: np.nan, 9: np.nan}, inplace=True)
print(nhanes["citizenship"].value_counts())
nhanes["edu_lvl"].replace({1: '<9th Grade', 2: '9-11th Grade', 3: 'HS Grad', 4: 'CollegeUndergrad', 5: 'Graduate',
                           9: np.nan}, inplace=True)
print(nhanes["edu_lvl"].value_counts())
nhanes["marital_status"].replace({1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never Married',
                                  6: 'Live-In', 77: np.nan, 99: np.nan}, inplace=True)
print(nhanes["marital_status"].value_counts())


################################# Data exploration phase ####################################################

print("\n")
print("Analysis of the potential outcome variables")
print("Systolic Blood Pressure:")
print(nhanes["sys_blood_pres"].describe().loc[['count', 'mean', 'min', '50%', 'max']])
print("\n")
print("Diastolic Blood Pressure:")
print(nhanes["dias_blood_pres"].describe().loc[['count', 'mean', 'min', '50%', 'max']])
print("\n")
print("Systolic Blood Pressure 2nd measurement:")
print(nhanes["sys_blood_pres2"].describe().loc[['count', 'mean', 'min', '50%', 'max']])
print("\n")
print("Diastolic Blood Pressure 2nd measurement:")
print(nhanes["dias_blood_pres2"].describe().loc[['count', 'mean', 'min', '50%', 'max']])

# fig, ax = plt.subplots(2, 1, figsize=(8,8))
# sns.scatterplot(x=nhanes["sys_blood_pres"], y=nhanes["sys_blood_pres2"], ax=ax[0]).set_title(
#     "Comparing both systolic blood pressure measurements")
# sns.scatterplot(x=nhanes["dias_blood_pres"], y=nhanes["dias_blood_pres2"], ax=ax[1]).set_title(
#     "Comparing both diastolic blood pressure measurements")
print("Because both systolic blood pressure measurements are heavily correlated, the focus will be on the first "
      "measurement. Same with diastolic blood pressure measurements.")
nhanes.drop(labels=["sys_blood_pres2", "dias_blood_pres2"], axis=1, inplace=True)


## Check the distributions of some of the continuous data
print("\b")
# fig, ax = plt.subplots(3, 3, figsize=(8,8))
# fig.suptitle("All continuous variables data distributions")
# sns.histplot(data=nhanes, ax=ax[0,0], x="sys_blood_pres", kde=True, alpha=0.75)
# ax[0,0].axvline(x=np.mean(nhanes["sys_blood_pres"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[0,1], x="age", kde=True, alpha=0.75)
# ax[0,1].axvline(x=np.mean(nhanes["age"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[0,2], x="weight", kde=True, alpha=0.75)
# ax[0,2].axvline(x=np.mean(nhanes["weight"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[1,0], x="height", kde=True, alpha=0.75)
# ax[1,0].axvline(x=np.mean(nhanes["height"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[1,1], x="bmi", kde=True, alpha=0.75)
# ax[1,1].axvline(x=np.mean(nhanes["bmi"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[1,2], x="leg_length", kde=True, alpha=0.75)
# ax[1,2].axvline(x=np.mean(nhanes["leg_length"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[2,0], x="arm_length", kde=True, alpha=0.75)
# ax[2,0].axvline(x=np.mean(nhanes["arm_length"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[2,1], x="arm_circum", kde=True, alpha=0.75)
# ax[2,1].axvline(x=np.mean(nhanes["arm_circum"]), color='black', linewidth=1.5)
# sns.histplot(data=nhanes, ax=ax[2,2], x="waist_circum", kde=True, alpha=0.75)
# ax[2,2].axvline(x=np.mean(nhanes["waist_circum"]), color='black', linewidth=1.5)
print("Based on the distributions of continuous variables, the following observations can be made:")
print("1. Systolic blood pressure is slightly right-skewed.")
print(f"2. Age has a uniform distribution with the mean at the center at {round(np.mean(nhanes.age))} years.")
print("3. Weight is slightly right-skewed.")
print("4. Height is normally distributed with the mean at the center.")
print("5. BMI is slightly right-skewed.")
print("6. Leg Length is normally distributed with the mean at the center.")
print("7. Arm Length is normally distributed with the mean at the center.")
print("8. Arm Circumference is normally distributed with the mean at the center.")
print("9. Waist Circumference is sligtly right-skewed.")


## Does systolic blood pressure vary with regard to the various categorical variables: ['alc_drink_year',
## 'smoke_100_cig', 'race', 'citizenship', 'edu_lvl', 'marital_status']

# fig, ax = plt.subplots(2, 3, sharey=True, figsize=(10,10))
# fig.suptitle("Boxplots of Systolic Blood Pressure at various levels of categorical variables")
# sns.boxplot(x=nhanes["alc_drink_year"], y=nhanes["sys_blood_pres"], ax=ax[0,0])
# sns.boxplot(x=nhanes["smoke_100_cig"], y=nhanes["sys_blood_pres"], ax=ax[0,1])
# sns.boxplot(x=nhanes["race"], y=nhanes["sys_blood_pres"], ax=ax[0,2])
# sns.boxplot(x=nhanes["citizenship"], y=nhanes["sys_blood_pres"], ax=ax[1,0])
# sns.boxplot(x=nhanes["edu_lvl"], y=nhanes["sys_blood_pres"], ax=ax[1,1])
# sns.boxplot(x=nhanes["marital_status"], y=nhanes["sys_blood_pres"], ax=ax[1,2])

# fig, ax = plt.subplots(2, 3, figsize=(10,10))
# fig.suptitle("Histplots of Systolic Blood Pressure at various levels of categorical variables")
# sns.histplot(x=nhanes["sys_blood_pres"], hue=nhanes["alc_drink_year"], ax=ax[0,0], element="step")
# sns.histplot(x=nhanes["sys_blood_pres"], hue=nhanes["smoke_100_cig"], ax=ax[0,1], element="step")
# sns.histplot(x=nhanes["sys_blood_pres"], hue=nhanes["edu_lvl"], ax=ax[0,2], element="step")
# sns.histplot(x=nhanes["sys_blood_pres"], hue=nhanes["race"], ax=ax[1,0], element="step")
# sns.histplot(x=nhanes["sys_blood_pres"], hue=nhanes["citizenship"], ax=ax[1,1], element="step")
# sns.histplot(x=nhanes["sys_blood_pres"], hue=nhanes["marital_status"], ax=ax[1,2], element="step")


## Is smoking related to education level?
print("\n")
smoking_edulvl = pd.crosstab(index=nhanes["smoke_100_cig"], columns=nhanes["edu_lvl"])
print(smoking_edulvl)
stat, p, dof, expected = chi2_contingency(smoking_edulvl)
alpha = 0.05
print("\b")
print("Degrees of freedom:", str(dof))
print("X-squared:", str(stat))
print("p-value:", str(p))
print("Expected values:", expected)
if p >= 0.05:
    print("There is no association between Smoking and Education Level. Therefore, we cannot reject the null hypothesis "
          "at the 5% level.")
else:
    print(f"There is some evidence of a statistically significant association (p-value: {round(p,2)})  between Smoking "
          f"and Education Level. Therefore, we can reject the null hypothesis in favor of the alternate hypothesis.")


## Is alcohol consumption related to education level?
print("\n")
alc_edulvl = pd.crosstab(index=nhanes["alc_drink_year"], columns=nhanes["edu_lvl"])
print(alc_edulvl)
stat, p, dof, expected = chi2_contingency(alc_edulvl)
alpha = 0.05
print("\b")
print("Degrees of freedom:", str(dof))
print("X-squared:", str(stat))
print("p-value:", str(p))
print("Expected values:", expected)
if p >= 0.05:
    print("There is no association between Alcohol consumption and Education Level. Therefore, we cannot reject the "
          "null hypothesis at the 5% level.")
else:
    print(f"There is some evidence of a statistically significant association (p-value: {round(p,2)})  between Alcohol "
          f"consumption and Education Level. Therefore, we can reject the null hypothesis in favor of the alternate "
          f"hypothesis.")


## Is alcohol consumption related to gender?
print("\n")
alc_gender = pd.crosstab(index=nhanes["alc_drink_year"], columns=nhanes["gender"])
print(alc_gender)
stat, p, dof, expected = chi2_contingency(alc_gender)
alpha = 0.05
print("\b")
print("Degrees of freedom:", str(dof))
print("X-squared:", str(stat))
print("p-value:", str(p))
print("Expected values:", expected)
if p >= 0.05:
    print("There is no association between Alcohol consumption and Gender. Therefore, we cannot reject the "
          "null hypothesis at the 5% level.")
else:
    print(f"There is some evidence of a statistically significant association (p-value: {round(p,2)}) between Alcohol "
          f"consumption and Gender. Therefore, we can reject the null hypothesis in favor of the alternate hypothesis.")


## Is the estimated mean of systolic blood pressure different for males vs. females?
print("\n")
statistic, pvalue = ttest_ind(nhanes["sys_blood_pres"][nhanes["gender"] == 1],
                              nhanes["sys_blood_pres"][nhanes["gender"] == 2], equal_var=False, nan_policy='omit')
print("Welch's Two Sample t-test")
print("t =", str(statistic))
print("p-value =", str(pvalue))
if pvalue <= alpha:
    print("Reject the null hypothesis H0 in favor of the alternate hypothesis H1: Systolic blood pressure differs with"
          " gender at the 5% significance level")
else:
    print("Not enough evidence to reject the null hypothesis H0.")
    print("Therefore, we conclude that there is no evidence of a difference in Systolic blood pressure between males and"
          " females")
# sns.boxplot(x= nhanes["gender_str"], y=nhanes["sys_blood_pres"]).set_title("Systolic Blood Pressure for Males and "
#                                                                            "Females")


## Is the estimated mean of systolic blood pressure different for smokers vs. non-smokers?
print("\n")
statistic, pvalue = ttest_ind(nhanes["sys_blood_pres"][nhanes["smoke_100_cig"] == 0],
                              nhanes["sys_blood_pres"][nhanes["smoke_100_cig"] == 1], equal_var=False, nan_policy='omit')
print("Welch's Two Sample t-test")
print("t =", str(statistic))
print("p-value =", str(pvalue))
if pvalue <= alpha:
    print("Reject the null hypothesis H0 in favor of the alternate hypothesis H1: Systolic blood pressure differs with"
          " smoking at the 5% significance level")
else:
    print("Not enough evidence to reject the null hypothesis H0.")
    print("Therefore, we conclude that there is no evidence of a difference in Systolic blood pressure between smokers "
          "and non-smokers")


## Is the estimated mean of systolic blood pressure different for alcohol consumption?
print("\n")
statistic, pvalue = ttest_ind(nhanes["sys_blood_pres"][nhanes["alc_drink_year"] == 0],
                              nhanes["sys_blood_pres"][nhanes["alc_drink_year"] == 1], equal_var=False, nan_policy='omit')
print("Welch's Two Sample t-test")
print("t =", str(statistic))
print("p-value =", str(pvalue))
if pvalue <= alpha:
    print("Reject the null hypothesis H0 in favor of the alternate hypothesis H1: Systolic blood pressure differs with"
          " alcohol consumption at the 5% significance level")
else:
    print("Not enough evidence to reject the null hypothesis H0.")
    print("Therefore, we conclude that there is no evidence of a difference in Systolic blood pressure for alcohol "
          "consumption.")


## Are there any significant correlations between the continuous variables in the dataset?
## This step is important to ensure that collinearity doesn't exist in the model that will be built eventually
print("\n")
nhanes_corr = nhanes[["sys_blood_pres", "age", "weight", "height", "bmi", "leg_length", "arm_length", "arm_circum",
                      "waist_circum"]].corr()
print(nhanes_corr)
# sns.heatmap(nhanes_corr, annot=True, cmap='YlGn', linecolor='white', linewidth=1).set_title("Correlation matrix")

#### Determine the correlation between weight and BMI
print("\b")
corr, pvalue = spearmanr(nhanes["weight"], nhanes["bmi"], nan_policy='omit')
print("Spearman's correlation test between Weight and BMI")
print("Correlation coefficient:", str(corr))
print("pvalue:", pvalue)
print("\b")
print("Based on the above correlation coefficient matrix and heatmap as well as the spearman's coefficient test "
      "the following continuous predictors will be chosen for the study.")
print("1. Age\t2. Height\t3. BMI")
print("The outcome or predicted variable is: Systolic Blood Pressure (sys_blood_pres)")

# sns.scatterplot(x=nhanes["weight"], y=nhanes["bmi"], alpha=0.5)           # Plotting Weight & BMI




########################################## Model building phase ###################################################

##Replace gender values with 0 and 1
print("\n")
nhanes["gender"].replace({2: 0}, inplace=True)

### The idea is to use the BackWard Selection method for variable selection. The outcome variable is Systolic Blood
### Pressure. Predictors: ["age", "height", "bmi", "alc_drink_year", "smoke_100_cig", "gender", "race", "citizenship",
### "edu_lvl", "marital_status"]

model = ols("sys_blood_pres ~ age + height + bmi + alc_drink_year + smoke_100_cig + gender + race + citizenship"
            "+ edu_lvl + marital_status", data=nhanes).fit()
residuals = model.resid
print(model.summary())
print("\b")
print("p-values:")
print(model.pvalues, "\b", model.f_pvalue)
print("\b")
print("Residuals:")
print(residuals)
print("\b")
print("The following variables were considered statistically not significant at the 5% level and therefore, can be "
      "reasonably removed from the model:")
print("1. height (p: 0.844)\t2. citizenship (p: 0.764)\t3. marital_status (p: 0.586, 0.026, 0.122, 0.989, 0.746).")
print("Even though marital_status still has one treatment ('Married': 0.026) that is statistically significant, because "
      "majority of the treatments are not, it can be safely removed from the model to limit size.")


### New model with the aforementioned predictors removed:
print("\n")
model_2 = ols("sys_blood_pres ~ age + bmi + alc_drink_year + smoke_100_cig + gender + race + "
              "C(edu_lvl, Treatment(reference='<9th Grade'))", data=nhanes).fit()
residuals = model_2.resid
print(model_2.summary())
print("\b")
print("p-values:")
print(model_2.pvalues, "\b", model_2.f_pvalue)
print("\b")
print("Residuals:")
print(residuals)
print("\b")
print("The following variables were considered statistically not significant at the 5% level and therefore, can be "
      "reasonably removed from the model:")
print("1. smoke_100_cig (p: 0.588) 2. alc_drink_year (p: 0.085)")
print("\n")


### New model with the aforementioned predictors removed:
model_3 = ols("sys_blood_pres ~ age + bmi + gender + race +C(edu_lvl, Treatment(reference='<9th Grade'))",
              data=nhanes).fit()
residuals = model_3.resid
print(model_3.summary())
print("\b")
print("p-values:")
print(model_3.pvalues, "\b", model_3.f_pvalue)
print("\b")
print("Residuals:")
print(residuals)
print("\b")
print("Nearly all the predictors are statistically significant at the 5% level now and the model fits about 25% of the "
      "data. Further tests are needed to see how the model changes once a few more predictors are removed and "
      "interaction terms added.")
print("\b")


#### New model with education level removed and testing with new interaction terms to recheck for model fit
model_4 = ols("sys_blood_pres ~ age + bmi + gender + race + alc_drink_year + smoke_100_cig + (smoke_100_cig * bmi) + "
              "(alc_drink_year * age)", data=nhanes).fit()
residuals = model_4.resid
print(model_4.summary())
print("\b")
print("p-values:")
print(model_4.pvalues, "\b", model_4.f_pvalue)
print("\b")
print("Residuals:")
print(residuals)
print("\b")
print("Notes:")
print("Smoking was re-introduced back into the model and is statistically significant (p: 0.030) again along with a "
      "new interaction term: (smoke_100_cig * bmi) with pvalue - 0.033")
print("Alcohol consumption was re-introduced back into the model and is statistically significant (p: 0.009) again"
      " along with a new interaction term: (alc_drink_year * age) with pvalue - 0.008")
print("\b")

fig = sm.qqplot(residuals, line='s')                    #Plotting the Q-Q Plot
fig2 = sm.graphics.plot_regress_exog(model_4, "age")     #Plotting the residuals vs fitted values for age
fig2.tight_layout(pad=1.0)
fig3 = sm.graphics.plot_regress_exog(model_4, "bmi")     #Plotting the residuals vs fitted values for age
fig3.tight_layout(pad=1.0)
# sns.histplot(residuals, kde=True)
print("Normality conditions are met!")
print("\n")
print("**************************************** Final Model *****************************************")
print("\b")
print("sys_blood_pres = 92.6825 - 5.9105*race.Caucasian - 4.1073*race.MexicanAmerican - "
      "2.9425*race.OtherHispanic - 4.3719*race.OtherRace + 0.5057*age + 0.3292*bmi + 3.7745*gender + "
      "3.5165*alc_drink_year + 4.4329*smoke_100_cig - 0.1419*(smoke_100_cig*bmi) - 0.0677*(alc_drink_year*age)")

plt.tight_layout()
plt.show()
