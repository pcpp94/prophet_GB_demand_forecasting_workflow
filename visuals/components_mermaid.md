
```mermaid
---
config:
  sankey:
    showValues: false
---
sankey-beta

Daily Model Trend, Monthly Model Trend, 1
Daily Model Yearly Seasonalities, Monthly Model Yearly reg., 1
Daily Model Yearly Seasonalities, Monthly Model Leap Year reg., 1
Daily Model Weekly Seasonalities, Monthly Model Weekdays reg., 1
Daily Model Temperature reg., Model Adjustment, 1
Daily Model HDD reg., Model Adjustment, 1
Daily Model CDD reg., Model Adjustment, 1
Daily Model Saturday Daylight reg., Monthly Model Weekdays reg., 1
Daily Model Sunday Daylight reg., Monthly Model Weekdays reg., 1
Daily Model Holidays (ex. Lockdowns | Beast of the East), Model Adjustment, 1
Daily Model Lockdowns, Monthly Model GDP reg., 1
Daily Model Trend, Monthly Model GDP reg., 1
Daily Model Beast of the East, Model Adjustment, 1


```
\
**The monthly model is going to have a different trend as we are:**
1. Removing the effect of weather and holidays (**external independent variables**) so we can forecast on data that is "unchanged" due to weather and holidays.
2. Adding a new variable on top of the "residual Daily Model variables" (All but weather + holidays), which is GDP.
3. GDP accounts for what was measured by "Lockdowns" in the Daily Model + signals that are not caught by the Daily Model.
  
So, what we are taking out from the Daily Model are all the variables that affect the demand that are not being caught by the Monthly Model.

We could also take out all the Daily Model variables related to the Monthly Model "Weekdays" regressor, and also this Monthly Model regressor. After that, we'd need to add these weights back to the outputs of the new Monthly Model.
