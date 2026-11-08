# EDA report - UCI Statlog (Vehicle Silhouettes)


## Shape

846 rows x 19 columns (18 features + 1 target)


## First five rows

|   compactness |   circularity |   distance_circularity |   radius_ratio |   pr_axis_aspect_ratio |   max_length_aspect_ratio |   scatter_ratio |   elongatedness |   pr_axis_rectangularity |   max_length_rectangularity |   scaled_variance_major |   scaled_variance_minor |   scaled_radius_of_gyration |   skewness_major |   skewness_minor |   kurtosis_minor |   kurtosis_major |   hollows_ratio | vehicle_class   |
|--------------:|--------------:|-----------------------:|---------------:|-----------------------:|--------------------------:|----------------:|----------------:|-------------------------:|----------------------------:|------------------------:|------------------------:|----------------------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|:----------------|
|            95 |            43 |                     96 |            202 |                     65 |                        10 |             189 |              35 |                       22 |                         143 |                     217 |                     534 |                         166 |               71 |                6 |               27 |              190 |             197 | opel            |
|            96 |            52 |                    104 |            222 |                     67 |                         9 |             198 |              33 |                       23 |                         163 |                     217 |                     589 |                         226 |               67 |               12 |               20 |              192 |             201 | opel            |
|           107 |            52 |                    101 |            218 |                     64 |                        11 |             202 |              33 |                       23 |                         164 |                     219 |                     610 |                         192 |               65 |               17 |                2 |              197 |             206 | opel            |
|            97 |            37 |                     78 |            181 |                     62 |                         8 |             161 |              41 |                       20 |                         131 |                     182 |                     389 |                         117 |               62 |                2 |               28 |              203 |             211 | opel            |
|            96 |            54 |                    104 |            175 |                     58 |                        10 |             215 |              31 |                       24 |                         175 |                     221 |                     682 |                         222 |               75 |               13 |               23 |              186 |             194 | opel            |


## Last five rows

|   compactness |   circularity |   distance_circularity |   radius_ratio |   pr_axis_aspect_ratio |   max_length_aspect_ratio |   scatter_ratio |   elongatedness |   pr_axis_rectangularity |   max_length_rectangularity |   scaled_variance_major |   scaled_variance_minor |   scaled_radius_of_gyration |   skewness_major |   skewness_minor |   kurtosis_minor |   kurtosis_major |   hollows_ratio | vehicle_class   |
|--------------:|--------------:|-----------------------:|---------------:|-----------------------:|--------------------------:|----------------:|----------------:|-------------------------:|----------------------------:|------------------------:|------------------------:|----------------------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|:----------------|
|            88 |            40 |                     55 |            114 |                     53 |                         7 |             132 |              53 |                       18 |                         139 |                     142 |                     249 |                         158 |               87 |                0 |                7 |              176 |             183 | van             |
|            86 |            39 |                     62 |            129 |                     59 |                         6 |             116 |              57 |                       17 |                         135 |                     137 |                     203 |                         145 |               64 |                7 |                9 |              199 |             204 | van             |
|            94 |            47 |                     85 |            333 |                    138 |                        49 |             155 |              43 |                       19 |                         155 |                     320 |                     354 |                         187 |              135 |               12 |                9 |              188 |             196 | van             |
|            86 |            40 |                     66 |            139 |                     59 |                         7 |             122 |              54 |                       17 |                         139 |                     145 |                     225 |                         143 |               63 |                7 |               11 |              202 |             208 | van             |
|            92 |            38 |                     60 |            130 |                     62 |                         5 |             114 |              58 |                       17 |                         132 |                     135 |                     194 |                         137 |               72 |               14 |                5 |              190 |             194 | van             |


## Data types

|       |   count |
|:------|--------:|
| int64 |      18 |
| str   |       1 |


## Missing values and duplicates

Missing cells: 0

Duplicate rows: 0

Rows with identical features: 0


## Descriptive statistics

|                           |   count |   mean |    std |   min |    25% |   50% |   75% |   max |   skew |
|:--------------------------|--------:|-------:|-------:|------:|-------:|------:|------:|------:|-------:|
| compactness               |     846 |  93.68 |   8.23 |    73 |  87    |  93   |   100 |   119 |   0.38 |
| circularity               |     846 |  44.86 |   6.17 |    33 |  40    |  44   |    49 |    59 |   0.26 |
| distance_circularity      |     846 |  82.09 |  15.77 |    40 |  70    |  80   |    98 |   112 |   0.11 |
| radius_ratio              |     846 | 168.94 |  33.47 |   104 | 141    | 167   |   195 |   333 |   0.39 |
| pr_axis_aspect_ratio      |     846 |  61.69 |   7.89 |    47 |  57    |  61   |    65 |   138 |   3.82 |
| max_length_aspect_ratio   |     846 |   8.57 |   4.6  |     2 |   7    |   8   |    10 |    55 |   6.78 |
| scatter_ratio             |     846 | 168.84 |  33.24 |   112 | 146.25 | 157   |   198 |   265 |   0.61 |
| elongatedness             |     846 |  40.93 |   7.81 |    26 |  33    |  43   |    46 |    61 |   0.05 |
| pr_axis_rectangularity    |     846 |  20.58 |   2.59 |    17 |  19    |  20   |    23 |    29 |   0.77 |
| max_length_rectangularity |     846 | 148    |  14.52 |   118 | 137    | 146   |   159 |   188 |   0.26 |
| scaled_variance_major     |     846 | 188.63 |  31.39 |   130 | 167    | 178.5 |   217 |   320 |   0.65 |
| scaled_variance_minor     |     846 | 439.91 | 176.69 |   184 | 318.25 | 364   |   587 |  1018 |   0.84 |
| scaled_radius_of_gyration |     846 | 174.7  |  32.55 |   109 | 149    | 173   |   198 |   268 |   0.28 |
| skewness_major            |     846 |  72.46 |   7.49 |    59 |  67    |  71.5 |    75 |   135 |   2.07 |
| skewness_minor            |     846 |   6.38 |   4.92 |     0 |   2    |   6   |     9 |    22 |   0.77 |
| kurtosis_minor            |     846 |  12.6  |   8.93 |     0 |   5    |  11   |    19 |    41 |   0.69 |
| kurtosis_major            |     846 | 188.93 |   6.16 |   176 | 184    | 188   |   193 |   206 |   0.25 |
| hollows_ratio             |     846 | 195.63 |   7.44 |   181 | 190.25 | 197   |   201 |   211 |  -0.23 |


## Target distribution

| vehicle_class   |   count |   share_% |
|:----------------|--------:|----------:|
| bus             |     218 |      25.8 |
| opel            |     212 |      25.1 |
| saab            |     217 |      25.7 |
| van             |     199 |      23.5 |


## Strongly correlated pairs (|r| >= 0.90)

14 pairs

|                                                     |      r |
|:----------------------------------------------------|-------:|
| ('scatter_ratio', 'scaled_variance_minor')          |  0.996 |
| ('pr_axis_rectangularity', 'scaled_variance_minor') |  0.992 |
| ('scatter_ratio', 'pr_axis_rectangularity')         |  0.992 |
| ('scatter_ratio', 'elongatedness')                  | -0.973 |
| ('circularity', 'max_length_rectangularity')        |  0.966 |
| ('elongatedness', 'scaled_variance_minor')          | -0.957 |
| ('scatter_ratio', 'scaled_variance_major')          |  0.952 |
| ('elongatedness', 'pr_axis_rectangularity')         | -0.951 |
| ('scaled_variance_major', 'scaled_variance_minor')  |  0.95  |
| ('elongatedness', 'scaled_variance_major')          | -0.938 |
| ('pr_axis_rectangularity', 'scaled_variance_major') |  0.938 |
| ('circularity', 'scaled_radius_of_gyration')        |  0.936 |
| ('distance_circularity', 'elongatedness')           | -0.912 |
| ('distance_circularity', 'scatter_ratio')           |  0.907 |


## Outliers per feature (1.5 x IQR rule)

|                         |   n_outliers |
|:------------------------|-------------:|
| skewness_major          |           15 |
| max_length_aspect_ratio |           13 |
| skewness_minor          |           12 |
| pr_axis_aspect_ratio    |            8 |
| radius_ratio            |            3 |
| scaled_variance_minor   |            2 |
| scaled_variance_major   |            1 |
| kurtosis_minor          |            1 |


## One-way ANOVA F-test of each feature vs class

|                           |   F_statistic |   p_value |
|:--------------------------|--------------:|----------:|
| elongatedness             |         98.3  |  1.41e-54 |
| scatter_ratio             |         82.3  |  1.09e-46 |
| scaled_variance_minor     |         76.4  |  1.08e-43 |
| pr_axis_rectangularity    |         73.4  |  3.58e-42 |
| distance_circularity      |         68.7  |  9.33e-40 |
| scaled_variance_major     |         67.5  |  4e-39    |
| radius_ratio              |         54.7  |  2.48e-32 |
| skewness_major            |         52.7  |  3.18e-31 |
| hollows_ratio             |         39.9  |  4.09e-24 |
| compactness               |         33.5  |  1.95e-20 |
| scaled_radius_of_gyration |         27.3  |  7.65e-17 |
| kurtosis_minor            |         26.1  |  3.74e-16 |
| circularity               |         21.5  |  2.14e-13 |
| max_length_aspect_ratio   |         13.4  |  1.45e-08 |
| skewness_minor            |         12.6  |  4.42e-08 |
| max_length_rectangularity |          7    |  0.000119 |
| pr_axis_aspect_ratio      |          4.9  |  0.00221  |
| kurtosis_major            |          3.83 |  0.00972  |


## Opel vs Saab separability (ANOVA F on the two car classes only)

|                           |   F_statistic |
|:--------------------------|--------------:|
| compactness               |           7   |
| skewness_minor            |           3.9 |
| max_length_rectangularity |           2.4 |
| circularity               |           2.4 |
| hollows_ratio             |           2.4 |
| scaled_variance_minor     |           0.8 |
| kurtosis_major            |           0.7 |
| scatter_ratio             |           0.6 |
| skewness_major            |           0.6 |
| elongatedness             |           0.4 |
| pr_axis_aspect_ratio      |           0.4 |
| pr_axis_rectangularity    |           0.3 |
| scaled_variance_major     |           0.3 |
| max_length_aspect_ratio   |           0.1 |
| kurtosis_minor            |           0.1 |
| distance_circularity      |           0.1 |
| radius_ratio              |           0   |
| scaled_radius_of_gyration |           0   |
