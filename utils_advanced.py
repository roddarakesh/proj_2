# General dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Models to use in our pipeline
from sklearn.linear_model import LinearRegression  # OLS
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso  # regularized linear regression
from sklearn.linear_model import Ridge  # regularized linear regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Preprocessing dependencies
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

random_seed = 42


def process_data(df, features, target):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest
    Features : List of columns of Independent variables(predictors)
    Target : List of Dependent variable(s)

    Returns: cleaned encoded dataframe with no null values
    """

    df = df.dropna().reset_index(drop=True)
    # features
    X = df.drop(target, axis=1)
    # target
    y = df[target]

    # Identify categorical columns and encode them
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Apply one-hot encoding to categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns)

    return train_test_split(X_encoded, y, random_state=random_seed)


def plot_numeric_distributions(df):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest

    Returns : histogram distribution, box plot and probplot of numerical columns in the dataframe
    """

    df = df.fillna(0)
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    plot_cols = [col for col in numeric_cols if df[col].nunique() > 10]
    n_rows = len(plot_cols)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))

    print("DISTRIBUTION OF DATA IN NUMERICAL COLUMNS")
    for i, col in enumerate(plot_cols):
        # Histogram distribution
        sns.histplot(df[col], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'Distribution of {col}', fontsize=15)

        # Box plot to detect outliers
        sns.boxplot(x=df[col], ax=axes[i, 1])
        axes[i, 1].set_title(f'Box Plot of {col}', fontsize=15)

        # Q-Q plot
        stats.probplot(df[col], dist="norm", plot=axes[i, 2])
        axes[i, 2].set_title(f'Q-Q Plot of {col}', fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df):
    # Identify categorical columns and encode them
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    plot_cols = [col for col in cat_cols]
    n_cols = len(plot_cols)

    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 5))

    for i, col in enumerate(plot_cols):
        sns.countplot(x=col, data=df, ax=axes[i] if n_cols > 1 else axes)
        axes[i].set_title(f'Distribution of {col}', fontsize=15)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """
    Args:
    DataFrame : A pandas dataframe object containing data of interest

    Returns : correlation heatmap of numerical columns in the dataframe
    """

    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of numerical columns')
    plt.show()


def r2_adj(X, y, model):
    """
    Calculates adjusted r-squared values

    Args:
    X: Independent variables, the data to fit
    y: dependent variable, the target data to try to predict
    model: The estimator or object to use to train the data

    Returns: adjusted r sqaured value accountign for number of predictors
    """
    r2 = model.score(X, y)
    n = X.shape[0]
    p = y.ndim

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def model_generator_imputed(df, features, target):
    X = df[features]
    y = df[target]

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    wip_imp = imp.fit_transform(X[['wip']])
    X['wip_imp'] = wip_imp
    X_imp = X.drop("wip", axis=1)

    X_imp = pd.get_dummies(X_imp)
    X_imp_train, X_imp_test, y_train, y_test = train_test_split(X_imp, y, random_state=random_seed)

    models = {
        "LR": LinearRegression(),
        "SVR": SVR(),
        "RFC": RandomForestRegressor(),
        "HGBC": GradientBoostingRegressor(),
        "DTC": DecisionTreeRegressor()}

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("Scale", StandardScaler(with_mean=False)),
            (name, model)
        ])

        pipeline.fit(X_imp_train, y_train)
        y_pred = pipeline.predict(X_imp_test)
        mse = mean_squared_error(y_test, y_pred)

        r2_value = r2_score(y_test, y_pred)
        r2_adj_value = r2_adj(X_imp_test, y_test, pipeline)

        results["R squared"] = r2_value
        results["Adjusted R squared"] = r2_adj_value
        results["Mean Squared error"] = mse
        print(f"{model} || R squared: {r2_value:.3f} || Adjusted R squared:{r2_adj_value:.3f}")
        print("===================================================================")


def model_generator(df, features, target):
    X_train, X_test, y_train, y_test = process_data(df, features, target)

    models = {
        "LR": LinearRegression(),
        "SVR": SVR(),
        "RFC": RandomForestRegressor(),
        "HGBC": GradientBoostingRegressor(),
        "DT": DecisionTreeRegressor()
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("Scale", StandardScaler(with_mean=False)),
            (name, model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        r2_value = r2_score(y_test, y_pred)
        r2_adj_value = r2_adj(X_test, y_test, pipeline)

        results["R squared"] = r2_value
        results["Adjusted R squared"] = r2_adj_value
        results["Mean Squared error"] = mse
        print(f"{model}|| R squared: {r2_value:.3f} || Adjusted R squared:{r2_adj_value:.3f}")
        print("===================================================================")


def model_optimization(df, features, target):
    X_train, X_test, y_train, y_test = process_data(df, features, target)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        'LR_LASSO': Lasso(),
        'LR_RIDGE': Ridge(),
        "SVR": SVR(),
        "RFC": RandomForestRegressor(),
        "HGBC": GradientBoostingRegressor()
    }

    pipes = {}

    for acronym, model in models.items():
        pipes[acronym] = Pipeline([('model', model)])

    param_grids = {}

    # Parameter grid for LASSO regression
    # The parameter grid of alpha
    alpha_grid = [0.01, 0.1, 0.5]
    max_iter = [1100]
    # Update param_grids
    param_grids['LR_LASSO'] = {'model__alpha': alpha_grid,
                               'model__max_iter': max_iter}

    # Parameter grid for RIDGE regression
    # The parameter grid of alpha
    alpha_grid = [0.01, 0.1, 0.5]
    max_iter = [1100]
    # Update param_grids
    param_grids['LR_RIDGE'] = {'model__alpha': alpha_grid,
                               'model__max_iter': max_iter}

    # Parameter grid for SVR
    estimator__C = [1, 10, 100, 1000]
    param_grids['SVR'] = {
        'model__C': estimator__C,
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto', 0.001, 0.0001]}

    # Parameter grid for RANDOM FOREST.
    # The grids for min_samples_split
    min_samples_split_grids = [2, 20, 200]
    # The grids for min_samples_leaf
    min_samples_leaf_grids = [1, 20, 200]
    # Update param_grids
    param_grids['RFC'] = {'model__min_samples_split': min_samples_split_grids,
                          'model__min_samples_leaf': min_samples_leaf_grids}

    # Parameter grid for GRADIENT BOOSTING The grids for learning_rate
    learning_rate_grids = [10 ** i for i in range(-4, 2)]

    # The grids for min_samples_leaf
    min_samples_leaf_grids = [1, 20, 100]

    # Update param_grids
    param_grids['HGBC'] = {'model__learning_rate': learning_rate_grids,
                           'model__min_samples_leaf': min_samples_leaf_grids}

    # The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
    best_score_params_estimator_gs = []

    # For each model
    for acronym in pipes.keys():
        print("===============================")
        print(f"Optimizing {acronym} .........")
        # GridSearchCV
        gs = GridSearchCV(estimator=pipes[acronym],
                          param_grid=param_grids[acronym],
                          scoring='r2',
                          n_jobs=-1,
                          cv=5,
                          return_train_score=True,
                          verbose=1)

        # Fit the pipeline
        gs = gs.fit(X_train, y_train)

        # Update best_score_params_estimator_gs
        best_score_params_estimator_gs.append([acronym, gs.best_score_, gs.best_params_, gs.best_estimator_])

        print(f"{acronym} best R² score: {gs.best_score_:.4f}")

    # Sort by best score
    best_score_params_estimator_gs.sort(key=lambda x: x[0], reverse=True)

    results_df = pd.DataFrame(
        best_score_params_estimator_gs,
        columns=['model name', 'best_score', 'best_params', 'best_estimator']
    )

    return results_df.sort_values(by='best_score', ascending=False)
