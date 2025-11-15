import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pwlf
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 



current_dir = Path.cwd()
work_dir = current_dir.parent.parent
data_dir = f'{work_dir}/data'


def load_nhanes_as_df():
    file_name = "NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"
    df = pd.read_csv(f"{data_dir}/{file_name}")
    return df

def load_preprocessed_df():
    file_name = "final_df.csv"
    df = pd.read_csv(f"{data_dir}/{file_name}")
    return df

def save_df_to_csv(df, filename):
    output_path = os.path.join(data_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved as {filename} in data directory.")


def plot_sample_counts(column, df, ax=None):
    counts = df[column].value_counts()
    data = pd.DataFrame({
        'Value': counts.index,
        'Count': counts.values
    })
    
    sns.barplot(x='Value', y='Count', data=data, errorbar=None, ax=ax)
    if ax:  # Add labels and title if ax is provided
        for container in ax.containers:
            ax.bar_label(container)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f'{column} Value Counts')
    else:
        plt.title(f'{column} Value Counts')
        plt.show()


def box_plots(df, categorical_variable, continuous_variables):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, var in enumerate(continuous_variables):
        sns.boxplot(data=df, x=categorical_variable, y=continuous_variables[i], ax=axes[i]) 
        axes[i].set_title(f'{var} Distribution by {categorical_variable}')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def scatterplot_with_targets(df, variable_, targets):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
 

    for ax, (col, label) in zip(axes, targets):
        sns.scatterplot(
            data=df,
            x=variable_,
            y=col,
            alpha=0.3,
            s=20,
            ax=ax
        )

        sns.regplot(
            data=df,
            x=variable_,
            y=col,
            scatter=False,
            lowess=True,
            ax=ax,
            color='red'
        )

        ax.set_title(f'{variable_} vs {label}', fontsize=14)
        ax.set_ylabel(label)

    axes[-1].set_xlabel(variable_)
    plt.tight_layout()
    plt.show()


def box_plot_two_by_three(df, variables, targets):
    # Two variables and 3 targets expected
    if len(variables) != 2 or len(targets) != 3:
        print("Error: 'variables' must contain 2 elements and 'targets' must contain 3 elements.")
        return
 
    sns.set(style="whitegrid", font_scale=0.8)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
    fig.suptitle('Distribution of Target Variables Across Key Categories', fontsize=18, y=1.02)
 
    for i, x_var in enumerate(variables): 
        for j, (y_col, y_label) in enumerate(targets):
            ax = axes[i, j]  
            sns.boxplot(
                data=df,
                x=x_var,
                y=y_col,
                ax=ax,
                palette='coolwarm',
                linewidth=1.2,
                fliersize=3
            ) 
            ax.set_title(f'{y_label} by {x_var}', fontsize=14)
            ax.set_ylabel(y_label, fontsize=12) 
            ax.set_xlabel('')

            if i == 1:
                ax.tick_params(axis='x', rotation=45)
                ax.set_xlabel(x_var, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()



def correlation_heatmap(df, corr_cols, title):
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=.5
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def select_optimal_segments(df, x_col, y_col, max_segments=6):
    """
    Fits piecewise linear models with 1 to max_segments, calculates BIC for each,
    and identifies the optimal segment count (lowest BIC).

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - x_col (str): The name of the predictor column (e.g., 'Age').
    - y_col (str): The name of the target column (e.g., 'FEV1').
    - max_segments (int): The maximum number of segments to test.
    """
    X = df[x_col].values
    Y = df[y_col].values
    n_samples = len(X)

    # results: {num_segments: BIC_value}
    bic_results = {}

    print(f"--- Testing Models from 1 to {max_segments} Segments ---")
    for n_segments in range(1, max_segments + 1):
        try:
            my_pwlf = pwlf.PiecewiseLinFit(X, Y)
            my_pwlf.fit(n_segments)
            Y_pred = my_pwlf.predict(X)
            rss = np.sum((Y - Y_pred)**2)
            k_parameters = 2 * n_segments
            bic_value = n_samples * np.log(rss / n_samples) + k_parameters * np.log(n_samples)
            bic_results[n_segments] = bic_value
            print(f"Segments: {n_segments} | BIC: {bic_value:.4f}") 
        except Exception as e: 
            print(f"Skipping {n_segments} segments due to error: {e}")
            break
 
    if not bic_results:
        print("No models were successfully fitted.")
        return

    optimal_segments = min(bic_results, key=bic_results.get)
    print(f"\n optimal segments (min BIC):: {optimal_segments}")

    # 4. Visualization  
    segments = list(bic_results.keys())
    bics = list(bic_results.values())

    plt.figure(figsize=(9, 6))
    plt.plot(segments, bics, marker='o', linestyle='-', color='indigo', linewidth=2)
    plt.scatter(optimal_segments, bic_results[optimal_segments], 
                color='red', s=150, zorder=5, 
                label=f'Optimal: {optimal_segments} Segments')
    
    plt.title('BIC for Model Selection', fontsize=16)
    plt.xlabel('Number of Segments:', fontsize=12)
    plt.ylabel('BIC Value', fontsize=12)
    plt.xticks(segments)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()
    
    return optimal_segments




def visualize_and_interpret_model(df, x_col, y_col, best_k): 
    X = df[x_col].values
    Y = df[y_col].values 
    final_pwlf = pwlf.PiecewiseLinFit(X, Y)
    final_breakpoints = final_pwlf.fit(best_k)
    
    #plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label='Original Data', alpha=0.5, s=20, color='#1f77b4')

    X_predict = np.linspace(X.min(), X.max(), 500)
    Y_predict = final_pwlf.predict(X_predict)
    plt.plot(X_predict, Y_predict, color='red', linewidth=3, label=f'Optimal {best_k}-Segment Fit')
    knots = final_breakpoints[1:-1]
    
    print(f"Model: ({best_k} Segments)")
    print(f"R-squared: {final_pwlf.r_squared():.4f}")
    
    # Calculate slopes for each segment 
    print(f"Breakpoints (Ages): {np.round(knots, 2)}")
    
    segment_ranges = [f"{X.min():.0f} to {knots[0]:.2f}"]
    for i in range(len(knots) - 1):
        segment_ranges.append(f"{knots[i]:.2f} to {knots[i+1]:.2f}")
    segment_ranges.append(f"{knots[-1]:.2f} to {X.max():.0f}")
 
    slopes = []
    for i in range(best_k):
        # Slope:  
        # (Y_pred_end - Y_pred_start) / (X_end - X_start)
        if i == 0:
            x_start = X.min()
            x_end = knots[0]
        elif i == best_k - 1:
            x_start = knots[-1]
            x_end = X.max()
        else:
            x_start = knots[i-1]
            x_end = knots[i]
        
        # jsut check validity
        if x_end > x_start:
            y_start = final_pwlf.predict(np.array([x_start]))[0]
            y_end = final_pwlf.predict(np.array([x_end]))[0]
            slope = (y_end - y_start) / (x_end - x_start)
            slopes.append(slope)
        else:
            slopes.append(np.nan)
 
    for knot_value in knots:
        plt.axvline(x=knot_value, color='green', linestyle='--', alpha=0.7)

    plt.title(f'Optimal Segmented Fit (R-sq: {final_pwlf.r_squared():.4f})', fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()
    return knots

 

def lasso_feature_selection(df, target_col):
    """
    Performs a mock LASSO regression to select important non-age predictors.
    In a real scenario, you'd tune the alpha parameter (penalty).
    """
    # Exclude Age  and target 
    feature_cols = [c for c in df.columns if c not in ['Age', target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Standardize data for LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use a moderate alpha (penalty)
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Select features wiht non-zero coefficients 
    selected_features = [feature_cols[i] for i, coef in enumerate(lasso.coef_) if abs(coef) > 1e-4]
    print(f"Selected features(except age): {selected_features}") 
    return selected_features

def create_segmented_basis_features(df, knots, age_col='Age'):
    """
    Creates the segmented basis functions (Hinge Functions) for the Age variable.
    The formula is: max(0, Age - Knot_i)
    """
    df_segmented = df.copy()
    for i, knot in enumerate(knots):
        new_col_name = f'{age_col}_Knot_{i+1}_{int(knot)}'
        df_segmented[new_col_name] = np.maximum(0, df_segmented[age_col] - knot)
        
    print(f"Created {len(knots)} segmented basis features for Age.")
    return df_segmented




def perform_model_diagnostics(model, X_data, final_predictors):
    """
    Performs the final model validation checks: Multicollinearity (VIF) and Residual Plots.
    """
    print("\n--- Model Validation and Diagnostics ---")
    
    # 1. Multicollinearity Check (Variance Inflation Factor - VIF)
     
    X_predictors = X_data[final_predictors]
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_predictors.columns
    vif_data["VIF"] = [variance_inflation_factor(X_predictors.values, i) 
                       for i in range(X_predictors.shape[1])]
    
    # Sort by VIF descending
    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    
    print("\n[1] Multicollinearity (VIF Score) Check:")
    print("Rule of Thumb: VIF > 5 or 10 suggests problematic multicollinearity.")
    print(vif_data)
    
    # 2. Residual Plots (Normality and Homoscedasticity)
    residuals = model.resid
    predictions = model.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Diagnostic Plots', fontsize=16)

    # Plot A: Homoscedasticity Check (Residuals vs. Fitted Values)
    axes[0].scatter(predictions, residuals, alpha=0.5)
    axes[0].hlines(0, xmin=predictions.min(), xmax=predictions.max(), color='red', linestyle='--')
    axes[0].set_title('Residuals vs. Fitted Values (Homoscedasticity)')
    axes[0].set_xlabel('Fitted Values (Predicted FEV1)')
    axes[0].set_ylabel('Residuals (Error)')
    axes[0].grid(True, linestyle=':')

    # Plot B: Normality Check (Histogram of Residuals)
    axes[1].hist(residuals, bins=50, edgecolor='k', alpha=0.7)
    axes[1].set_title('Histogram of Residuals (Normality)')
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, linestyle=':')

    # Plot C: Normality Check (Q-Q Plot)
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Normal Q-Q Plot of Residuals')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()