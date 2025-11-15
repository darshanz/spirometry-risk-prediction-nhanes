import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = Path.cwd()
work_dir = current_dir.parent.parent
data_dir = f'{work_dir}/data'


def load_nhanes_as_df():
    file_name = "NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"
    df = pd.read_csv(f"{data_dir}/{file_name}")
    return df


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