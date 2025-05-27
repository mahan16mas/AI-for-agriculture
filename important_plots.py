import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def plot_features(data, *features, bins=30, title=''):
    n = len(features)
        
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
    
    
    axes = axes.flatten()
    
    colors = sns.color_palette("Set1", n)
    
    for i, (feature, color) in enumerate(zip(features, colors)):
        axes[i].hist(data[feature], bins=bins, color=color, density=True)
        axes[i].set_title(f'{feature} Distribution'.strip())
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Distribution')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=20, y=1.02)

    
    plt.tight_layout()
    plt.show()


def box_plot(data, *features):
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols  
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()
    
    palette = sns.color_palette("Set1", n_colors=n)
    
    for i, feature in enumerate(features):
        sns.boxplot(y=data[feature], ax=axes[i], color=palette[i])
        axes[i].set_title(f'Box Plot of {feature}')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def correlation(data, *features):
    corr_matrix = data[list(features)].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix')
    plt.show()


def pie_plot(data, name):
    name_counts = data[name].value_counts()

    name_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, figsize=(6,6))

    plt.ylabel('')
    plt.title(f'Distribution of {name}')
    plt.show()