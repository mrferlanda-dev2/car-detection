#!/usr/bin/env python3
"""
Dataset Distribution Visualization Script
Visualizes the distribution of images across different vehicle classes in the classification dataset.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from pathlib import Path

def count_images_in_directory(directory_path):
    """
    Count the number of image files in a directory and its subdirectories.
    
    Args:
        directory_path (str): Path to the directory to count images in
        
    Returns:
        int: Number of image files found
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    count = 0
    
    if not os.path.exists(directory_path):
        return 0
    
    try:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    count += 1
    except Exception as e:
        print(f"Error counting images in {directory_path}: {e}")
        return 0
    
    return count

def visualize_dataset_distribution(dataset_path="dataset/train"):
    """
    Create visualizations for the dataset distribution.
    
    Args:
        dataset_path (str): Path to the classification dataset directory
    """
    
    # Get all class directories
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist!")
        return
    
    class_counts = {}
    
    # Count images in each class directory
    for item in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, item)
        if os.path.isdir(class_path):
            count = count_images_in_directory(class_path)
            class_counts[item] = count
            print(f"Class '{item}': {count} images")
    
    if not class_counts:
        print("No class directories found in the dataset!")
        return
    
    # Sort classes by count for better visualization
    sorted_classes = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Calculate statistics
    total_images = sum(class_counts.values())
    avg_images = total_images / len(class_counts) if class_counts else 0
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total classes: {len(class_counts)}")
    print(f"Total images: {total_images}")
    print(f"Average images per class: {avg_images:.1f}")
    print(f"Min images in a class: {min(class_counts.values()) if class_counts else 0}")
    print(f"Max images in a class: {max(class_counts.values()) if class_counts else 0}")
    
    # Create visualizations
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Vehicle Classification Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of class distribution
    classes = list(sorted_classes.keys())
    counts = list(sorted_classes.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    bars = ax1.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Number of Images per Vehicle Class', fontweight='bold')
    ax1.set_xlabel('Vehicle Class')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart of class distribution
    if total_images > 0:
        wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Percentage Distribution of Vehicle Classes', fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'No images found', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Percentage Distribution of Vehicle Classes', fontweight='bold')
    
    # 3. Horizontal bar chart for better readability
    y_pos = np.arange(len(classes))
    bars_h = ax3.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(classes)
    ax3.set_xlabel('Number of Images')
    ax3.set_title('Images per Class (Horizontal View)', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels on horizontal bars
    for i, (bar, count) in enumerate(zip(bars_h, counts)):
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{count}', ha='left', va='center', fontweight='bold')
    
    # 4. Statistics summary
    ax4.axis('off')
    stats_text = f"""
    Dataset Statistics Summary
    
    Total Classes: {len(class_counts)}
    Total Images: {total_images}
    Average per Class: {avg_images:.1f}
    
    Class Breakdown:
    """
    
    for class_name, count in sorted_classes.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        stats_text += f"  • {class_name}: {count} images ({percentage:.1f}%)\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'dataset_distribution_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {filename}")
    
    # Use a non-interactive backend to avoid display issues
    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not display plot interactively: {e}")
        print("However, the plot has been saved to file successfully.")
    
    return class_counts

if __name__ == "__main__":
    # Add pandas import for timestamp
    import pandas as pd
    
    print("=== Vehicle Classification Dataset Distribution Analysis ===")
    print("Analyzing image distribution across vehicle classes...\n")
    
    # Visualize the dataset distribution
    class_counts = visualize_dataset_distribution()
    
    if class_counts:
        print("\n=== Analysis Complete ===")
        print("Check the generated visualization plot for detailed insights!")
    else:
        print("\n=== No data found ===")
        print("Make sure your classification_dataset directory contains class subdirectories with images.") 