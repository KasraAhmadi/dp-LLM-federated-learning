import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_multiple(title,file_names):
    plt.figure(figsize=(10, 6))

    colors = ['b', 'g', 'orange']  # Define colors for each plot
    for idx, file_name in enumerate(file_names):
        # Read the CSV file
        data = pd.read_csv(file_name)

        # Extract the "Round" and "Accuracy" columns
        rounds = data["Round"]
        accuracy = data["Accuracy"]

        # Find the maximum accuracy and its corresponding round
        max_accuracy = accuracy.max()
        max_accuracy_round = rounds[accuracy.idxmax()]

        # Plot the chart
        label = file_name.split("/")[-1].split(".")[0]
        plt.plot(rounds, accuracy, marker='o', linestyle='-', color=colors[idx % len(colors)], label=f'{label}')

        # Mark the maximum accuracy
        plt.scatter(max_accuracy_round, max_accuracy, color=colors[idx % len(colors)], s=100, zorder=5)
        plt.annotate(
            f'({max_accuracy_round}, {max_accuracy:.2f})',
            (max_accuracy_round, max_accuracy),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            color=colors[idx % len(colors)]
        )

    # Add title, labels, and legend
    plt.ylim(0.3, 1)
    plt.title(title, fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot to a file
    output_file = "./figs/"+title+".png"  # Change to your desired file name and format
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")



title = "SST2_Iid"
folder_path = "./test_result"  # Replace with the path to your folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
plot_multiple(title,csv_files)