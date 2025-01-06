# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# Our_Theorem = [1.88,1.71,1.63,1.59]
# RDP = [0.96,0.86,0.82,0.8]
# x_labels = ['1', '2', '3', '4']

# # Plot configuration
# x = np.arange(len(x_labels))
# width = 0.35

# fig, ax = plt.subplots(figsize=(8, 6))
# bars1 = ax.bar(x - width/2, Our_Theorem, width, label='Our_Theorem', color='skyblue')
# bars2 = ax.bar(x + width/2, RDP, width, label='RDP', color='orange')

# # Adding details
# ax.set_xlabel('Partition')
# ax.set_ylabel('Values')
# ax.set_title('QQP Linear Noise Value')
import matplotlib.pyplot as plt
import numpy as np

# Data
def plot(title,Our_Theorem,RDP  ):
    x_labels = ['1', '2', '3', '4']
    dataset_sizes = ['36384', '72769', '109153', '145540']  # Example dataset sizes

    # Create new labels with dataset size
    x_labels_with_sizes = [f"{label}\n(Size: {size})" for label, size in zip(x_labels, dataset_sizes)]

    # Plot configuration
    x = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, Our_Theorem, width, label='Our_Theorem', color='skyblue')
    bars2 = ax.bar(x + width/2, RDP, width, label='RDP', color='orange')

    # Adding details
    ax.set_xlabel('Partition')
    ax.set_ylabel('Sigma')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels_with_sizes)
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.show()
        # Save the plot to a file
    output_file = "./figs/"+title+".png"  # Change to your desired file name and format
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

Our_Theorem = [1.88,1.71,1.63,1.59]
RDP = [0.96,0.86,0.82,0.8]
plot("QQP_Linear_Noise_Level",Our_Theorem,RDP)