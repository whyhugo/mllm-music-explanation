import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 16))
train_loss = [9.2134, 8.5635, 7.9246, 7.2884, 6.7227, 6.1667, 9.0432, 8.4522, 8.0534, 7.7430, 7.5216, 9.2922, 8.9698, 8.9123, 8.8581]
val_loss = [1.1250, 1.0940, 1.0890, 1.1000, 1.1170, 1.1420, 1.1230, 1.1050, 1.1000, 1.0980, 1.1010, 1.1560, 1.1450, 1.1410, 1.1410]

# Setup plot
fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

# Colors
color_train = '#1f77b4'  # Professional blue
color_val = '#d62728'    # Professional red

# Plot Train Loss (Left Axis)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Train Loss (Average)', color=color_train, fontsize=12, fontweight='bold')
line1 = ax1.plot(epochs, train_loss, color=color_train, marker='o', linewidth=2, label='Train Loss (Average)')
ax1.tick_params(axis='y', labelcolor=color_train)
ax1.set_xticks(epochs)
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot Validation Loss (Right Axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Loss', color=color_val, fontsize=12, fontweight='bold')
line2 = ax2.plot(epochs, val_loss, color=color_val, marker='s', linewidth=2, linestyle='--', label='Validation Loss')
ax2.tick_params(axis='y', labelcolor=color_val)

# Highlight Best Epoch (Epoch 3)
best_epoch = 3
best_val_loss = val_loss[best_epoch - 1]
ax2.plot(best_epoch, best_val_loss, marker='*', color='#e377c2', markersize=14, linestyle='None', label='Best Val Loss (Epoch 3: 1.0890)')

# Combine legends from both axes
lines = line1 + line2 + [plt.Line2D([0], [0], marker='*', color='#e377c2', markersize=10, linestyle='None')]
labels = [l.get_label() for l in line1 + line2] + ['Best Val Loss (Epoch 3: 1.0890)']
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, shadow=True)

plt.tight_layout()

# Save plot
plt.savefig('loss_curve.png', bbox_inches='tight')
plt.savefig('loss_curve.pdf', bbox_inches='tight')
print("Clean loss curve plots saved successfully as loss_curve.png and loss_curve.pdf!")
