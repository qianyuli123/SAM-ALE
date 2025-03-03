import torch
import torch.nn as nn

class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # (3, 1024, 1024) -> (64, 512, 512)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # (64, 512, 512) -> (128, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # (128, 256, 256) -> (256, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=1),  # (256, 128, 128) -> (384, 64, 64)
            nn.ReLU()
        )
        # Add LayerNorm for normalization
        self.norm = nn.LayerNorm(384)  # Normalize across spatial and channel dimensions

    def forward(self, x):
        # Pass through CNN
        features = self.feature_extractor(x)  # (6, 384, 64, 64)
        # Reshape to (batch_size, height*width, channels)
        batch_size, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1)  # (6, 64, 64, 384)
        features = features.reshape(batch_size, height * width, channels)  # (6, 4096, 384)
        # Apply normalization
        normalized_features = self.norm(features)  # (6, 4096, 384)
        return normalized_features

if __name__ == "__main__":
    # Instantiate the model
    model = FeatureExtractorCNN()

    print(model)
    # Generate random input data (batch of 6 images, 3 channels, 1024x1024 resolution)
    input_data = torch.randn(6, 3, 1024, 1024)

    # Forward pass through the model
    output_features = model(input_data)

    # Print the shape of the output features
    print("Input shape:", input_data.shape)
    print("Output shape:", output_features.shape)
