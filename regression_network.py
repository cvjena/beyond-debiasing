import torch
from torch import nn
from contextual_decomposition import get_cd_1d_by_modules
from mixed_cmi_estimator import mixed_cmi_model

class RegressionNetwork(nn.Module):
    def __init__(self, input_shape, n_hidden_layers=2, hidden_dim_size=32, device='cpu'):
        super().__init__()
        self.device = device
        
        # The network always has at least one hidden layer (input_shape -> 32).
        # Make sure that n_hidden_layers is valid.
        if n_hidden_layers < 1:
            raise ValueError("The network cannot have less than 1 hidden layer.")

        # Generate and initialize hidden layers.
        # Note: we only need to generate n_hidden_layers-1 hidden layers!
        lin_layers = [nn.Linear(in_features=hidden_dim_size, out_features=hidden_dim_size)] *  (n_hidden_layers - 1)
        for lin_layer in lin_layers:
            nn.init.xavier_uniform_(lin_layer.weight)
            nn.init.zeros_(lin_layer.bias)
        relus = [nn.ReLU()] * len(lin_layers)

        # Generate and intialize first and last layer.
        input_layer = nn.Linear(input_shape, hidden_dim_size)
        output_layer = nn.Linear(hidden_dim_size, 1)
        for lin_layer in [input_layer, output_layer]:
            nn.init.xavier_uniform_(lin_layer.weight)
            nn.init.zeros_(lin_layer.bias)

        # Combine layers to a model.
        modules = [ nn.Flatten(),
                    input_layer,
                    nn.ReLU(),
                    *[z for tuple in zip(lin_layers, relus) for z in tuple],
                    output_layer,
                  ]
        self.layers = nn.Sequential(*modules)
        self.to(device)

    def forward(self, x):
        return self.layers(x)

    def feat_steering_loss(self, inputs, targets, outputs, feat_steering_config=None):
        # Get configuration for feature steering.
        # Do not perform feature steering if it is not desired.
        if feat_steering_config["steering_mode"] == "none":
            return torch.tensor(0.0)
        elif not feat_steering_config["steering_mode"] in ["loss_l1", "loss_l2"]:
            raise ValueError("The feature steering mode is invalid.")
        if not feat_steering_config["attrib_mode"] in ["contextual_decomposition", "cmi"]:
            raise ValueError("The feature attribution mode is invalid.")
        feat_to_encourage, feat_to_discourage = feat_steering_config["encourage"], feat_steering_config["discourage"]

        # Feature attribution.
        if feat_steering_config["attrib_mode"] == "contextual_decomposition":
            scores_feat_to_encourage, _ = get_cd_1d_by_modules(self, self.layers, inputs, feat_to_encourage, device=self.device)
            scores_feat_to_discourage, _ = get_cd_1d_by_modules(self, self.layers, inputs, feat_to_discourage, device=self.device)
        
        elif feat_steering_config["attrib_mode"] == "cmi":
            # Estimate CMI.
            if len(feat_to_encourage) > 0:
                scores_feat_to_encourage = torch.stack([mixed_cmi_model(inputs[:,feat], outputs, targets, feature_is_categorical=False, target_is_categorical=False) for feat in feat_to_encourage], 0)
                # scores_feat_to_encourage = torch.stack([get_continuous_cmi(inputs[:,feat], outputs, z=targets, knn=0.2, seed=42) for feat in feat_to_encourage], 0)
            else:
                scores_feat_to_encourage = torch.tensor([]).float()
            if len(feat_to_discourage) > 0:
                scores_feat_to_discourage = torch.stack([mixed_cmi_model(inputs[:,feat], outputs, targets, feature_is_categorical=False, target_is_categorical=False) for feat in feat_to_discourage], 0)
                # scores_feat_to_discourage = torch.stack([get_continuous_cmi(inputs[:,feat], outputs, z=targets, knn=0.2, seed=42) for feat in feat_to_discourage], 0)
            else:
                scores_feat_to_discourage = torch.tensor([]).float()

            # Transform to [0,1].
            # NOTE: Even though in theory CMI >= 0, in practice our estimates
            # can be smaler than zero.
            # Make sure that sqrt does not receive values < 0. In analogy to the 
            # Straight-Through Estimators (STEs) we apply our transformation only
            # to inputs >= 0 and use the identity transformation for inputs < 0.
            scores_feat_to_encourage[scores_feat_to_encourage > 0] = torch.sqrt(1 - torch.exp(-2*scores_feat_to_encourage[scores_feat_to_encourage > 0]))
            scores_feat_to_discourage[scores_feat_to_discourage > 0] = torch.sqrt(1 - torch.exp(-2*scores_feat_to_discourage[scores_feat_to_discourage > 0]))

        else:
            raise NotImplementedError("The selected feature attribution mode is not yet implemented!")

        # Small corrections:
        # If there are no features to en- or discourage, we can explicitly set their contribution to 0.
        if len(feat_to_encourage) == 0:
            scores_feat_to_encourage = torch.tensor(0)
        if len(feat_to_discourage) == 0:
            scores_feat_to_discourage = torch.tensor(0)
        
        # Feature steering.
        if feat_steering_config["attrib_mode"] == "cmi":
            # With the CMI estimates we can have negative values even though in theory CMI >= 0.
            # L1 / L2 would emphasize them, but we want values < 0 to result in a smaller loss.
            # L1-Loss:
            #   We know that our values should be almost > 0. Therefore, we apply the absolute value
            #   only to values >= 0 and the identity transformation to all others (analogous to
            #   Straight-Through Estimators, keeps gradients).
            #   In practice, this results in ignoring the absolute value.
            # 
            # L2-Loss:
            #   Here, we also only square for values >= 0 and the identity transformation to all
            #   others.
            if feat_steering_config["steering_mode"] == "loss_l2":
                scores_feat_to_encourage[scores_feat_to_encourage >= 0] = torch.square(scores_feat_to_encourage[scores_feat_to_encourage >= 0])
                scores_feat_to_discourage[scores_feat_to_discourage >= 0] = torch.square(scores_feat_to_discourage[scores_feat_to_discourage >= 0])
            return feat_steering_config["lambda"] * (torch.sum(scores_feat_to_discourage) - torch.sum(scores_feat_to_encourage)) / inputs.size()[0] # Average over Batch

        if feat_steering_config["lambda"] == 0:
            return torch.tensor(0.0)
        elif feat_steering_config["steering_mode"] == "loss_l1":
            feat_steering_loss = feat_steering_config["lambda"] * (torch.sum(torch.abs(scores_feat_to_discourage)) - torch.sum(torch.abs(scores_feat_to_encourage)))
        elif feat_steering_config["steering_mode"] == "loss_l2":
            feat_steering_loss = feat_steering_config["lambda"] * (torch.sum(torch.square(scores_feat_to_discourage)) - torch.sum(torch.square(scores_feat_to_encourage)))
        else:
            raise NotImplementedError("The selected feature steering mode is not yet implemented!")
        
        return feat_steering_loss / inputs.size()[0] # Average over Batch
    
    def loss(self, inputs, targets, outputs, feat_steering_config=None):
        # For MSE make sure that outputs is a 1D tensor. That is, we need to
        # prevent tensors of shape torch.Size([batch_size, 1]).
        if len(outputs.size()) > 1:
            outputs = outputs.squeeze(axis=1)

        # Compute default loss.
        loss_func = nn.MSELoss()
        loss = loss_func(outputs, targets)

        # No feature steering if in evaluation mode or explicitly specified.
        if not self.training or feat_steering_config["steering_mode"] == "none":
            return loss
        else:
            feat_steering_loss = self.feat_steering_loss(inputs, targets, outputs, feat_steering_config=feat_steering_config)
            if feat_steering_loss.isnan():
                raise ValueError("The feature steering loss of your model is nan. Thus, no reasonable gradient can be computed! \
                                 The feature steering config was: " + str(feat_steering_config) + ".")
            return loss + feat_steering_loss
        
    def train(self, train_dataloader, feat_steering_config, epochs=90, learning_rate=0.01):
        optimizer = torch.optim.AdamW(self.layers.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0

            for inputs, targets in train_dataloader:
                # Pass data to GPU / CPU if necessary.
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the gradients.
                optimizer.zero_grad()

                # Perform forward pass.
                outputs = self(inputs)
                if outputs.isnan().any():
                    raise ValueError("The output of the model contains nan. Thus, no \
                                    reasonable loss can be computed!")

                # Calculate loss.
                loss = self.loss(inputs, targets, outputs, feat_steering_config=feat_steering_config)

                # Perform backward pass and modify weights accordingly.
                if loss == torch.inf:
                    raise ValueError("The loss of your model is inf. Thus, no reasonable gradient can be computed!")
                if loss.isnan():
                    raise ValueError("The loss of your model is nan. Thus, no reasonable gradient can be computed!")
            
                loss.backward()
                optimizer.step()

                # Print statistics.
                epoch_loss += loss.item()
            print("Loss (per sample) after epoch " + str(epoch+1) + ": " + str(epoch_loss / len(train_dataloader)))
        
    
