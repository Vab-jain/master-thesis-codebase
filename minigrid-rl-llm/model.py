import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_rgb=False, use_hints=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.use_rgb = use_rgb
        self.use_hints = use_hints

        # Define image embedding
        if not use_rgb:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2),padding=1 if obs_space['image'][0] < 7 else 0),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
            n = obs_space["image"][0]
            m = obs_space["image"][1]
            self.image_embedding_size = max(((n-1)//2-2)*((m-1)//2-2)*64, 64)
        else:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 32, (8, 8),padding=0, stride=4),
                nn.ReLU(),
                # nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, (4, 4), padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), padding=1, stride=1),
                nn.ReLU()
            )
            H = obs_space["image"][0]
            W = obs_space["image"][1]
            # Conv Layer 1
            H1 = (H-8) // 4 + 1
            W1 = (W-8) // 4 + 1
            # Conv Layer 2
            H2 = (H1-4+2) // 2 + 1
            W2 = (W1-4+2) // 2 + 1
            # Conv Layer 3
            H3 = (H2-3+2) // 1 + 1
            W3 = (W2-3+2) // 1 + 1
            self.image_embedding_size = max(H3*W3*64, 64)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Define hint embedding
        if self.use_hints:
            # Hint embedding: convert hint integer to embedding
            self.hint_embedding_size = 32
            # For subgoals: 8 possible values (0-7), for actions: 7 possible values (0-6)
            # Check if hint field exists and get max value
            if 'hint' in obs_space:
                if hasattr(obs_space['hint'], 'n'):
                    max_hint_value = obs_space['hint'].n
                else:
                    # obs_space['hint'] is already an integer
                    max_hint_value = obs_space['hint']
            else:
                max_hint_value = 8  # Default for subgoals
            
            # Ensure we have at least 8 values for subgoals or 7 for actions
            max_hint_value = max(max_hint_value, 8)
            
            self.hint_embedding = nn.Embedding(max_hint_value, self.hint_embedding_size)
            # Availability embedding: whether hint is available (0 or 1)
            self.hint_availability_embedding = nn.Embedding(2, self.hint_embedding_size // 2)
            self.hint_embedding_size = self.hint_embedding_size + self.hint_embedding_size // 2

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size
        if self.use_hints:
            self.embedding_size += self.hint_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        if self.use_hints:
            # Handle hints from preprocessed observations
            if hasattr(obs, 'hint') and hasattr(obs, 'hint_available'):
                # Hints are available as separate fields
                hint_data = {
                    'hint': obs.hint,
                    'hint_available': obs.hint_available
                }
            else:
                # No hints available - use neutral "none" values instead of 0 to avoid confusion
                # Use 7 ("none") as the neutral value since we ensure max_hint_value >= 8 in __init__
                hint_data = {
                    'hint': torch.full((obs.image.shape[0],), 7, dtype=torch.long, device=obs.image.device),  # 7 = "none"
                    'hint_available': torch.zeros(obs.image.shape[0], dtype=torch.long, device=obs.image.device)
                }
            
            hint_embedding = self._get_hint_embedding(hint_data)
            embedding = torch.cat((embedding, hint_embedding), dim=1)

        # Get action logits
        logits = self.actor(embedding)
        # Pass raw (unnormalised) logits to the categorical distribution – it will
        # apply log-softmax internally. Feeding already log-softmaxed values would
        # distort the probabilities and harm learning.
        dist = Categorical(logits=logits)

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def _get_hint_embedding(self, hint):
        """Process hint information and return embedding."""
        if not self.use_hints:
            return torch.zeros(self.hint_embedding_size)
        
        # Extract hint value and availability
        hint_value = hint['hint'] if isinstance(hint, dict) else hint
        hint_available = hint['hint_available'] if isinstance(hint, dict) else 1
        
        # Ensure tensors are on the correct device and have correct shape
        if not isinstance(hint_value, torch.Tensor):
            hint_value = torch.tensor(hint_value, dtype=torch.long)
        if not isinstance(hint_available, torch.Tensor):
            hint_available = torch.tensor(hint_available, dtype=torch.long)
        
        # Get embeddings
        hint_emb = self.hint_embedding(hint_value)
        availability_emb = self.hint_availability_embedding(hint_available)
        
        # Concatenate hint value and availability embeddings
        combined_emb = torch.cat([hint_emb, availability_emb], dim=-1)
        
        # Zero out the hint embedding when hint is not available (hint_available == 0)
        # This ensures the model clearly distinguishes between "real hint" and "no hint"
        hint_available_mask = hint_available.float().unsqueeze(-1)  # Shape: (batch_size, 1)
        combined_emb = combined_emb * hint_available_mask  # Zero out when hint_available == 0
        
        return combined_emb
