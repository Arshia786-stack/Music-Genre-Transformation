import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    def __init__(self):
        # Dataset config
        self.data_path = "/content/gtzan_dataset/genres_original"  # Update with your dataset path
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.num_genres = len(self.genres)
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.segment_duration = 3  # seconds
        self.segment_samples = self.sample_rate * self.segment_duration
        
        # Model config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epochs = 200
        self.lambda_cls = 1.0
        self.lambda_rec = 10.0
        self.lambda_gp = 10.0
        self.n_critic = 5
        
        # Checkpoint and sample paths
        self.checkpoint_dir = "checkpoints"
        self.sample_dir = "samples"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

config = Config()
print(f"Using device: {config.device}")

# Data preprocessing
class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config.genres)
        
    def audio_to_melspectrogram(self, audio_path):
        # Load audio file
        try:
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # Apply some basic preprocessing
            y = librosa.effects.preemphasis(y)
            
            # Handle variable length by taking segments
            segments = []
            for i in range(0, len(y), self.config.segment_samples):
                segment = y[i:i + self.config.segment_samples]
                if len(segment) == self.config.segment_samples:
                    segments.append(segment)
            
            # If no complete segments, pad the audio
            if not segments:
                if len(y) < self.config.segment_samples:
                    # Pad with zeros if shorter than segment length
                    y = np.pad(y, (0, self.config.segment_samples - len(y)))
                segments = [y[:self.config.segment_samples]]
            
            # Convert segments to mel spectrograms
            mel_spectrograms = []
            for segment in segments:
                # Convert to mel spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=segment,
                    sr=self.config.sample_rate,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    n_mels=self.config.n_mels
                )
                
                # Convert to log scale
                mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                # Normalize
                mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
                
                mel_spectrograms.append(mel_spectrogram)
            
            return mel_spectrograms
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def melspectrogram_to_audio(self, mel_spectrogram):
        # Scale back from normalized
        mel_db = mel_spectrogram * 80 - 80  # Approximate dB scale
        
        # Convert back to power
        mel_power = librosa.db_to_power(mel_db)
        
        # Inverse mel spectrogram
        y = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Apply noise reduction using spectral gating
        y = self._reduce_noise(y)
        
        return y

    def _reduce_noise(self, audio):
        # Simple spectral gating noise reduction
        # Better noise reduction would use specialized libraries or more complex techniques
        stft = librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        magnitude, phase = librosa.magphase(stft)
        
        # Estimate noise profile from the first 0.2 seconds
        noise_magnitude = np.mean(magnitude[:, :int(0.2 * self.config.sample_rate / self.config.hop_length)], axis=1)
        noise_magnitude = noise_magnitude.reshape(-1, 1)
        
        # Apply soft mask
        threshold = 2.0
        smoothed = magnitude - threshold * noise_magnitude
        mask = (smoothed > 0).astype(float)
        
        # Smooth mask transitions
        smoothed_mask = np.zeros_like(mask)
        for i in range(len(smoothed_mask)):
            smoothed_mask[i] = np.convolve(mask[i], np.hanning(11), mode='same')
        
        # Apply mask and reconstruct
        smoothed_stft = stft * smoothed_mask
        y_denoised = librosa.istft(smoothed_stft, hop_length=self.config.hop_length)
        
        return y_denoised
    
    def prepare_dataset(self):
        data = []
        
        # Process each genre folder
        for genre in self.config.genres:
            genre_folder = os.path.join(self.config.data_path, genre)
            if not os.path.exists(genre_folder):
                print(f"Warning: {genre_folder} does not exist!")
                continue
                
            files = [f for f in os.listdir(genre_folder) if f.endswith('.wav') or f.endswith('.au')]
            print(f"Processing {len(files)} files for genre: {genre}")
            
            for file in files:
                file_path = os.path.join(genre_folder, file)
                mel_spectrograms = self.audio_to_melspectrogram(file_path)
                
                if mel_spectrograms:
                    for mel_spec in mel_spectrograms:
                        genre_label = self.label_encoder.transform([genre])[0]
                        data.append({
                            'file_path': file_path,
                            'genre': genre,
                            'genre_label': genre_label,
                            'mel_spectrogram': mel_spec
                        })
        
        return data

# Dataset class
class MusicGenreDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        mel_spec = torch.FloatTensor(item['mel_spectrogram'])
        genre_label = torch.LongTensor([item['genre_label']])
        
        # Create one-hot encoded label
        one_hot_label = torch.zeros(config.num_genres)
        one_hot_label[item['genre_label']] = 1
        
        return mel_spec.unsqueeze(0), genre_label, one_hot_label

# StarGAN model components

# Generator
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        dim_in = 1  # Input channels (grayscale spectrogram)
        dim_out = 1  # Output channels (grayscale spectrogram)
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(dim_in + config.num_genres, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        
        # Downsampling layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        
        # Bottleneck layers - residual blocks
        self.bottleneck = nn.ModuleList([
            ResidualBlock(256) for _ in range(9)
        ])
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.upnorm1 = nn.InstanceNorm2d(128, affine=True)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.upnorm2 = nn.InstanceNorm2d(64, affine=True)
        
        # Output layer
        self.conv_out = nn.Conv2d(64, dim_out, kernel_size=7, stride=1, padding=3, bias=False)
        
    def forward(self, x, c):
        # c is one-hot vector of target domain (genre)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        
        # Concatenate input and domain information
        x = torch.cat([x, c], dim=1)
        
        # Initial convolution
        x = F.relu(self.norm1(self.conv1(x)))
        
        # Downsampling
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        
        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x)
        
        # Upsampling
        x = F.relu(self.upnorm1(self.upconv1(x)))
        x = F.relu(self.upnorm2(self.upconv2(x)))
        
        # Output
        x = torch.tanh(self.conv_out(x))
        
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        dim_in = 1  # Input channels (grayscale spectrogram)
        
        # Architecture based on PatchGAN
        self.conv1 = nn.Conv2d(dim_in, 64, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=True)
        
        # Output layers
        kernel_size = int(config.n_mels / 16)  # Adjust based on your mel spectrogram size
        
        # Source classification (real/fake)
        self.conv_src = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Domain classification (genre)
        self.conv_cls = nn.Conv2d(512, config.num_genres, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.norm2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.norm3(self.conv3(h)), 0.2)
        h = F.leaky_relu(self.norm4(self.conv4(h)), 0.2)
        
        # Source output
        src_out = self.conv_src(h)
        
        # Class output
        cls_out = self.conv_cls(h)
        cls_out = cls_out.view(cls_out.size(0), cls_out.size(1))
        
        return src_out, cls_out

# Genre Classifier for evaluation
class GenreClassifier(nn.Module):
    def __init__(self, config):
        super(GenreClassifier, self).__init__()
        self.config = config
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Calculate input size for the fully connected layer
        fc_input_size = self._get_fc_input_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, config.num_genres)
        
    def _get_fc_input_size(self):
        # Calculate the size of the flattened feature map
        # This depends on your input size and pooling layers
        h, w = config.n_mels, (config.segment_samples // config.hop_length) + 1
        h, w = h // 2, w // 2  # After pool1
        h, w = h // 2, w // 2  # After pool2
        h, w = h // 2, w // 2  # After pool3
        h, w = h // 2, w // 2  # After pool4
        return 512 * h * w
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))
        x = F.relu(self.bn3(self.pool3(self.conv3(x))))
        x = F.relu(self.bn4(self.pool4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# StarGAN Trainer
class StarGANTrainer:
    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize models
        self.G = Generator(config).to(config.device)
        self.D = Discriminator(config).to(config.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        
        # Initialize the classifier for evaluation
        self.classifier = GenreClassifier(config).to(config.device)
        self.cls_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.cls_criterion = nn.CrossEntropyLoss()
        
        # For evaluation
        self.audio_processor = AudioProcessor(config)
        
    def train_classifier(self, epochs=10):
        print("Training genre classifier for evaluation...")
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for i, (x_real, y_real, _) in enumerate(tqdm(self.train_loader, desc=f"Classifier Epoch {epoch+1}/{epochs}")):
                x_real = x_real.to(self.config.device)
                y_real = y_real.squeeze().to(self.config.device)
                
                # Forward pass
                self.cls_optimizer.zero_grad()
                outputs = self.classifier(x_real)
                loss = self.cls_criterion(outputs, y_real)
                
                # Backward and optimize
                loss.backward()
                self.cls_optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y_real.size(0)
                correct += (predicted == y_real).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Classifier - Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}%")
            
            # Validation
            self.classifier.eval()
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                for x_val, y_val, _ in self.val_loader:
                    x_val = x_val.to(self.config.device)
                    y_val = y_val.squeeze().to(self.config.device)
                    
                    outputs = self.classifier(x_val)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_val.size(0)
                    val_correct += (predicted == y_val).sum().item()
                
                val_accuracy = 100 * val_correct / val_total
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save the trained classifier
        torch.save(self.classifier.state_dict(), os.path.join(self.config.checkpoint_dir, "genre_classifier.pth"))
    
    def train(self):
        # Initialize losses for tracking
        g_losses = []
        d_losses = []
        
        for epoch in range(self.config.epochs):
            self.G.train()
            self.D.train()
            
            d_loss_avg = 0
            g_loss_avg = 0
            
            for i, (x_real, y_real, y_org) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")):
                # Prepare data
                x_real = x_real.to(self.config.device)  # Real spectrograms
                y_org = y_org.to(self.config.device)    # Original domain labels
                
                # Generate target domain labels randomly
                rand_idx = torch.randperm(y_org.size(0))
                y_trg = y_org[rand_idx]
                
                # -----------------------
                # Train Discriminator
                # -----------------------
                
                # Reset gradients
                self.d_optimizer.zero_grad()
                
                # Compute loss with real spectrograms - source
                out_src, out_cls = self.D(x_real)
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = self.cls_criterion(out_cls, y_real.squeeze())
                
                # Generate fake spectrograms
                x_fake = self.G(x_real, y_trg)
                
                # Compute loss with fake spectrograms - source
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)
                
                # Compute gradient penalty
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.config.device)
                x_hat = (alpha * x_real + (1 - alpha) * x_fake.detach()).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self._gradient_penalty(out_src, x_hat)
                
                # Compute total D loss
                d_loss = d_loss_real + d_loss_fake + d_loss_cls * self.config.lambda_cls + d_loss_gp * self.config.lambda_gp
                
                # Backward and optimize
                d_loss.backward()
                self.d_optimizer.step()
                
                d_loss_avg += d_loss.item()
                
                # -----------------------
                # Train Generator
                # -----------------------
                
                # Skip generator update based on n_critic
                if (i + 1) % self.config.n_critic == 0:
                    # Reset gradients
                    self.g_optimizer.zero_grad()
                    
                    # Generate fake spectrograms
                    x_fake = self.G(x_real, y_trg)
                    
                    # Compute adversarial loss
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = self.cls_criterion(out_cls, y_real.squeeze())
                    
                    # Original-to-Original (identity mapping)
                    x_rec = self.G(x_fake, y_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_rec))
                    
                    # Compute total G loss
                    g_loss = g_loss_fake + g_loss_cls * self.config.lambda_cls + g_loss_rec * self.config.lambda_rec
                    
                    # Backward and optimize
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    g_loss_avg += g_loss.item()
                
            # Log the average loss for the epoch
            d_loss_avg /= len(self.train_loader)
            g_loss_avg /= (len(self.train_loader) // self.config.n_critic)
            
            d_losses.append(d_loss_avg)
            g_losses.append(g_loss_avg)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, D Loss: {d_loss_avg:.4f}, G Loss: {g_loss_avg:.4f}")
            
            # Save model checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save(self.G.state_dict(), os.path.join(self.config.checkpoint_dir, f"generator_{epoch+1}.pth"))
                torch.save(self.D.state_dict(), os.path.join(self.config.checkpoint_dir, f"discriminator_{epoch+1}.pth"))
                
                # Generate and save samples
                self.sample_spectrograms(epoch + 1)
                
                # Plot losses
                self._plot_losses(d_losses, g_losses, epoch + 1)
                
        # Save final models
        torch.save(self.G.state_dict(), os.path.join(self.config.checkpoint_dir, "generator_final.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.config.checkpoint_dir, "discriminator_final.pth"))
    
    def _gradient_penalty(self, out, x):
        # Compute gradient penalty for WGAN-GP
        batch_size = x.size(0)
        grad_dout = torch.autograd.grad(
            outputs=out.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        grad_dout = grad_dout.view(batch_size, -1)
        grad_norm = torch.sqrt(torch.sum(grad_dout ** 2, dim=1) + 1e-12)
        
        return torch.mean((grad_norm - 1) ** 2)
    
    def sample_spectrograms(self, epoch):
        """Generate and save sample spectrograms from all genres to all genres"""
        self.G.eval()
        
        # Create folder for this epoch samples
        sample_dir = os.path.join(self.config.sample_dir, f"epoch_{epoch}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get one example from each genre
        with torch.no_grad():
            samples = {}
            for i, (x, y, y_onehot) in enumerate(self.val_loader):
                genre_idx = y.item()
                if genre_idx not in samples:
                    samples[genre_idx] = (x.to(self.config.device), y_onehot.to(self.config.device))
                
                if len(samples) == config.num_genres:
                    break
            
            # Generate transformations for each source-target pair
            for src_idx, (src_x, _) in samples.items():
                for trg_idx in range(config.num_genres):
                    # Create target domain label
                    trg_c = torch.zeros(1, config.num_genres).to(self.config.device)
                    trg_c[0, trg_idx] = 1.0
                    
                    # Generate transformed spectrogram
                    fake_x = self.G(src_x, trg_c)
                    
                    # Convert tensors to numpy for visualization
                    real_np = src_x.cpu().numpy().squeeze()
                    fake_np = fake_x.cpu().numpy().squeeze()
                    
                    # Plot and save
                    self._plot_spectrogram_comparison(
                        real_np, fake_np,
                        os.path.join(sample_dir, f"{config.genres[src_idx]}_to_{config.genres[trg_idx]}.png"),
                        f"{config.genres[src_idx]} to {config.genres[trg_idx]}"
                    )
                    
                    # Save audio (if it's a validation sample)
                    if epoch % 50 == 0:
                        # Convert spectrogram back to audio
                        fake_audio = self.audio_processor.melspectrogram_to_audio(fake_np)
                        
                        # Save audio file
                        sf.write(
                            os.path.join(sample_dir, f"{config.genres[src_idx]}_to_{config.genres[trg_idx]}.wav"),
                            fake_audio,
                            self.config.sample_rate
                        )
    
    def _plot_spectrogram_comparison(self, real_spec, fake_spec, save_path, title):
        """Plot real and fake spectrograms side by side"""
        plt.figure(figsize=(12, 6))
        
        # Plot real spectrogram
        plt.subplot(1, 2, 1)
        plt.title("Original")
        librosa.display.specshow(
            real_spec,
            y_axis='mel',
            x_axis='time',
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length
        )
        plt.colorbar(format='%+2.0f dB')
        
        # Plot fake spectrogram
        plt.subplot(1, 2, 2)
        plt.title("Transformed")
        librosa.display.specshow(
            fake_spec,
            y_axis='mel',
            x_axis='time',
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length
        )
        plt.colorbar(format='%+2.0f dB')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_losses(self, d_losses, g_losses, epoch):
        """Plot discriminator and generator losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.title(f'Training Losses up to Epoch {epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.checkpoint_dir, f'losses_epoch_{epoch}.png'))
        plt.close()
    
    def evaluate(self):
        """Evaluate the trained model"""
        self.G.eval()
        self.classifier.eval()
        
        # Load the best models if not already loaded
        if os.path.exists(os.path.join(self.config.checkpoint_dir, "generator_final.pth")):
            self.G.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, "generator_final.pth"), map_location=self.config.device))
        
        if os.path.exists(os.path.join(self.config.checkpoint_dir, "genre_classifier.pth")):
            self.classifier.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, "genre_classifier.pth"), map_location=self.config.device))
        
        # Evaluation directory
        eval_dir = os.path.join(self.config.sample_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Collect results for all genre transformations
        results = {
            'source_genre': [],
            'target_genre': [],
            'predicted_genre': [],
            'confidence': []
        }
        
        # Generate transformed samples for each source-target pair
        with torch.no_grad():
            for i, (x_real, y_real, y_org) in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                if i >= 100:  # Limit to 100 samples for evaluation
                    break
                
                x_real = x_real.to(self.config.device)
                y_real = y_real.to(self.config.device)
                y_org = y_org.to(self.config.device)
                
                source_genre = self.config.genres[y_real.item()]
                
                # Transform to each target genre
                for trg_idx in range(self.config.num_genres):
                    # Skip if source and target are the same
                    if trg_idx == y_real.item():
                        continue
                    
                    # Create target domain label
                    y_trg = torch.zeros_like(y_org)
                    y_trg[0, trg_idx] = 1.0
                    
                    # Generate transformed spectrogram
                    x_fake = self.G(x_real, y_trg)
                    
                    # Classify the generated spectrogram
                    cls_output = self.classifier(x_fake)
                    probs = F.softmax(cls_output, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_idx].item()
                    
                    target_genre = self.config.genres[trg_idx]
                    predicted_genre = self.config.genres[pred_idx]
                    
                    # Save results
                    results['source_genre'].append(source_genre)
                    results['target_genre'].append(target_genre)
                    results['predicted_genre'].append(predicted_genre)
                    results['confidence'].append(confidence)
                    
                    # Save some examples
                    if i < 5:
                        # Convert to audio and save
                        fake_audio = self.audio_processor.melspectrogram_to_audio(x_fake.cpu().numpy().squeeze())
                        sf.write(
                            os.path.join(eval_dir, f"sample_{i}_{source_genre}_to_{target_genre}.wav"),
                            fake_audio,
                            self.config.sample_rate
                        )
                        
                        # Plot spectrograms
                        self._plot_spectrogram_comparison(
                            x_real.cpu().numpy().squeeze(),
                            x_fake.cpu().numpy().squeeze(),
                            os.path.join(eval_dir, f"sample_{i}_{source_genre}_to_{target_genre}.png"),
                            f"{source_genre} to {target_genre}"
                        )
        
        # Create a DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Analysis 1: Success rate per target genre
        success_rate = df.groupby('target_genre').apply(
            lambda x: (x['predicted_genre'] == x['target_genre']).mean()
        ).reset_index(name='success_rate')
        
        print("Success rate per target genre:")
        print(success_rate)
        
        # Plot success rate
        plt.figure(figsize=(12, 6))
        sns.barplot(x='target_genre', y='success_rate', data=success_rate)
        plt.title('Transformation Success Rate by Target Genre')
        plt.xlabel('Target Genre')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, 'success_rate_by_target.png'))
        plt.close()
        
        # Analysis 2: Confusion matrix
        cm = confusion_matrix(df['target_genre'], df['predicted_genre'], labels=self.config.genres)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.config.genres, yticklabels=self.config.genres)
        plt.title('Confusion Matrix of Target vs Predicted Genres')
        plt.xlabel('Predicted Genre')
        plt.ylabel('Target Genre')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Analysis 3: Source to target success rate heatmap
        pivot_df = df.pivot_table(
            index='source_genre',
            columns='target_genre',
            values='confidence',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='viridis')
        plt.title('Average Confidence Score: Source to Target Transformation')
        plt.xlabel('Target Genre')
        plt.ylabel('Source Genre')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, 'source_to_target_confidence.png'))
        plt.close()
        
        # Save detailed results
        df.to_csv(os.path.join(eval_dir, 'evaluation_results.csv'), index=False)
        
        return df
    
# Music Genre Transformation Application
class MusicGenreTransformer:
    def __init__(self, config):
        self.config = config
        self.G = Generator(config).to(config.device)
        self.audio_processor = AudioProcessor(config)
        
        # Load trained model
        model_path = os.path.join(config.checkpoint_dir, "generator_final.pth")
        if os.path.exists(model_path):
            self.G.load_state_dict(torch.load(model_path, map_location=config.device))
            self.G.eval()
            print(f"Loaded generator from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
    
    def transform_audio(self, input_path, target_genre, output_path):
        """Transform audio from one genre to another"""
        # Make sure target genre is valid
        if target_genre not in self.config.genres:
            raise ValueError(f"Target genre {target_genre} not in list of genres: {self.config.genres}")
        
        # Convert audio to mel spectrogram
        mel_spectrograms = self.audio_processor.audio_to_melspectrogram(input_path)
        
        if not mel_spectrograms:
            raise ValueError(f"Failed to process audio file: {input_path}")
        
        # Prepare target domain label
        target_idx = self.config.genres.index(target_genre)
        target_label = torch.zeros(1, self.config.num_genres).to(self.config.device)
        target_label[0, target_idx] = 1.0
        
        # Transform each segment
        transformed_segments = []
        
        with torch.no_grad():
            for mel_spec in mel_spectrograms:
                # Convert to tensor
                mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.config.device)
                
                # Transform
                fake_mel = self.G(mel_tensor, target_label)
                
                # Convert back to numpy
                fake_mel_np = fake_mel.squeeze().cpu().numpy()
                
                # Convert to audio
                audio_segment = self.audio_processor.melspectrogram_to_audio(fake_mel_np)
                transformed_segments.append(audio_segment)
        
        # Concatenate audio segments
        if transformed_segments:
            transformed_audio = np.concatenate(transformed_segments)
            
            # Save to output path
            sf.write(output_path, transformed_audio, self.config.sample_rate)
            print(f"Transformed audio saved to {output_path}")
            
            return output_path
        else:
            raise ValueError("No audio segments were transformed")
    
    def process_batch(self, input_dir, output_dir, target_genre=None, visualize=False):
        """Process a batch of audio files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.au', '.flac')):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            file_name = os.path.basename(audio_file)
            file_base = os.path.splitext(file_name)[0]
            
            # If target genre is specified, transform to that genre
            if target_genre:
                output_file = os.path.join(output_dir, f"{file_base}_to_{target_genre}.wav")
                try:
                    self.transform_audio(audio_file, target_genre, output_file)
                    
                    if visualize:
                        # Create visualization
                        orig_spec = self.audio_processor.audio_to_melspectrogram(audio_file)[0]
                        trans_spec = self.audio_processor.audio_to_melspectrogram(output_file)[0]
                        
                        plt.figure(figsize=(12, 6))
                        
                        plt.subplot(1, 2, 1)
                        librosa.display.specshow(
                            orig_spec,
                            y_axis='mel',
                            x_axis='time',
                            sr=self.config.sample_rate,
                            hop_length=self.config.hop_length
                        )
                        plt.title("Original")
                        plt.colorbar(format='%+2.0f dB')
                        
                        plt.subplot(1, 2, 2)
                        librosa.display.specshow(
                            trans_spec,
                            y_axis='mel',
                            x_axis='time',
                            sr=self.config.sample_rate,
                            hop_length=self.config.hop_length
                        )
                        plt.title(f"Transformed to {target_genre}")
                        plt.colorbar(format='%+2.0f dB')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"{file_base}_to_{target_genre}.png"))
                        plt.close()
                
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
            
            # Otherwise, transform to all genres
            else:
                for genre in self.config.genres:
                    output_file = os.path.join(output_dir, f"{file_base}_to_{genre}.wav")
                    try:
                        self.transform_audio(audio_file, genre, output_file)
                    except Exception as e:
                        print(f"Error transforming {audio_file} to {genre}: {e}")

# Main execution
def main():
    # Initialize configuration
    config = Config()
    
    # Initialize audio processor
    audio_processor = AudioProcessor(config)
    
    # Check if we have a processed dataset
    save_path = "processed_data.pkl"
    
    if os.path.exists(save_path):
        print("Loading preprocessed data...")
        data = torch.load(save_path)
    else:
        print("Preprocessing dataset...")
        data = audio_processor.prepare_dataset()
        torch.save(data, save_path)
    
    print(f"Dataset size: {len(data)} segments")
    
    # Split dataset into train and validation sets
    random.shuffle(data)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train set: {len(train_data)}, Validation set: {len(val_data)}")
    
    # Create datasets and data loaders
    train_dataset = MusicGenreDataset(train_data)
    val_dataset = MusicGenreDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize trainer
    trainer = StarGANTrainer(config, train_loader, val_loader)
    
    # Train the classifier for evaluation
    trainer.train_classifier()
    
    # Train the StarGAN model
    trainer.train()
    
    # Evaluate the model
    results = trainer.evaluate()
    
    print("Training and evaluation complete!")
    
    # Create an example transformer application
    transformer = MusicGenreTransformer(config)
    
    # Example usage (uncomment to use):
    # input_file = "path/to/input.wav"
    # output_file = "path/to/output.wav"
    # target_genre = "jazz"
    # transformer.transform_audio(input_file, target_genre, output_file)
    
    # Or process a batch of files:
    # input_dir = "path/to/input/directory"
    # output_dir = "path/to/output/directory"
    # transformer.process_batch(input_dir, output_dir, target_genre="jazz", visualize=True)

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Music Genre Transformation System")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "transform"], help="Mode: train or transform")
    parser.add_argument("--input", type=str, help="Input audio file or directory for transformation")
    parser.add_argument("--output", type=str, help="Output file or directory for transformed audio")
    parser.add_argument("--target_genre", type=str, help="Target genre for transformation")
    parser.add_argument("--data_path", type=str, help="Path to GTZAN dataset")
    parser.add_argument("--batch", action="store_true", help="Process a batch of files")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations for transformations")
    
    args = parser.parse_args()
    
    config = Config()
    
    # Update data path if provided
    if args.data_path:
        config.data_path = args.data_path
    
    if args.mode == "train":
        main()
    elif args.mode == "transform":
        transformer = MusicGenreTransformer(config)
        
        if not args.input:
            print("Error: Input file or directory required for transformation mode")
            parser.print_help()
            exit(1)
        
        if args.batch:
            if not args.output:
                print("Error: Output directory required for batch processing")
                parser.print_help()
                exit(1)
            
            transformer.process_batch(args.input, args.output, args.target_genre, args.visualize)
        else:
            if not args.output or not args.target_genre:
                print("Error: Output file and target genre required for single file transformation")
                parser.print_help()
                exit(1)
            
            transformer.transform_audio(args.input, args.target_genre, args.output)
    else:
        parser.print_help()
